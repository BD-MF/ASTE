# encoding=utf-8

import os
import math
import argparse
import random
import torch
import torch.nn as nn
import numpy as np
from bucket_iterator import BucketIterator
from data_utils import ASTEDataReader, build_tokenizer, build_embedding_matrix, Tokenizer
from models.pointerGeneratorSeq2Seq import PoGenSeq2Seq4ASTE


class Instructor:
    def __init__(self, config):
        self.config = config

        aste_data_reader = ASTEDataReader(data_dir=config.data_dir)
        tokenizer = build_tokenizer(data_dir=config.data_dir)
        target_vocab = Tokenizer()
        embedding_matrix = build_embedding_matrix(
            config.data_dir, tokenizer.word2idx, config.emb_dim, config.dataset)

        embedding_matrix = torch.tensor(
            embedding_matrix, dtype=torch.float).to(config.device)

        self.train_data_loader = BucketIterator(data=aste_data_reader.get_train(
            tokenizer), batch_size=config.batch_size, shuffle=True, tokenizer=tokenizer, max_decode=config.tgt_max_train, device=config.device)

        self.dev_data_loader = BucketIterator(data=aste_data_reader.get_dev(
            tokenizer), batch_size=config.batch_size, shuffle=False, tokenizer=tokenizer, max_decode=config.tgt_max_test, device=config.device)

        self.test_data_loader = BucketIterator(data=aste_data_reader.get_test(
            tokenizer), batch_size=config.batch_size, shuffle=False, tokenizer=tokenizer, max_decode=config.tgt_max_test, device=config.device)

        self.model = config.model_class(
            embedding_matrix=embedding_matrix, config=config, vocab=tokenizer)
        self.model = self.model.to(config.device)

        if torch.cuda.is_available():
            print('>>> cuda memory allocated:',
                  torch.cuda.memory_allocated(device=config.device.index))

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('>>> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(
            n_trainable_params, n_nontrainable_params))
        print('>>> training arguments:')
        for arg in vars(self.config):
            print('>>> {0}: {1}'.format(arg, getattr(self.config, arg)))

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.config.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, optimizer):
        max_dev_f1 = 0
        best_state_dict_path = ''
        global_step = 0
        continue_not_increase = 0
        for epoch in range(self.config.num_epochs):
            print('>'*30)
            print('epoch: {0}'.format(epoch+1))
            increase_flag = False

            for i_batch, sample_batched in enumerate(self.train_data_loader):

                global_step += 1
                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()

                loss = self.model.training_step(sample_batched, i_batch)

                # import pdb
                # pdb.set_trace()

                loss.backward()
                optimizer.step()

                if global_step % self.config.log_step == 0:
                    vali_pred = self.model.validation_step(
                        sample_batched, i_batch)
                    rouge_1, rouge_2, rouge_l, val_loss = vali_pred['rouge_1'], vali_pred[
                        'rouge_2'], vali_pred['rouge_l'], vali_pred['val_loss']

                    print('rouge_1: {:.4f}, rouge_2: {:.4f}, rouge_l: {:.4f}, val_loss: {:.4f}'.format(
                        rouge_1, rouge_2, rouge_l, val_loss))

                    if val_loss > max_dev_f1:
                        increase_flag = True
                        max_dev_f1 = val_loss
                        best_state_dict_path = 'state_dict/' + \
                            self.config.model+'_'+self.config.dataset+'.pkl'
                        torch.save(self.model.state_dict(),
                                   best_state_dict_path)
                        print('>>> best model saved.')
                if increase_flag == False:
                    continue_not_increase += 1
                    if continue_not_increase >= self.config.patience:
                        print('early stop.')
                        break
                else:
                    continue_not_increase = 0

        return best_state_dict_path

    # TODO
    def _evaluate(self, data_loader):
        # switch model to evaluation mode
        self.model.eval()
        t_ap_spans_all, t_op_spans_all, t_triplets_all = None, None, None

        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = [t_sample_batched[col].to(
                    self.config.device) for col in self.config.input_cols]
                outputs = self.model.inference(t_inputs)

        return self._metrics(t_ap_spans_all, t_ap_spans_pred_all)

    # TODO
    def _metrics(targets, outputs):
        TP, FP, FN = 0, 0, 0
        n_sample = len(targets)
        assert n_sample == len(outputs)
        for i in range(n_sample):
            n_hit = 0
            n_output = len(outputs[i])
            n_target = len(targets[i])
            for t in outputs[i]:
                if t in targets[i]:
                    n_hit += 1
            TP += n_hit
            FP += (n_output - n_hit)
            FN += (n_target - n_hit)
        precision = float(TP) / float(TP + FP + 1e-5)
        recall = float(TP) / float(TP + FN + 1e-5)
        f1 = 2 * precision * recall / (precision + recall + 1e-5)

    # TODO
    def run(self, repeats=20):
        if not os.path.exists('log/'):
            os.mkdir('log/')

        if not os.path.exists('state_dict/'):
            os.mkdir('state_dict/')
        f_out = open('log/'+self.config.model+'_' +
                     self.config.dataset+'_val.txt', 'w', encoding='utf-8')

        for i in range(repeats):
            print('repeat: {0}'.format(i+1))
            f_out.write('repeat: {0}\n'.format(i+1))
            self._reset_params()
            _params = filter(lambda p: p.requires_grad,
                             self.model.parameters())
            optimizer = torch.optim.Adam(
                _params, lr=self.config.lr, weight_decay=self.config.l2reg)
            best_state_dict_path = self._train(optimizer)

        f_out.close()


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='asteseq2seq', type=str)
    parser.add_argument('--dataset', default='laptop14',
                        type=str, help='laptop14, rest14, rest15, rest16')
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--emb_dim', default=300, type=int)
    parser.add_argument('--hidden_size', default=300, type=int)
    parser.add_argument('--polarities_dim', default=4, type=int)
    parser.add_argument('--seed', default=776, type=int)
    parser.add_argument('--device', default=None, type=str)

    parser.add_argument('--beam_size', default=4, type=int)
    parser.add_argument('--encoder_hidden_size', default=300, type=int)
    parser.add_argument('--attention_hidden_size', default=300, type=int)

    parser.add_argument('--decoder_hidden_size', default=300, type=int)
    parser.add_argument('--char_hidden_size', default=50, type=int)
    parser.add_argument('--label_embedding_size', default=50, type=int)
    parser.add_argument('--char_embedding_size', default=50, type=int)

    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--encoder_num_layers', default=2, type=int)
    parser.add_argument('--decoder_num_layers', default=2, type=int)

    parser.add_argument('--label_size', default=3, type=int)
    parser.add_argument('--max_len', default=83, type=int)
    parser.add_argument('--max_char_len', default=10, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)

    parser.add_argument('--bidirectional', default=True, type=bool)
    parser.add_argument('--max_enc_steps', default=400, type=int)
    parser.add_argument('--max_dec_steps', default=100, type=int)
    parser.add_argument('--max_tes_steps', default=100, type=int)

    parser.add_argument('--min_dec_steps', default=350, type=int)
    parser.add_argument('--vocab_size', default=11, type=int)
    parser.add_argument('--cov_loss_wt', default=1.0, type=float)
    parser.add_argument('--pointer_gen', default=True, type=bool)

    parser.add_argument('--is_coverage', default=True, type=bool)
    parser.add_argument('--max_grad_norm', default=2.0, type=int)
    parser.add_argument('--adagrad_init_acc', default=0.1, type=float)
    parser.add_argument('--rand_unif_init_mag', default=0.02, type=float)

    parser.add_argument('--trunc_norm_init_std', default=1e-4, type=float)
    parser.add_argument('--eps', default=11, type=int)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--lr_coverage', default=0.15, type=float)

    parser.add_argument('--max_iterations', default=500000, type=int)
    parser.add_argument('--d_k', default=64, type=int)
    parser.add_argument('--d_v', default=64, type=int)
    parser.add_argument('--n_head', default=6, type=int)

    parser.add_argument('--tran', default=True, type=bool)
    parser.add_argument('--n_layers', default=6, type=int)
    parser.add_argument('--d_model', default=128, type=int)
    parser.add_argument('--d_inner', default=512, type=int)

    parser.add_argument('--PAD', default=0, type=int)
    parser.add_argument('--UNK', default=1, type=int)
    parser.add_argument('--BOS', default=2, type=int)
    parser.add_argument('--EOS', default=3, type=int)

    parser.add_argument('--PAD_TOKEN', default='<pad>', type=str)
    parser.add_argument('--UNK_TOKEN', default='<unk>', type=str)
    parser.add_argument('--BOS_TOKEN', default='<bos>', type=str)
    parser.add_argument('--EOS_TOKEN', default='<eos>', type=str)

    parser.add_argument('--src_max_train', default=400, type=int)
    parser.add_argument('--src_max_test', default=400, type=int)
    parser.add_argument('--tgt_max_train', default=100, type=int)
    parser.add_argument('--tgt_max_test', default=100, type=int)

    parser.add_argument('--num_return_seq', default=1, type=int)

    parser.add_argument('--n_warmup_steps', default=4000, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    opt = parser.parse_args()

    model_classes = {
        'asteseq2seq': PoGenSeq2Seq4ASTE,
    }
    input_colses = {
        'asteseq2seq': ['src_text', 'tgt_text', 'enc_input', 'enc_input_ext', 'enc_len', 'enc_pad_mask', 'src_oovs', 'max_oov_len', 'dec_input', 'dec_target', 'dec_len', 'dec_pad_mask'],

    }
    target_colses = {
        'asteseq2seq': ['target_indices'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    data_dirs = {
        'rest14': 'datasets/14rest',
        'rest15': 'datasets/15rest',
        'rest16': 'datasets/16rest',
        'laptop14': 'datasets/14lap',
    }
    opt.model_class = model_classes[opt.model]
    opt.input_cols = input_colses[opt.model]
    opt.target_cols = target_colses[opt.model]
    opt.eval_cols = ['target_indices']
    opt.initializer = initializers[opt.initializer]
    opt.data_dir = data_dirs[opt.dataset]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ins = Instructor(opt)
    ins._print_args()
    ins.run()
