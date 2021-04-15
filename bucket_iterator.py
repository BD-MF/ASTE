# encoding=utf-8

import math
import random
import torch
import numpy as np


class BucketIterator(object):
    def __init__(self, data, batch_size, shuffle=True, sort=True, tokenizer=None, max_decode=0, device=None):

        self.shuffle = shuffle
        self.device = device
        self.sort = sort
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.pad()
        self.max_decode = max_decode
        self.batch_size = batch_size
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data)/batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x['text_indices']))
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(
                sorted_data[i*batch_size: (i+1)*batch_size]))
        return batches

    def pad_data(self, batch_data):

        batch_src_text_words = []
        batch_target_text_words = []

        batch_src_text = []
        batch_target_text = []

        batch_text_indices = []
        batch_target_indices = []

        max_len = max([len(t['text_indices']) for t in batch_data])

        pad_idx = self.tokenizer.pad()
        target_max_len = max([len(t['target_indices']) for t in batch_data])

        batch_enc_len = []
        batch_tgt_len = []
        for item in batch_data:
            text_indices, target_indices, text, target, = item['text_indices'], item[
                'target_indices'], item['text'], item['target']
            src_text = text.split()
            # print(text)
            # print(src_text)
            # print(target)
            batch_enc_len.append(len(src_text))

            batch_src_text.append(text)
            batch_src_text_words.append(src_text)
            tgt_text = target.split()
            batch_tgt_len.append(len(tgt_text))
            batch_target_text.append(target)
            batch_target_text_words.append(tgt_text)

            batch_text_indices.append(text_indices)
            batch_target_indices.append(target_indices)

        enc_len = torch.LongTensor(batch_enc_len)  # [B]
        dec_len = torch.LongTensor(batch_tgt_len)  # [B]

        # TODO 从这里开始 做输入修改
        enc_input = self.collate_tokens(values=batch_text_indices,
                                        pad_idx=pad_idx)
        enc_pad_mask = (enc_input == pad_idx)

        # Save additional info for pointer-generator
        # Determine max number of source text OOVs in this batch
        src_ids_ext, oovs = zip(
            *[self.tokenizer.source2ids_ext(s) for s in batch_src_text_words])
        # Store the version of the encoder batch that uses article OOV ids
        enc_input_ext = self.collate_tokens(values=src_ids_ext,
                                            pad_idx=pad_idx)
        # import pdb
        # pdb.set_trace()
        max_oov_len = max([len(oov) for oov in oovs])
        # Store source text OOVs themselves
        src_oovs = oovs

        tgt_ids_ext = [self.tokenizer.target2ids_ext(
            t, oov) for t, oov in zip(batch_target_text, src_oovs)]

        # create decoder inputs
        dec_input, _ = zip(
            *[self.get_decoder_input_target(t, self.max_decode) for t in batch_target_indices])

        dec_input = self.collate_tokens(values=dec_input,
                                        pad_idx=pad_idx,
                                        pad_to_length=self.max_decode)

        # create decoder targets using extended vocab
        _, dec_target = zip(
            *[self.get_decoder_input_target(t, self.max_decode) for t in tgt_ids_ext])

        dec_target = self.collate_tokens(values=dec_target,
                                         pad_idx=pad_idx,
                                         pad_to_length=self.max_decode)

        dec_pad_mask = (dec_input == pad_idx)
        # print(enc_input.shape)
        # import pdb
        # pdb.set_trace()
        return {
            'src_text': batch_src_text_words,
            'tgt_text': batch_target_text_words,
            'enc_input': torch.tensor(enc_input).to(self.device),  # [B x L]
            # [B x L]
            'enc_input_ext': torch.tensor(enc_input_ext).to(self.device),
            'enc_len': torch.tensor(enc_len).to(self.device),  # [B]
            # [B x L]
            'enc_pad_mask': torch.tensor(enc_pad_mask).to(self.device),
            # [list of length B]
            'src_oovs': torch.tensor(src_oovs).to(self.device),
            # [single int value]
            'max_oov_len': torch.tensor(max_oov_len).to(self.device),
            'dec_input': torch.tensor(dec_input).to(self.device),  # [B x T]
            # [B x T]
            'dec_target': torch.tensor(dec_target).to(self.device),
            'dec_len': torch.tensor(dec_len).to(self.device),  # [B]
            # [B x T]
            'dec_pad_mask': torch.tensor(dec_pad_mask).to(self.device),
        }

    def get_decoder_input_target(self, tgt, max_len):
        dec_input = [self.tokenizer.start()] + tgt
        dec_target = tgt + [self.tokenizer.stop()]
        # truncate inputs longer than max length
        if len(dec_input) > max_len:
            dec_input = dec_input[:max_len]
            dec_target = dec_target[:max_len]
        assert len(dec_input) == len(dec_target)
        return dec_input, dec_target

    def collate_tokens(self, values, pad_idx, left_pad=False,
                       pad_to_length=None):
        # Simplified version of `collate_tokens` from fairseq.data.data_utils
        """Convert a list of 1d tensors into a padded 2d tensor."""
        values = list(map(torch.LongTensor, values))
        size = max(v.size(0) for v in values)
        size = size if pad_to_length is None else max(size, pad_to_length)
        res = values[0].new(len(values), size).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][size - len(v):]
                        if left_pad else res[i][:len(v)])
        return res

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]
