# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np


Triplet_End = '</s>'
Inner_Interval = '<d>'
Sentence_Start = '<e>'
Sentence_End = '</e>'


# This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
PAD_TOKEN = '<pad>'
# This has a vocab id, which is used to represent out-of-vocabulary words
UNK_TOKEN = '<unk>'
# # This has a vocab id, which is used at the start of every decoder input sequence
# BOS_TOKEN = '<bos>'
# # This has a vocab id, which is used at the end of untruncated target sequences
# EOS_TOKEN = '<eos>'

# This has a vocab id, which is used at the start of every decoder input sequence
START_DECODING = '<start>'
# This has a vocab id, which is used at the end of untruncated target sequences
STOP_DECODING = '<end>'


def load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


def build_embedding_matrix(data_dir, word2idx, embed_dim, type):
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(
        str(embed_dim), type)
    if os.path.exists(os.path.join(data_dir, embedding_matrix_file_name)):
        print('>>> loading embedding matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(
            open(os.path.join(data_dir, embedding_matrix_file_name), 'rb'))
    else:
        print('>>> loading word vectors ...')
        # words not found in embedding index will be randomly initialized.
        embedding_matrix = np.random.uniform(-1/np.sqrt(
            embed_dim), 1/np.sqrt(embed_dim), (len(word2idx), embed_dim))
        # <pad>
        embedding_matrix[0, :] = np.zeros((1, embed_dim))
        fname = './glove/glove.840B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
        print('>>> building embedding matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(os.path.join(
            data_dir, embedding_matrix_file_name), 'wb'))
    return embedding_matrix


class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1

            self.word2idx['<start>'] = self.idx
            self.idx2word[self.idx] = '<start>'
            self.idx += 1

            self.word2idx['<end>'] = self.idx
            self.idx2word[self.idx] = '<end>'
            self.idx += 1

            # self.word2idx['<bos>'] = self.idx
            # self.idx2word[self.idx] = '<bos>'
            # self.idx += 1

            # self.word2idx['<eos>'] = self.idx
            # self.idx2word[self.idx] = '<eos>'
            # self.idx += 1

            self.word2idx['<e>'] = self.idx
            self.idx2word[self.idx] = '<e>'
            self.idx += 1

            self.word2idx['</e>'] = self.idx
            self.idx2word[self.idx] = '</e>'
            self.idx += 1

            self.word2idx['<d>'] = self.idx
            self.idx2word[self.idx] = '<d>'
            self.idx += 1

            self.word2idx['</s>'] = self.idx
            self.idx2word[self.idx] = '</s>'
            self.idx += 1

            self.word2idx['positive'] = self.idx
            self.idx2word[self.idx] = 'positive'
            self.idx += 1
            self.word2idx['negative'] = self.idx
            self.idx2word[self.idx] = 'negative'
            self.idx += 1
            self.word2idx['neutral'] = self.idx
            self.idx2word[self.idx] = 'neutral'
            self.idx += 1

        else:
            self.word2idx = word2idx
            self.idx2word = {v: k for k, v in word2idx.items()}

    def fit_on_text(self, text):
        text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text):
        text = text.lower()
        words = text.split()
        unknownidx = self.unk()
        sequence = [self.word2idx[w]
                    if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return sequence

    def __len__(self):
        '''Returns size of the Vocabulary.'''
        return len(self.word2idx)

    def word2id(self, word):
        '''Return thr id (integer) of a word (string). Return id if word is OOV.'''
        unk_id = self.unk()
        return self.word2idx.get(word, unk_id)

    def id2word(self, word_id):
        '''Return the word string corresponding to an id (integer).'''
        if word_id not in self.idx2word:
            raise ValueError(f'Id not found in vocab:{word_id}')
        return self.idx2word[word_id]

    def size(self):
        '''Return the total size of the vocabulary.'''
        return len(self.word2idx)

    def start(self):
        return self.word2id(START_DECODING)

    def stop(self):
        return self.word2id(STOP_DECODING)

    def pad(self):
        return self.word2idx[PAD_TOKEN]

    def unk(self):
        return self.word2idx[UNK_TOKEN]

    def eos(self):
        return self.word2idx[EOS_TOKEN]

    def bos(self):
        return self.word2idx[BOS_TOKEN]

    def sentence_start(self):
        return self.word2idx[Sentence_Start]

    def sentence_end(self):
        return self.word2id[Sentence_End]

    def triplet_end(self):
        return self.word2idx[Triplet_End]

    def inner_Interval(self):
        return self.word2idx[Inner_Interval]

    def extend(self, oovs):
        extended_vocab = self.idx2word+list(oovs)
        return extended_vocab

    def tokens2ids(self, tokens):
        ids = [self.word2id(t) for t in tokens]
        return ids

    def source2ids_ext(self, src_tokens):
        """Maps source tokens to ids if in vocab, extended vocab ids if oov.

        Args:
            src_tokens: list of source text tokens

        Returns:
            ids: list of source text token ids
            oovs: list of oovs in source text
        """
        ids = []
        oovs = []
        for t in src_tokens:
            t_id = self.word2id(t)
            unk_id = self.word2idx[UNK_TOKEN]
            if t_id == unk_id:
                if t not in oovs:
                    oovs.append(t)
                    ids.append(self.size()+oovs.index(t))
                else:
                    ids.append(t_id)
            return ids, oovs

    def target2ids_ext(self, tgt_tokens, oovs):
        """Maps target text to ids, using extended vocab (vocab + oovs).

        Args:
            tgt_tokens: list of target text tokens
            oovs: list of oovs from source text (copy mechanism)

        Returns:
            ids: list of target text token ids
        """
        ids = []
        for t in tgt_tokens:
            t_id = self.word2id(t)
            unk_id = self.word2idx[UNK_TOKEN]
            if t_id == unk_id:
                if t in oovs:
                    ids.append(self.size() + oovs.index(t))
                else:
                    ids.append(unk_id)
            else:
                ids.append(t_id)
        return ids

    def outputids2words(self, ids, src_oovs):
        """Maps output ids to words
        Args:
            ids: list of ids
            src_oovs: list of oov words

        Returns:
            words: list of words mapped from ids
        """
        words = []
        extended_vocab = self.extend(src_oovs)
        for i in ids:
            try:
                w = self.idx2word(i)  # might be oov
            except ValueError as e:
                assert src_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary."
                try:
                    w = extended_vocab[i]
                except IndexError as e:
                    raise ValueError(f'Error: model produced word ID {i} \
                                       but this example only has {len(src_oovs)} article OOVs')
            words.append(w)
        return words


def build_tokenizer(data_dir):
    # if os.path.exists(os.path.join(data_dir, 'word2idx.pkl')):
    #     print('>>> loading {0} tokenizer...'.format(data_dir))
    #     with open(os.path.join(data_dir, 'word2idx.pkl'), 'rb') as f:
    #         word2idx = pickle.load(f)
    #         tokenizer = Tokenizer(word2idx=word2idx)
    # else:
    filenames = [os.path.join(data_dir, '%s.txt' % set_type)
                 for set_type in ['train', 'dev', 'test']]
    all_text = ''
    for filename in filenames:
        print("********************-------->>>>>>>", filename)
        fp = open(filename, 'r', encoding='utf-8')
        lines = fp.readlines()
        fp.close()
        for i in range(0, len(lines), 2):
            text = lines[i].strip()
            all_text += (text + ' ')
    tokenizer = Tokenizer()
    tokenizer.fit_on_text(all_text)
    print('>>> saving {0} tokenizer...'.format(data_dir))
    with open(os.path.join(data_dir, 'word2idx.pkl'), 'wb') as f:
        pickle.dump(tokenizer.word2idx, f)
    return tokenizer


class ASTEDataReader(object):
    def __init__(self, data_dir):
        self.polarity_map = {'neutral': 1,
                             'negtive': 2, 'positive': 3}  # NO_RELATION is 0

        self.reverse_polarity_map = {
            v: k for k, v in self.polarity_map.items()}

        self.data_dir = data_dir

    def get_train(self, tokenizer):
        return self._create_dataset('train', tokenizer)

    def get_dev(self, tokenizer):
        return self._create_dataset('dev', tokenizer)

    def get_test(self, tokenizer):
        return self._create_dataset('test', tokenizer)

    def _create_dataset(self, set_type, tokenizer):
        all_data = []

        file_name = os.path.join(self.data_dir, '%s.txt' % set_type)
        fp = open(file_name, 'r', encoding='utf-8')
        lines = fp.readlines()
        fp.close()

        for i in range(0, len(lines), 2):
            text = lines[i].strip().lower()
            target = lines[i+1].strip().lower()
            text_indices = tokenizer.text_to_sequence(text)
            target_indices = tokenizer.text_to_sequence(target)
            data = {
                'text': text,
                'target': target,
                'text_indices': text_indices,
                'target_indices': target_indices,
            }
            all_data.append(data)
        return all_data
