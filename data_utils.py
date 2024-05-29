#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


def opinion_lexicon():
    pos_file = 'lexicon/positive-words.txt'
    neg_file = 'lexicon/negative-words.txt'
    lexicon = {}
    fin1 = open(pos_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    fin2 = open(neg_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines1 = fin1.readlines()
    lines2 = fin2.readlines()
    fin1.close()
    fin2.close()
    for pos_word in lines1:
        lexicon[pos_word.strip()] = 'positive'
    for neg_word in lines2:
        lexicon[neg_word.strip()] = 'negative'
    return lexicon


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)

    def opinion_in_text(self, text, aspects, lexicon):
        aspect_index = []
        for asp_idx in aspects:
            _, start, end = asp_idx
            aspect_index.extend(list(range(start + 1, end + 1)))

        opinion_index = []
        for idx, word in enumerate(text.split()):
            for t in self.tokenizer.tokenize(word):
                if idx in aspect_index:
                    opinion_index.append(-1)  # skip aspect words
                elif word in lexicon.keys():
                    if lexicon[word] == 'negative':
                        opinion_index.append(0)
                    elif lexicon[word] == 'positive':
                        opinion_index.append(2)
                else:
                    opinion_index.append(1)
        assert len(opinion_index) == len(self.tokenizer.tokenize(text))
        return pad_and_truncate(opinion_index, self.max_seq_len, value=-1)

    def map_bert_1D(self, text):
        words = text.split()
        # bert_tokens = []
        bert_map = []
        for src_i, word in enumerate(words):
            for subword in self.tokenizer.tokenize(word):
                # bert_tokens.append(subword)  # * ['expand', '##able', 'highly', 'like', '##ing']
                bert_map.append(src_i)  # * [0, 0, 1, 2, 2]

        return bert_map


class ABSADataset(Dataset):
    def __init__(self, file, tokenizer):
        self.file = file
        self.tokenizer = tokenizer
        self.load_data()

    def load_data(self):
        fin = open(self.file, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        fin = open(self.file + '_relation.pkl', 'rb')
        rel_matrix = pickle.load(fin)
        fin.close()
        fin = open(self.file + '_opinion.pkl', 'rb')
        lex_matrix = pickle.load(fin)
        fin.close()
        fin = open(self.file + '_distance.pkl', 'rb')
        dis_matrix = pickle.load(fin)
        fin.close()

        lexicon = opinion_lexicon()
        all_data = []
        for i in range(0, len(lines), 3):
            text = lines[i].lower().strip()
            all_aspect = lines[i + 1].lower().strip()
            all_polarity = lines[i + 2].strip()
            aspects = []
            for aspect_idx in all_aspect.split('\t'):
                aspect, start, end = aspect_idx.split('#')
                aspects.append([aspect, int(start), int(end)])
            labels = []
            for label in all_polarity.split('\t'):
                labels.append(int(label) + 1)

            text_len = len(self.tokenizer.tokenizer.tokenize(text))
            input_ids = self.tokenizer.text_to_sequence('[CLS] ' + text + ' [SEP]')
            token_type_ids = [0] * (text_len + 2)
            attention_mask = [1] * len(token_type_ids)
            token_type_ids = pad_and_truncate(token_type_ids, self.tokenizer.max_seq_len)
            attention_mask = pad_and_truncate(attention_mask, self.tokenizer.max_seq_len)
            opinion_indices = self.tokenizer.opinion_in_text('[CLS] ' + text + ' [SEP]', aspects, lexicon)

            distance_adj = np.zeros((self.tokenizer.max_seq_len, self.tokenizer.max_seq_len)).astype('float32')
            distance_adj[1:text_len + 1, 1:text_len + 1] = dis_matrix[i]
            relation_adj = np.zeros((5, self.tokenizer.max_seq_len, self.tokenizer.max_seq_len)).astype('float32')
            for j in range(0, 4):
                r_tmp = np.where(rel_matrix[i] == j + 1, 1, 0)
                relation_adj[j, 1:text_len + 1, 1:text_len + 1] = r_tmp
            for k in range(4, 5):
                l_tmp = np.where(lex_matrix[i] == k + 1, 1, 0)
                relation_adj[k, 1:text_len + 1, 1:text_len + 1] = l_tmp
            polarities = [-1] * self.tokenizer.max_seq_len

            bert_index = self.tokenizer.map_bert_1D(text)
            for asp_idx, pol in zip(aspects, labels):
                _, start, end = asp_idx
                # label the first token of aspect
                polarities[bert_index.index(start) + 1] = pol  # +1 for cls

            polarities = np.asarray(polarities)
            data = {
                'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'distance_adj': distance_adj,
                'relation_adj': relation_adj,
                'polarities': polarities,
                'opinion_indices': opinion_indices,
            }

            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
