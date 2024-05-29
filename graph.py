#!/usr/bin/env python
# -*- coding: utf-8 -*-
import benepar
import numpy as np
import pickle
import spacy

from spacy.tokens import Doc
from tqdm import tqdm
from transformers import BertTokenizer


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


# spaCy + Berkeley
nlp = spacy.load('en_core_web_md')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
nlp.add_pipe("benepar", config={"model": "benepar_en3"})
# BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def get_unique_elements(lists, aspects):
    unique_lists = []
    for i, lst in enumerate(lists):
        other_lists = lists[:i] + lists[i + 1:]
        unique = set(lst) - set.union(*map(set, other_lists))
        if len(unique) > 0:
            cons = max(list(unique), key=lambda x: len(x))
            unique_lists.append([cons.start, cons.end])
        else:
            start = aspects[i][1]
            end = aspects[i][2]
            unique_lists.append([start, end])
    return unique_lists


def single_aspect(text, aspects):
    # https://spacy.io/docs/usage/processing-text
    tokens = nlp(text)
    words = text.split()
    assert len(words) == len(list(tokens))

    token = aspects[0]
    asp, start, end = token[0], token[1], token[2]
    aspect_specific = []
    all_cons = []
    for sent in tokens.sents:
        for cons in sent._.constituents:
            if cons.text == text:
                continue
            all_cons.append(cons)
            if cons.start <= start <= end <= cons.end:
                if len(cons._.labels) > 0:  # len(cons) > 1:
                    aspect_specific.append(cons)
    aspect_specific_cons = []
    aspect_tag = ''
    for cons in aspect_specific:
        if cons._.labels[0] != 'S':
            aspect_specific_cons.append([cons.start, cons.end])
            aspect_tag = cons._.labels[0]
            break

    for cons in all_cons:
        if len(cons._.labels) > 0 and cons._.labels[0] == aspect_tag:  # len(cons) != 1
            flag = True
            for asp_cons in aspect_specific_cons:
                if cons.end <= asp_cons[0] or cons.start >= asp_cons[1]:
                    continue
                else:
                    flag = False
            if flag:
                aspect_specific_cons.append([cons.start, cons.end])
    if aspect_specific_cons == []:
        aspect_specific_cons.append([0, len(words)])
    return aspect_specific_cons


def distance_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    tokens = nlp(text)
    words = text.split()
    matrix = np.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(list(tokens))

    for sent in tokens.sents:
        for cons in sent._.constituents:
            if len(cons) == 1:
                continue
            matrix[cons.start:cons.end, cons.start:cons.end] += np.ones([len(cons), len(cons)])

    hops_matrix = np.amax(matrix, axis=1, keepdims=True) - matrix  # hops
    dis_matrix = 2 - hops_matrix / (np.amax(hops_matrix, axis=1, keepdims=True) + 1)

    return dis_matrix


def relation_matrix(text, aspects):
    # https://spacy.io/docs/usage/processing-text
    tokens = nlp(text)
    words = text.split()
    matrix = np.eye(len(words)).astype('float32')
    if len(words) != len(list(tokens)):
        print(words)
        print(list(tokens))
    assert len(words) == len(list(tokens))

    all_start = [aspect[1] for aspect in aspects]
    relations = [False] * len(tokens)

    if len(aspects) > 1:
        # intra-aspect
        # aspect-related collection
        aspect_nodes = [[] for _ in range(len(aspects))]
        for sent in tokens.sents:
            for cons in sent._.constituents:
                for idx, token in enumerate(aspects):
                    asp, start, end = token[0], token[1], token[2]
                    if cons.start <= start and end <= cons.end:
                        aspect_nodes[idx].append(cons)
        # aspect-specific
        aspect_specific_cons = get_unique_elements(aspect_nodes, aspects)
        for idx, cons in enumerate(aspect_specific_cons):
            for i in range(cons[0], cons[1]):
                matrix[all_start[idx]][i] = 2
                matrix[i][all_start[idx]] = 2
                relations[i] = True
        # globally-shared
        for i in range(len(relations)):
            if not relations[i]:
                for j in all_start:
                    matrix[i][j] = 3
                    matrix[j][i] = 3
        # inter-aspect
        for i in range(len(all_start)):
            for j in range(i + 1, len(all_start)):
                matrix[all_start[i]][all_start[j]] = 4
                matrix[all_start[j]][all_start[i]] = 4
    else:
        # pseudo aspect
        # intra-aspect
        # aspect-related collection
        aspect_specific_cons = single_aspect(text, aspects)
        all_start += [aspect[0] for aspect in aspect_specific_cons[1:]]

        # aspect-specific
        for idx, cons in enumerate(aspect_specific_cons):
            for i in range(cons[0], cons[1]):
                matrix[all_start[idx]][i] = 2
                matrix[i][all_start[idx]] = 2
                relations[i] = True
        # globally-shared
        for i in range(len(relations)):
            if not relations[i]:
                for j in all_start:
                    matrix[i][j] = 3
                    matrix[j][i] = 3

        # inter-aspect
        for i in range(len(all_start)):
            for j in range(i + 1, len(all_start)):
                matrix[all_start[i]][all_start[j]] = 4
                matrix[all_start[j]][all_start[i]] = 4

    return matrix


def lexicon_matrix(text, aspects, lexicon):
    # https://spacy.io/docs/usage/processing-text
    tokens = nlp(text)
    words = text.lower().split()
    assert len(words) == len(list(tokens))

    aspects_index = []
    for aspect in aspects:
        start = aspect[1]
        end = aspect[2]
        aspects_index.extend(list(range(start, end)))
    labels = []
    for i in range(len(tokens)):
        if words[i] not in lexicon.keys() or i in aspects_index:
            labels.append(0)
        else:
            labels.append(5)
    lex_matrix = np.tile(np.array(labels), (len(tokens), 1))
    return lex_matrix


def build_graph(text, aspects, lexicon):
    rel = relation_matrix(text, aspects)
    np.fill_diagonal(rel, 1)
    mask = (np.zeros_like(rel) != rel).astype('float32')

    lex = lexicon_matrix(text, aspects, lexicon)
    lex = lex * mask

    dis = distance_matrix(text)
    np.fill_diagonal(dis, 1)
    dis = dis * mask

    return dis, rel, lex


def map_bert_2D(ori_adj, text):
    words = text.split()
    bert_tokens = []
    bert_map = []
    for src_i, word in enumerate(words):
        for subword in tokenizer.tokenize(word):
            bert_tokens.append(subword)  # * ['expand', '##able', 'highly', 'like', '##ing']
            bert_map.append(src_i)  # * [0, 0, 1, 2, 2]

    truncate_tok_len = len(bert_tokens)
    bert_adj = np.zeros((truncate_tok_len, truncate_tok_len), dtype='float32')
    for i in range(truncate_tok_len):
        for j in range(truncate_tok_len):
            bert_adj[i][j] = ori_adj[bert_map[i]][bert_map[j]]
    return bert_adj


def opinion_lexicon():
    pos_file = 'opinion-lexicon-English/positive-words.txt'
    neg_file = 'opinion-lexicon-English/negative-words.txt'
    fin1 = open(pos_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    fin2 = open(neg_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines1 = fin1.readlines()
    lines2 = fin2.readlines()
    fin1.close()
    fin2.close()
    lexicon = {}
    for pos_word in lines1:
        lexicon[pos_word.strip()] = 'positive'
    for neg_word in lines2:
        lexicon[neg_word.strip()] = 'negative'

    return lexicon


def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()

    idx2graph_dis, idx2graph_rel, idx2graph_lex = {}, {}, {}
    lexicon = opinion_lexicon()

    fout1 = open(filename + '_distance.pkl', 'wb')
    fout2 = open(filename + '_relation.pkl', 'wb')
    fout3 = open(filename + '_opinion.pkl', 'wb')

    for i in tqdm(range(0, len(lines), 3)):
        text = lines[i].strip()
        all_aspect = lines[i + 1].strip()
        aspects = []
        for aspect_index in all_aspect.split('\t'):
            aspect, start, end = aspect_index.split('#')
            aspects.append([aspect, int(start), int(end)])

        dis_adj, rel_adj, lex_adj = build_graph(text, aspects, lexicon)
        bert_dis_adj = map_bert_2D(dis_adj, text)
        bert_rel_adj = map_bert_2D(rel_adj, text)
        bert_lex_adj = map_bert_2D(lex_adj, text)

        idx2graph_dis[i] = bert_dis_adj
        idx2graph_rel[i] = bert_rel_adj
        idx2graph_lex[i] = bert_lex_adj

    pickle.dump(idx2graph_dis, fout1)
    pickle.dump(idx2graph_rel, fout2)
    pickle.dump(idx2graph_lex, fout3)
    fout1.close()
    fout2.close()
    fout3.close()


if __name__ == '__main__':
    process('dataset/lap14_train')
    process('dataset/lap14_test')
    process('dataset/rest14_train')
    process('dataset/rest14_test')
    process('dataset/mams_train')
    process('dataset/mams_dev')
    process('dataset/mams_test')
