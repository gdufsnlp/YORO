#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import argparse
import math
import os
import sys
import random
import numpy as np

from sklearn import metrics
from time import strftime, localtime

from transformers import BertModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_utils import Tokenizer4Bert, ABSADataset
from model import YORO
from layer.supervisedcontrastiveloss import SupervisedContrastiveLoss

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
        bert = BertModel.from_pretrained(opt.pretrained_bert_name)
        self.model = opt.model_class(bert, opt).to(opt.device)
        self.trainset = ABSADataset(opt.dataset_file['train'], tokenizer)
        self.testset = ABSADataset(opt.dataset_file['test'], tokenizer)
        if self.opt.dataset == 'mams':
            self.valset = ABSADataset(opt.dataset_file['dev'], tokenizer)
        else:
            assert 0 <= opt.valset_ratio < 1
            if opt.valset_ratio > 0:
                valset_len = int(len(self.trainset) * opt.valset_ratio)
                self.trainset, self.valset = random_split(self.trainset, (len(self.trainset) - valset_len, valset_len))
            else:
                self.valset = self.testset

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info(
            '> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) not in [BertModel, nn.Embedding]:  # skip bert params and embedding
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader, test_data_loader):
        max_val_acc = 0
        max_val_f1 = 0
        max_val_epoch = 0
        global_step = 0
        path = None
        for i_epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(i_epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            n_op_correct, n_op_total = 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs, opinion_outputs = self.model(inputs)

                targets = batch['polarities'].to(self.opt.device)
                outputs = outputs.view(-1, self.opt.polarities_dim)  # bz*128,3
                targets = targets.view(-1)  # bz*128,1
                mask = targets != -1  # bz*128, 1   non-aspect False aspect True
                mask_outputs = outputs[mask]
                mask_targets = targets[mask]
                loss1 = criterion[0](mask_outputs, mask_targets)

                opinion_targets = batch['opinion_indices'].to(self.opt.device)
                opinion_outputs = opinion_outputs.view(-1, self.opt.polarities_dim)
                opinion_targets = opinion_targets.view(-1)  # bz*128,1
                opinion_mask = opinion_targets != -1
                mask_opinion_outputs = opinion_outputs[opinion_mask]
                mask_opinion_targets = opinion_targets[opinion_mask]
                loss2 = criterion[0](mask_opinion_outputs, mask_opinion_targets)

                loss3 = criterion[1](nn.functional.normalize(mask_outputs, dim=1), mask_targets)

                loss = loss1 + loss2 + self.opt.alpha * loss3  # 0.5
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(mask_outputs, -1) == mask_targets).sum().item()
                n_total += len(mask_outputs)
                n_op_correct += (torch.argmax(mask_opinion_outputs, -1) == mask_opinion_targets).sum().item()
                n_op_total += len(mask_opinion_outputs)

                loss_total += loss.item()
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    train_op_acc = n_op_correct / n_op_total
                    logger.info('loss: {:.4f}, acc: {:.4f}, op_acc: {:.4f}, '
                                'loss1: {:.4f}, loss2: {:.4f}, loss3: {:.4f}'.format(train_loss, train_acc,
                                                                                     train_op_acc, loss1, loss2, loss3))

            val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)
            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))

            if val_acc > max_val_acc:  # acc improve
                max_val_acc = val_acc
                max_val_f1 = val_f1
                max_val_epoch = i_epoch

                if not os.path.exists(
                        '{}/{}/{}'.format(self.opt.save_model_dir, self.opt.model_name, self.opt.dataset)):
                    os.makedirs('{}/{}/{}'.format(self.opt.save_model_dir, self.opt.model_name, self.opt.dataset))
                path = '{0}/{1}/{2}/acc_{3}_f1_{4}_{5}.model'.format(self.opt.save_model_dir, self.opt.model_name,
                                                                     self.opt.dataset,
                                                                     round(val_acc, 4), round(val_f1, 4),
                                                                     strftime("%y%m%d-%H%M", localtime()))
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
            if i_epoch - max_val_epoch >= self.opt.patience:
                print('>> early stop.')
                break

        return path

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_batch['polarities'].to(self.opt.device)
                t_outputs, t_opinion_outputs = self.model(t_inputs)

                t_targets = t_targets.view(-1)
                t_outputs = t_outputs.view(-1, self.opt.polarities_dim)
                t_mask = t_targets.view(-1) != -1
                t_mask_outputs = t_outputs[t_mask]
                t_mask_targets = t_targets[t_mask]

                n_correct += (torch.argmax(t_mask_outputs, -1) == t_mask_targets).sum().item()
                n_total += len(t_mask_outputs)

                if t_targets_all is None:
                    t_targets_all = t_mask_targets
                    t_outputs_all = t_mask_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_mask_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_mask_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                              average='macro')
        return acc, f1

    def run(self):
        acc_list, f1_list = [], []
        # Loss and Optimizer
        criterion = [nn.CrossEntropyLoss(), SupervisedContrastiveLoss()]
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)

        for i in range(self.opt.repeat):
            self._reset_params()
            best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader, test_data_loader)
            self.model.load_state_dict(torch.load(best_model_path))
            test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
            logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))
            acc_list.append(test_acc)
            f1_list.append(test_f1)
        all_acc = np.asarray(acc_list)
        avg_acc = np.average(all_acc)
        all_f1 = np.asarray(f1_list)
        avg_f1 = np.average(all_f1)
        for acc, f1 in zip(acc_list, f1_list):
            logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(acc, f1))
        logger.info('>> avg_test_acc: {:.4f}, avg_test_f1: {:.4f}'.format(avg_acc, avg_f1))


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='YORO', type=str)
    parser.add_argument('--dataset', default='rest14', type=str, help='mams, rest14, lap14')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--repeat', default=1, type=int)
    parser.add_argument('--lr', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--l2reg', default=1e-4, type=float)
    parser.add_argument('--num_epoch', default=20, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--hidden_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=128, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=1, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float,help='set ratio between 0 and 1 for validation support')
    parser.add_argument('--save_model_dir', default='/Your_Path', type=str)

    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt.seed)

    model_classes = {
        'YORO': YORO,
    }
    input_colses = {
        'YORO': ['input_ids', 'token_type_ids', 'attention_mask', 'distance_adj', 'relation_adj'],
    }
    dataset_files = {
        'lap14': {
            'train': './dataset/lap14_train',
            'test': './dataset/lap14_test'
        },
        'rest14': {
            'train': './dataset/rest14_train',
            'test': './dataset/rest14_test'
        },
        'mams': {
            'train': './dataset/mams_train',
            'dev': './dataset/mams_dev',
            'test': './dataset/mams_test'
        }
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
        'adamw': torch.optim.AdamW,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    if not os.path.exists('log/{}'.format(opt.model_name)):
        os.makedirs('log/{}'.format(opt.model_name))
    log_file = 'log/{}/{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
