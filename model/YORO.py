#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from layer.rgcn import RelationalGraphConvLayer


class YORO(nn.Module):
    def __init__(self, bert, args):
        super(YORO, self).__init__()
        self.bert = bert
        self.rgc1 = RelationalGraphConvLayer(5, args.bert_dim, args.bert_dim)
        self.rgc2 = RelationalGraphConvLayer(5, args.bert_dim, args.bert_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.op_dense = nn.Linear(args.bert_dim, args.polarities_dim)
        self.dense = nn.Linear(args.bert_dim, args.polarities_dim)

    def forward(self, inputs):
        input_ids, token_type_ids, attention_mask, distance_adj, relation_adj = inputs
        output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        hidden = output.last_hidden_state

        adj = distance_adj.unsqueeze(1).expand(-1, 5, -1, -1) * relation_adj
        x = F.relu(self.rgc1(hidden, adj))
        x = self.dropout(x)
        x = F.relu(self.rgc2(x, adj))

        hidden_output = self.dropout(x)
        op_logits = self.op_dense(hidden_output)
        logits = self.dense(hidden_output)
        return logits, op_logits
