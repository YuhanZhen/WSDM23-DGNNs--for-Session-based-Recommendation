"""
Pytorch Implementation of Dual Graph Neural Network (DGNN) model in:
Zihao Li et al. Exploiting Explicit and Implicit Item relationships for Session-based Recommendation. In WSDM 2023.

@Time   : 2022/11/24
@Author : Zihao Li
@Email  : zihao.li@student.uts.edu.au
"""

import datetime
import math
import numpy as np
import torch

from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


class Global_GNN(Module):
    def __init__(self, fuse_A, hidden_size, step=1):
        super(Global_GNN, self).__init__()
        self.fuse_A = fuse_A
        self.step = step
        self.hidden_size = hidden_size
        self.w_h = Parameter(torch.Tensor(self.hidden_size*3, self.hidden_size))
        self.w_hf = Parameter(torch.Tensor(self.hidden_size*2, self.hidden_size))

    def GateCell(self, A, hidden):
        hidden_w = F.linear(hidden, self.w_h)

        hidden_w_0, hidden_w_1, hidden_w_2 = hidden_w.chunk(3, -1)
        hidden_fuse_0, hidden_fuse_1 = F.linear(torch.matmul(A, hidden_w_0), self.w_hf).chunk(2, -1)

        gate = torch.relu(hidden_fuse_0 + hidden_w_1)
        return hidden_w_2 + gate * hidden_fuse_1

    def Fuse_with_correlation(self, A_Global, hidden):
        correlation_A = torch.matmul(hidden, hidden.transpose(1, 0))
        correlation_A_std = torch.norm(correlation_A, p=2, dim=1, keepdim=True)
        correlation_A = correlation_A/correlation_A_std
        return A_Global + correlation_A

    def forward(self, A_Global, hidden):
        seqs = []
        if self.fuse_A:
            A_Global = self.Fuse_with_correlation(A_Global, hidden)
        for i in range(self.step):
            hidden = self.GateCell(A_Global, hidden)
            seqs.append(hidden)
        return hidden, torch.mean(torch.stack(seqs, dim=1), dim=1)


class Global_ATT(Module):
    def __init__(self, hidden_size, num_heads, dp_att, dp_ffn):
        super(Global_ATT, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_att = dp_att
        self.q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.attention_dropout = nn.Dropout(self.dropout_att)
        self.ffn = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.dropout_ffn = dp_ffn
        self.ffn_dropout = nn.Dropout(self.dropout_ffn)

    def transpose_qkv(self, hidden):
        hidden = hidden.reshape(hidden.shape[0], self.num_heads, -1)
        hidden = hidden.permute(1, 0, 2)
        return hidden

    def transpose_output(self, inputs):
        inputs = inputs.permute(1, 0, 2)
        inputs = inputs.contiguous().view(inputs.size()[0], 1, -1).squeeze(1)
        return inputs

    def forward(self, inputs):
        query_h = self.transpose_qkv(self.q(inputs))
        key_h = self.transpose_qkv(self.k(inputs))
        value_h = self.transpose_qkv(self.v(inputs))
        softmax_score = torch.tanh(torch.matmul(query_h, key_h.transpose(dim0=1, dim1=-1)))
        att_hidden = self.transpose_output(torch.matmul(self.attention_dropout(softmax_score), value_h))
        att_hidden = self.ffn_dropout(torch.relu(self.ffn(att_hidden)))
        return inputs, att_hidden


class Global_ATT_blocks(Module):
    def __init__(self, block_nums_global, hidden_size, global_att_num_heads, dropout_global_att, dropout_global_ffn):
        super(Global_ATT_blocks, self).__init__()
        self.block_nums = block_nums_global
        self.att_layer = Global_ATT(hidden_size, global_att_num_heads, dropout_global_att, dropout_global_ffn)
        # self.multi_block_att = [self.att_layer for _ in range(self.block_nums)]
        self.multi_block_att = [Global_ATT(hidden_size, global_att_num_heads, dropout_global_att, dropout_global_ffn) for _ in range(self.block_nums)]
        for i, global_attention in enumerate(self.multi_block_att):
            self.add_module('global_attention_{}'.format(i), global_attention)

    def forward(self, inputs):
        for global_att_temp in self.multi_block_att:
            inputs, att_hidden = global_att_temp(inputs)
            inputs = inputs + att_hidden
        return inputs


class DGNN(Module):
    def __init__(self, opt, n_node):
        super(DGNN, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize

        self.global_att_block_nums = opt.global_att_block_nums
        self.global_att_num_heads = opt.global_att_head_nums
        self.dropout_global_att = opt.dropout_global_att
        self.dropout_global_ffn = opt.dropout_global_ffn
        self.fuse_A = opt.fuse_A

        self.embedding = nn.Embedding(self.n_node, self.hidden_size)


        self.global_gnn = Global_GNN(self.fuse_A, self.hidden_size, step=opt.step_global)
        self.global_att_blocks = Global_ATT_blocks(self.global_att_block_nums, self.hidden_size, self.global_att_num_heads, self.dropout_global_att,
                                                   self.dropout_global_ffn)

        self.global_fuse_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.convariance = True
        self.normalization_session = nn.BatchNorm1d(opt.len_max)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.embedding_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.nonhybrid = opt.nonhybrid
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=opt.lr, momentum=opt.mt)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        if self.convariance:
            hidden_std = self.normalization_session(hidden)
            convariance = torch.matmul(hidden_std, hidden_std.transpose(2, 1))
            hidden_conv = torch.matmul(convariance, hidden)
            hidden = hidden + hidden_conv

        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])
        q2 = self.linear_two(hidden)

        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        # b = self.embedding_linear(self.embedding.weight)[1:]
        b = self.embedding.weight[1:]
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def forward(self, A_global, inputs_global, inputs_global_index):
        hidden_global = self.embedding(inputs_global)
        hidden_last_global, hidden_fuse_global = self.global_gnn(A_global, hidden_global)
        hidden_att_global = self.global_att_blocks(hidden_global)
        hidden_last_global = self.global_fuse_linear(torch.cat([hidden_last_global, hidden_att_global], dim=-1))
        fuse_global = hidden_last_global[inputs_global_index]
        return fuse_global


def forward(model, i, data):
    alias_inputs, mask, targets, A_global, inputs, global_inputs_index = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    A_global = trans_to_cuda(torch.Tensor(A_global).float())
    inputs = trans_to_cuda(torch.Tensor(inputs).long())
    global_inputs_index = trans_to_cuda(torch.Tensor(global_inputs_index).long())

    hidden = model(A_global, inputs, global_inputs_index)

    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    scores = model.compute_scores(seq_hidden, mask)
    return targets, scores


def train_test(model, train_data, test_data, seed_random, logger):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    logger.info('start training:{}'.format(datetime.datetime.now()))
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size, seed_random)

    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss.backward()

        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
            logger.info('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)
    logger.info('\tLoss:\t%.3f' % total_loss)
    print('start predicting: ', datetime.datetime.now())
    logger.info('start predicting: {}'.format(datetime.datetime.now()))
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size, seed_random)

    for i in slices:
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr, model


