#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
import time
import pandas as pd
import csv
import pickle
import operator
import datetime
import os
import collections
import numpy as np
from itertools import chain


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name: diginetica/yoochoose/sample')
opt = parser.parse_args()


prefix_digi_path = r'../'


dataset = 'sample_train-item-views.csv'
if opt.dataset == 'diginetica':
    dataset = prefix_digi_path + '/train-item-views.csv'
    session_id = 'sessionId'
    item_id = 'itemId'

elif opt.dataset =='yoochoose':
    dataset = 'yoochoose/yoochoose-clicks/yoochoose-clicks.dat'
    session_id = 'session_id'
    item_id = 'item_id'
else:
    session_id = 'session_id'
    item_id = 'item_id'

print("-- Starting @ %ss" % datetime.datetime.now())
with open(dataset, "r") as f:
    if opt.dataset == 'yoochoose':
        reader = csv.DictReader(f, delimiter=',', fieldnames=["session_id", "timestamp", "item_id", "Category"])
    else:
        reader = csv.DictReader(f, delimiter=';')

    sess_clicks = {}
    sess_date = {}
    ctr = 0
    curid = -1
    curdate = None

    for data in reader:

        sessid = data[session_id]
        if curdate and not curid == sessid:
            date = ''
            if opt.dataset == 'yoochoose':
                date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
            else:
                date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))

            sess_date[curid] = date
        curid = sessid

        if opt.dataset == 'yoochoose':
            item = data[item_id]
        else:
            item = data[item_id], int(data['timeframe'])

        curdate = ''
        if opt.dataset == 'yoochoose':
            curdate = data['timestamp']
        else:
            curdate = data['eventdate']


        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        ctr += 1
    date = ''
    if opt.dataset == 'yoochoose':
        date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
    else:
        date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
        for i in list(sess_clicks):
            sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
            sess_clicks[i] = [c[0] for c in sorted_clicks]

    sess_date[curid] = date
print("-- Reading data @ %ss" % datetime.datetime.now())

# Filter out length 1 sessions
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_date[s]

# Count number of times each item appears
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

length = len(sess_clicks)

for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))  ## delete the items which count number less than 5

    if len(filseq) < 2:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq

# Split out test set based on dates
dates = list(sess_date.items())

maxdate = dates[0][1]

for _, date in dates:
    if maxdate < date:
        maxdate = date

# 7 days for test
splitdate = 0
if opt.dataset == 'yoochoose':
    splitdate = maxdate - 86400 * 1  # the number of seconds for a day：86400
else:
    splitdate = maxdate - 86400 * 7

print('Splitting date', splitdate)      # diginetica: ('Split date', 1464105600.0); Yoochoose: ('Split date', 1411930799.0)
tra_sess = filter(lambda x: x[1] < splitdate, dates)
tes_sess = filter(lambda x: x[1] > splitdate, dates)

# Sort sessions by date
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
print(len(tra_sess))    # 186670 for diginetica   # 7966257 for yoochoose
print(len(tes_sess))    # 15979 for diginetica    # 15324 for yoochoose
print(tra_sess[:3])
print(tes_sess[:3])
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())


# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}
# Convert training sessions to sequences and renumber items to start from 1
def obtian_tra():
    train_ids = []
    train_seqs = []
    train_dates = []
    item_ctr = 1

    for s, date in tra_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
        if len(outseq) < 2:  # Doesn't occur
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]
    print(item_ctr)     # 43098 for diginetica; 37484 for yoochoose
    return train_ids, train_dates, train_seqs


# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes():
    test_ids = []
    test_seqs = []
    test_dates = []
    for s, date in tes_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
    return test_ids, test_dates, test_seqs


tra_ids, tra_dates, tra_seqs = obtian_tra()  ## recode the item id
tes_ids, tes_dates, tes_seqs = obtian_tes()


def process_seqs(iseqs, idates):
    out_seqs = []
    out_dates = []
    labs = []
    ids = []

    for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]
            ids += [id]
    return out_seqs, out_dates, labs, ids


def process_seqs_bi(iseqs, idates):
    out_seqs = []
    out_dates = []
    labs = []
    ids = []

    for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]
            ids += [id]

    num = ids[-1]+1
    for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
        seq = seq[::-1]
        id += num
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]
            ids += [id]

    return out_seqs, out_dates, labs, ids


def items_cor_list(seqs, nodes_num):
    pair_mat = np.zeros([nodes_num+1, nodes_num+1], dtype=int)
    for seq in seqs:
        for i in range(len(seq)-1):
            m = seq[i]
            n = seq[i+1]
            pair_mat[m][n] += 1
            pair_mat[n][m] += 1
    ## softmax-norm?
    # norm = np.sum(pair_mat, 1)
    # norm[norm == 0] = 1
    # pair_mat_norm = (pair_mat.T/norm).T

    return pair_mat


# nodes_num = max(list(chain.from_iterable(tra_seqs)) + list(chain.from_iterable(tes_seqs)))
#
# tra_mat = items_cor_list(tra_seqs, nodes_num)
# te_mat = items_cor_list(tes_seqs, nodes_num)

# for i in range(len(tra_mat)):
#     tra_temp = np.array(tra_mat[i], dtype=bool)
#     te_temp = np.array(te_mat[i], dtype=bool)
#     print(np.sum((tra_temp & te_temp) != 0)/(np.sum(tra_temp != 0)+np.sum(te_temp != 0)))
# quit()

tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)  ## create labels
te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)

tra = (tr_seqs, tr_labs)
tes = (te_seqs, te_labs)
print(tr_seqs[20])
print(type(tr_labs))
quit()

print(len(tr_seqs))  ## 719470 for diginetica; 23670982 for yoochoose;
print(len(te_seqs))  ## 60858 for diginetica; 55898 for yoochoose;
print(tr_seqs[:3], tr_dates[:3], tr_labs[:3])
print(te_seqs[:3], te_dates[:3], te_labs[:3])
all = 0

print(max(collections.Counter([len(i) for i in tra_seqs]).keys()))  ## 70 for diginetica; 200 for yoochoose;
print(max(collections.Counter([len(i) for i in tes_seqs]).keys()))  ## 41 for diginetica; 74 for yoochoose;

for seq in tra_seqs:
    all += len(seq)

for seq in tes_seqs:
    all += len(seq)


print('avg length: ', all/(len(tra_seqs) + len(tes_seqs) * 1.0))  ## 4.850942344040704 for diginetica; 3.9727042800167034 for yoochoose;

if opt.dataset == 'diginetica':
    if not os.path.exists('diginetica'):
        os.makedirs('diginetica')

    pickle.dump(tra, open('diginetica/train.txt', 'wb'))
    pickle.dump(tes, open('diginetica/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('diginetica/all_train_seq.txt', 'wb'))
    # np.save('diginetica/train_mat.npy', tra_mat)
    # np.save('diginetica/test_mat.npy', te_mat)

elif opt.dataset == 'yoochoose':
    if not os.path.exists('yoochoose1_4'):
        os.makedirs('yoochoose1_4')
    if not os.path.exists('yoochoose1_64'):
        os.makedirs('yoochoose1_64')
    pickle.dump(tes, open('yoochoose1_4/test.txt', 'wb'))
    pickle.dump(tes, open('yoochoose1_64/test.txt', 'wb'))

    split4, split64 = int(len(tr_seqs) / 4), int(len(tr_seqs) / 64)
    print(len(tr_seqs[-split4:]))
    print(len(tr_seqs[-split64:]))

    tra4, tra64 = (tr_seqs[-split4:], tr_labs[-split4:]), (tr_seqs[-split64:], tr_labs[-split64:])
    seq4, seq64 = tra_seqs[tr_ids[-split4]:], tra_seqs[tr_ids[-split64]:]

    pickle.dump(tra4, open('yoochoose1_4/train.txt', 'wb'))
    pickle.dump(seq4, open('yoochoose1_4/all_train_seq.txt', 'wb'))

    pickle.dump(tra64, open('yoochoose1_64/train.txt', 'wb'))
    pickle.dump(seq64, open('yoochoose1_64/all_train_seq.txt', 'wb'))

else:
    if not os.path.exists('sample'):
        os.makedirs('sample')
    pickle.dump(tra, open('sample/train.txt', 'wb'))
    pickle.dump(tes, open('sample/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('sample/all_train_seq.txt', 'wb'))

print('Done.')
