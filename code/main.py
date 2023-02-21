"""
Pytorch Implementation of Dual Graph Neural Network (DGNN) model in:
Zihao Li et al. Exploiting Explicit and Implicit Item relationships for Session-based Recommendation. In WSDM 2023.

@Time   : 2022/11/24
@Author : Zihao Li
@Email  : zihao.li@student.uts.edu.au
"""

import argparse
import copy
import pickle
import time
from utils import Data, split_validation
from model import *
import os
import logging


ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose1_64', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/Gowalla/LastFM/sample')
parser.add_argument('--random_seed', default=2023, help='random_seed')
parser.add_argument('--len_max', type=int, default=70, help='max lenghth of sequences')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--fuse_A', action='store_true', default=False, help='whether to fuse an auxiliary adjacent matrix via correlation or self-attention')
parser.add_argument('--global_att_block_nums', type=int, default=5, help='the number of global attention blocks')
parser.add_argument('--global_att_head_nums', type=int, default=4, help='the number of multi-heads for global attention')
parser.add_argument('--dropout_global_att', type=float, default=0.5, help='dropout of global attention')
parser.add_argument('--dropout_global_ffn', type=float, default=0.5, help='dropout of ffn in global attention block')
parser.add_argument('--epoch', type=int, default=50, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--mt', type=float, default=0.9, help='the momentum of SGD')
parser.add_argument('--lr_dc', type=float, default=0.5, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step_global', type=int, default=2, help='global gnn propogation steps')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--patience', type=int, default=5, help='the number of epoch to wait before early stop ')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
# parser.add_argument('--model_description', default='GateCell+last_global+Att_global', help='model description')
parser.add_argument('--log_file', default='log/', help='log dir path')


opt = parser.parse_args()
print(opt)



if not os.path.exists(opt.log_file):
        os.makedirs(opt.log_file)

if not os.path.exists(opt.log_file + opt.dataset):
        os.makedirs(opt.log_file + opt.dataset)



logging.basicConfig(level=logging.INFO, filename=opt.log_file + opt.dataset + '/' + opt.dataset + '-' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.log',
                    datefmt='%Y/%m/%d %H:%M:%S', format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s', filemode='w')
logger = logging.getLogger(__name__)
logger.info(opt)


def config_dataset():
    if opt.dataset == 'LastFM':
        opt.random_seed = 8679
        opt.global_att_block_nums = 5
        opt.step_global = 2
    elif opt.dataset == 'yoochoose1_64':
        opt.random_seed = 1008
        opt.global_att_block_nums = 6
        opt.step_global = 2
    elif opt.dataset == 'yoochoose1_4':
        opt.random_seed = 9681
        opt.global_att_block_nums = 6
        opt.step_global = 3
    elif opt.dataset == 'Gowalla':
        opt.random_seed = 2994
        opt.global_att_block_nums = 4
        opt.step_global = 4
    elif opt.dataset == 'diginetica':
        opt.random_seed = 9218
        opt.global_att_block_nums = 5
        opt.step_global = 2


def lens_max():
    dict_user_count = {'diginetica': 70, 'yoochoose1_4': 200, 'yoochoose1_64': 146, 'Gowalla': 20, 'LastFM': 20, 'samples': 17}
    if opt.dataset in dict_user_count:
        opt.len_max = dict_user_count[opt.dataset]
    else:
        raise Exception('Dataset-{} is not included in the options'.format(opt.dataset))


def main():

    lens_max()
    config_dataset()
    # train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
    train_data = pickle.load(open(ROOT_DIR+'\datasets\\' + opt.dataset + '\\train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:

        # test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
        test_data = pickle.load(open(ROOT_DIR+'\datasets\\' + opt.dataset + '\\test.txt', 'rb'))

    train_data = Data(train_data, opt.len_max, shuffle=True)
    test_data = Data(test_data, opt.len_max, shuffle=False)

    # del all_train_seq, g
    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    elif opt.dataset == 'Gowalla':
        n_node = 29511
    elif opt.dataset == 'LastFM':
        n_node = 38616
    else:
        n_node = 310

    model = trans_to_cuda(DGNN(opt, n_node))

    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        logger.info('-------------------------------------------------------')
        print('epoch: ', epoch)
        logger.info('epoch: {}'.format(epoch))
        hit, mrr, model_temp = train_test(model, train_data, test_data, opt.random_seed, logger)

        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
            bad_counter = 0
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
            bad_counter = 0
        if flag == 1:
            model_best = copy.deepcopy(model_temp)
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d' % (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        logger.info('Best Result:')
        logger.info('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d' % (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    logger.info('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))
    logger.info("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
