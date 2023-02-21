"""
Pytorch Implementation of Dual Graph Neural Network (DGNN) model in:
Zihao Li et al. Exploiting Explicit and Implicit Item relationships for Session-based Recommendation. In WSDM 2023.

@Time   : 2022/11/24
@Author : Zihao Li
@Email  : zihao.li@student.uts.edu.au
"""

import numpy as np
import itertools


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))

    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]

    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


def data_masks(all_usr_pois, item_tail, len_max):
    us_lens = [len(upois) for upois in all_usr_pois]
    # len_max = max(us_lens)+1
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max


class Data():
    def __init__(self, data, len_max, shuffle=False, graph=None):
        ## 0 is not in raw data
        inputs_raw = data[0]
        self.targets = np.asarray(data[1])
        inputs, mask, len_max = data_masks(inputs_raw, [0], len_max)

        self.inputs = np.asarray(inputs)

        self.mask = np.asarray(mask)
        self.len_max = len_max

        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

    def generate_batch(self, batch_size, seed_random):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.seed(seed_random)
            # np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        n_node, alias_inputs, global_input_index = [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)

        nodes = np.array(list(set(itertools.chain.from_iterable(inputs))))

        num_nodes = len(nodes)
        A_global = np.zeros((num_nodes, num_nodes))

        for u_input, u_mask in zip(inputs, mask):
            node = np.unique(u_input)

            # global_input_index.append([nodes.index(i) for i in node] + (max_n_node - len(node)) * [nodes.index(0)])
            global_input_index.append([np.where(nodes == i)[0][0] for i in node] + (max_n_node - len(node)) * [np.where(nodes == 0)[0][0]])
            u_A = np.zeros((max_n_node, max_n_node))

            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    u = np.where(node == u_input[i])[0][0]
                    u_A[u][u] = 1
                    u_g = np.where(nodes == u_input[i])[0][0]
                    # u_g = nodes.index(u_input[i])
                    A_global[u_g][u_g] = 1
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] += 1
                u_A[v][u] += 1
                u_A[u][u] = 1
                u_A[v][v] = 1

                u_g = np.where(nodes == u_input[i])[0][0]
                v_g = np.where(nodes == u_input[i+1])[0][0]
                # u_g = nodes.index(u_input[i])
                # v_g = nodes.index(u_input[i+1])
                A_global[u_g][v_g] += 1
                A_global[v_g][u_g] += 1
                A_global[u_g][u_g] = 1
                A_global[v_g][v_g] = 1

            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1

            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        A_global_sum_out = np.sum(A_global, 1)
        A_global_sum_out[np.where(A_global_sum_out == 0)] = 1
        A_global_norm = np.divide(A_global.transpose(), A_global_sum_out).transpose()
        return alias_inputs, mask, targets, A_global_norm, list(nodes), global_input_index

