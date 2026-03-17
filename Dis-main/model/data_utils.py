import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
import scipy.sparse as sp
import os


from hypergraph_utils import build_hypergraph_structure


def get_graph(dict_data, n_item):

    transfer_sep = [[], []]
    for u in dict_data.keys():
        item_seq = dict_data[u]
        if len(item_seq) > 1:
            transfer_sep[0].extend(item_seq[:-1])
            transfer_sep[1].extend(item_seq[1:])

    adj = sp.coo_matrix((np.ones(len(transfer_sep[0]), dtype=np.float32), transfer_sep), shape=(n_item, n_item))
    adj = (adj + adj.T).tolil()
    adj.setdiag(0)

    rowsum = np.array(adj.sum(axis=1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    normalized_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
    return normalized_adj


def convert_scipy_to_torch_sparse(sp_matrix):

    sp_matrix = sp_matrix.tocoo()
    indices = torch.from_numpy(np.vstack((sp_matrix.row, sp_matrix.col))).long()
    values = torch.from_numpy(sp_matrix.data).float()
    shape = torch.Size(sp_matrix.shape)
    return torch.sparse_coo_tensor(indices, values, shape, requires_grad=False)


def data_load(train_path, valid_path, test_path, w_min=None, w_max=1.0):

    train_list = np.load(train_path, allow_pickle=True)
    valid_list = np.load(valid_path, allow_pickle=True)
    test_list = np.load(test_path, allow_pickle=True)

    all_data = np.concatenate([train_list, valid_list, test_list])
    n_user = all_data[:, 0].max() + 1
    n_item = all_data[:, 1].max() + 1
    print(f'Number of users: {n_user}')
    print(f'Number of items: {n_item}')

    train_dict = {}
    for uid, iid in train_list:
        if uid not in train_dict:
            train_dict[uid] = []
        train_dict[uid].append(iid)


    ii_graph = get_graph(train_dict, n_item)
    print("Item-item graph created.")


    hyper_graph = build_hypergraph_structure(train_dict, n_item, window_sizes=[3,10])




    rows, cols, vals_ori, vals_w = [], [], [], []
    for uid, item_seq in train_dict.items():
        int_num = len(item_seq)


        if w_min is not None:
            weights = np.linspace(w_min, w_max, int_num, dtype=np.float64)
        else:
            weights = np.ones(int_num, dtype=np.float64)

        for i, iid in enumerate(item_seq):
            rows.append(uid)
            cols.append(iid)
            vals_ori.append(1.0)
            vals_w.append(weights[i])


    train_data_ori = sp.csr_matrix((vals_ori, (rows, cols)),
                                   dtype='float64', shape=(n_user, n_item))


    train_data_weighted = sp.csr_matrix((vals_w, (rows, cols)),
                                        dtype='float64', shape=(n_user, n_item))

    valid_y_data = sp.csr_matrix((np.ones_like(valid_list[:, 0]), (valid_list[:, 0], valid_list[:, 1])),
                                 dtype='float64', shape=(n_user, n_item))
    test_y_data = sp.csr_matrix((np.ones_like(test_list[:, 0]), (test_list[:, 0], test_list[:, 1])),
                                dtype='float64', shape=(n_user, n_item))


    return train_data_weighted, train_data_ori, valid_y_data, test_y_data, n_user, n_item, ii_graph, hyper_graph


class DataDiffusion(Dataset):


    def __init__(self, data_ori, data_weighted):
        self.data_ori = data_ori
        self.data_weighted = data_weighted

    def __getitem__(self, index):
      
        x_ori = torch.FloatTensor(self.data_ori[index].toarray().squeeze())
        x_weighted = torch.FloatTensor(self.data_weighted[index].toarray().squeeze())
        return x_ori, x_weighted

    def __len__(self):
        return self.data_ori.shape[0]