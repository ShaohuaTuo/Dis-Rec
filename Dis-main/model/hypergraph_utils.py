import numpy as np
import scipy.sparse as sp
import torch


def build_hypergraph_structure(train_dict, n_items, window_sizes=[3, 10]):
    print(f"Building Multi-Scale Hypergraph with windows: {window_sizes}...")

    items = []
    edges = []
    edge_idx = 0


    for uid, seq in train_dict.items():
        seq_len = len(seq)


        for w_size in window_sizes:
            if seq_len < w_size:
                continue


            for i in range(seq_len - w_size + 1):
                window = seq[i: i + w_size]
                for item_id in window:
                    items.append(item_id)
                    edges.append(edge_idx)
                edge_idx += 1


    # H[i, e] = 1
    H = sp.coo_matrix((np.ones(len(items)), (items, edges)),
                      shape=(n_items, edge_idx))

    print(f"Original Multi-Scale H shape: {H.shape}. Start Normalization...")


    # Z = D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2}

    # 1. Degree
    dv = np.array(H.sum(1)).flatten()
    de = np.array(H.sum(0)).flatten()

    # 2. Inverse Sqrt
    d_v_inv_sqrt = np.power(dv, -0.5)
    d_v_inv_sqrt[np.isinf(d_v_inv_sqrt)] = 0.
    Dv_mat = sp.diags(d_v_inv_sqrt)

    d_e_inv = np.power(de, -1.0)
    d_e_inv[np.isinf(d_e_inv)] = 0.
    De_mat = sp.diags(d_e_inv)

    # 3. Matrix Multiplication
    H = H.tocsr()
    HT = H.T

    # H_mod = H * De^{-1} * HT
    H_mod = H.dot(De_mat).dot(HT)

    # L = Dv^{-1/2} * H_mod * Dv^{-1/2}
    L = Dv_mat.dot(H_mod).dot(Dv_mat)

    # 4. To Tensor
    L = L.tocoo()
    indices = torch.from_numpy(np.vstack((L.row, L.col))).long()
    values = torch.from_numpy(L.data).float()
    shape = torch.Size(L.shape)

    HyperGraph_Adj = torch.sparse_coo_tensor(indices, values, shape, requires_grad=False)

    print(f"Multi-Scale Hypergraph built! Shape: {HyperGraph_Adj.shape}")
    return HyperGraph_Adj