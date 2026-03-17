import torch
import torch.nn as nn
import torch.nn.functional as F


class DisentangledHyperDecoder(nn.Module):
    def __init__(self, n_item, latent_dim, hyper_graph, n_channels=4, ssl_temp=0.1, dropout=0.2, tau=1.0):
        super(DisentangledHyperDecoder, self).__init__()
        self.n_item = n_item
        self.latent_dim = latent_dim
        self.n_channels = n_channels
        self.ssl_temp = ssl_temp
        self.tau = tau


        if hyper_graph.is_sparse:
            self.hyper_graph = hyper_graph.to_dense()
        else:
            self.hyper_graph = hyper_graph
        self.register_buffer('adj', self.hyper_graph)

        assert latent_dim % n_channels == 0, "Latent dim must be divisible by n_channels"
        self.channel_dim = latent_dim // n_channels

        self.item_embeddings = nn.Embedding(self.n_item, self.latent_dim)
        nn.init.xavier_normal_(self.item_embeddings.weight)


        self.layer_norm = nn.LayerNorm(self.channel_dim)
        self.dropout = nn.Dropout(dropout)


        self.gating_network = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.Tanh(),
            nn.Linear(latent_dim // 2, n_channels)
        )

    def forward(self, z_u):
        """
        z_u: [Batch, Latent_Dim]
        """
        # --- Channel Slicing ---
        z_u_chunks = z_u.view(-1, self.n_channels, self.channel_dim)
        item_chunks = self.item_embeddings.weight.view(-1, self.n_channels, self.channel_dim)

        # --- Hypergraph Convolution ---
        refined_item_chunks_list = []
        for k in range(self.n_channels):
            e_k = item_chunks[:, k, :]


            e_k_refined = torch.mm(self.adj, e_k)


            e_k_refined = self.layer_norm(e_k_refined)
            e_k_refined = self.dropout(e_k_refined)

            refined_item_chunks_list.append(e_k_refined)

        refined_item_embs = torch.stack(refined_item_chunks_list, dim=1)  # [N, K, D]

        
        # [Batch, Item, K]
        scores_per_channel = torch.einsum('bkd,nkd->bnk', z_u_chunks, refined_item_embs)


        # gating_weights: [Batch, K]
        gating_logits = self.gating_network(z_u)

        gating_weights = F.softmax(gating_logits / self.tau, dim=1).unsqueeze(1)  # [Batch, 1, K]


        final_scores = torch.sum(scores_per_channel * gating_weights, dim=-1)  # [Batch, Item]

        return final_scores, z_u_chunks, refined_item_embs

    def cal_loss_infonce(self, emb1, emb2):
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)
        pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.ssl_temp)
        neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / self.ssl_temp), axis=1)
        loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
        loss /= pos_score.shape[0]
        return loss

    def calculate_disentangle_loss(self, z_u_chunks):
        z_norm = F.normalize(z_u_chunks, p=2, dim=-1)
        sim_matrix = torch.matmul(z_norm, z_norm.transpose(1, 2))
        mask = torch.eye(self.n_channels, device=z_u_chunks.device).unsqueeze(0)
        off_diag = sim_matrix * (1 - mask)
        loss = torch.mean(off_diag ** 2)
        return loss