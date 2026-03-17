import argparse
import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random


from models.vae_encoder import VAE_Encoder, reparameterize
from models.denoiser import DenoiseMLP
from models.mlp_decoder import MLPDecoder
from models.Dis_decoder import DisentangledHyperDecoder
import models.gaussian_diffusion as gd
import evaluate_utils
import data_utils

random_seed = 2024
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)


parser = argparse.ArgumentParser()

parser.add_argument('--latent_dim', type=int, default=128)
parser.add_argument('--encoder_dims', type=str, default='[1024, 512]')
parser.add_argument('--denoiser_dims', type=str, default='[512, 1024]')
parser.add_argument('--graph_layers', type=int, default=2)
parser.add_argument('--dataset', type=str, default='ml-1m')
parser.add_argument('--data_path', type=str, default='../datasets/')
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--epochs', type=int, default=600)
parser.add_argument('--topN', type=str, default='[10, 20, 50]')
parser.add_argument('--loss_beta', type=float, default=0.2)
parser.add_argument('--loss_gamma', type=float, default=100.0)
parser.add_argument('--mean_type', type=str, default='x0')
parser.add_argument('--steps', type=int, default=5)
parser.add_argument('--sampling_steps', type=int, default=2)
parser.add_argument('--w_min', type=float, default=0.1)
parser.add_argument('--patience', type=int, default=60)
parser.add_argument('--cuda', action='store_true', default=True)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--cl_rate', type=float, default=0.2)
parser.add_argument('--loss_delta', type=float, default=0.1)
parser.add_argument('--cl_temp', type=float, default=0.1)
parser.add_argument('--n_channels', type=int, default=4)
parser.add_argument('--loss_alpha', type=float, default=0.005)
parser.add_argument('--decoder_mode', type=str, default='full')
parser.add_argument('--disable_diffusion', action='store_true')
parser.add_argument('--tau', type=float, default=1.0, help='Temperature parameter for gating softmax')

def augment_interactions(batch_ori, dropout_rate=0.2):
    random_tensor = torch.rand(batch_ori.shape).to(batch_ori.device)
    dropout_mask = torch.where(random_tensor > dropout_rate, 1.0, 0.0)
    augmented_batch = batch_ori * dropout_mask
    return augmented_batch


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")

train_path = os.path.join(args.data_path, args.dataset, 'train_list.npy')
valid_path = os.path.join(args.data_path, args.dataset, 'valid_list.npy')
test_path = os.path.join(args.data_path, args.dataset, 'test_list.npy')

train_data_weighted, train_data_ori, valid_y_data, test_y_data, n_user, n_item, ii_graph, hyper_graph = data_utils.data_load(
    train_path, valid_path, test_path, w_min=args.w_min
)

hyper_graph = hyper_graph.to(device)

train_dataset = data_utils.DataDiffusion(train_data_ori, train_data_weighted)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                          worker_init_fn=worker_init_fn, num_workers=4)

eval_dataset = data_utils.DataDiffusion(train_data_ori, train_data_ori)
test_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
mask_tv = train_data_ori + valid_y_data



encoder_full_dims = [n_item] + eval(args.encoder_dims) + [args.latent_dim]
encoder = VAE_Encoder(dims=encoder_full_dims).to(device)

denoiser_full_dims = [args.latent_dim] + eval(args.denoiser_dims) + [args.latent_dim]
denoising_model = DenoiseMLP(dims=denoiser_full_dims, emb_size=128).to(device)

dis_decoder = DisentangledHyperDecoder(
    n_item=n_item,
    latent_dim=args.latent_dim,
    hyper_graph=hyper_graph,
    n_channels=args.n_channels,
    ssl_temp=args.cl_temp,
    dropout=0.3,
    tau=args.tau
).to(device)

mlp_decoder_dims = '[512]'
mlp_decoder = MLPDecoder(args.latent_dim, mlp_decoder_dims, n_item).to(device)

if args.mean_type == 'x0':
    mean_type = gd.ModelMeanType.START_X
else:
    mean_type = gd.ModelMeanType.EPSILON

diffusion = gd.GaussianDiffusion(mean_type, 'linear', 1.0, 0.0001, 0.02, args.steps, device)

optimizer = optim.AdamW(
    list(encoder.parameters()) +
    list(denoising_model.parameters()) +
    list(dis_decoder.parameters()) +
    list(mlp_decoder.parameters()),
    lr=args.lr,
    weight_decay=args.weight_decay
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

ALPHA_TRAIN = 0.4
W_INFERENCE = 0.793


def evaluate(data_loader, data_te, mask_his, topN):

    encoder.eval()
    denoising_model.eval()
    dis_decoder.eval()
    mlp_decoder.eval()

    target_items = [data_te[i, :].nonzero()[1].tolist() for i in range(data_te.shape[0])]
    predict_items = []

    with torch.no_grad():

        for batch_ori, _ in data_loader:
            batch_ori = batch_ori.to(device)
            mu, _ = encoder(batch_ori)
            denoised_z0 = diffusion.p_sample(
                denoising_model,
                mu,
                args.sampling_steps,
                sampling_noise=False
            )

            logits_graph, _, _ = dis_decoder(denoised_z0)
            logits_mlp = mlp_decoder(denoised_z0)
            prediction_logits = (1 - W_INFERENCE) * logits_mlp + W_INFERENCE * logits_graph
            start_idx = len(predict_items)
            end_idx = start_idx + batch_ori.shape[0]
            prediction_logits[mask_his[start_idx:end_idx].toarray() > 0] = -np.inf
            max_k = topN[-1]
            _, indices = torch.topk(prediction_logits, max_k)
            predict_items.extend(indices.cpu().numpy().tolist())

    eval_results = evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)

    return eval_results


# --- Training ---

best_valid_recall_20 = -1
best_epoch = 0
final_test_results = None
patience_counter = 0

print(f"Starting training. Latent: {args.latent_dim}. Gamma: {args.loss_gamma}. Alpha: {args.loss_alpha}")

topn_list = eval(args.topN)

for epoch in range(1, args.epochs + 1):

    encoder.train()
    denoising_model.train()
    dis_decoder.train()
    mlp_decoder.train()

    total_epoch_loss = 0.0
    total_diff_loss = 0.0  

    for batch_ori, batch_weighted in train_loader:
        batch_ori = batch_ori.to(device)
        batch_weighted = batch_weighted.to(device)
        optimizer.zero_grad()
        view1 = augment_interactions(batch_ori, dropout_rate=args.cl_rate)
        view2 = augment_interactions(batch_ori, dropout_rate=args.cl_rate)
        mu1, _ = encoder(view1)
        mu2, _ = encoder(view2)
        mu1_norm = F.normalize(mu1, dim=1)
        mu2_norm = F.normalize(mu2, dim=1)
        sim_matrix = torch.matmul(mu1_norm, mu2_norm.T)
        logits = sim_matrix / args.cl_temp
        labels = torch.arange(batch_ori.shape[0]).to(device)
        cl_loss = F.cross_entropy(logits, labels)
        mu_main, logvar_main = encoder(batch_ori)
        z0 = reparameterize(mu_main, logvar_main)
        diffusion_loss, predicted_z0 = diffusion.latent_training_losses(
            denoising_model, z0
        )

        logits_graph, z_chunks, _ = dis_decoder(predicted_z0)
        logits_mlp = mlp_decoder(predicted_z0)
        log_softmax_graph = F.log_softmax(logits_graph, dim=1)
        recon_loss_graph = -torch.mean(
            torch.sum(log_softmax_graph * batch_weighted, dim=1)
        )
        log_softmax_mlp = F.log_softmax(logits_mlp, dim=1)
        recon_loss_mlp = -torch.mean(
            torch.sum(log_softmax_mlp * batch_weighted, dim=1)
        )
        batch_recon_loss = (1 - ALPHA_TRAIN) * recon_loss_mlp + ALPHA_TRAIN * recon_loss_graph
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + logvar_main - mu_main.pow(2) - logvar_main.exp(), dim=1)
        )
        diff_loss = torch.mean(diffusion_loss)
        dis_loss = dis_decoder.calculate_disentangle_loss(z_chunks)
        total_loss = (
                batch_recon_loss
                + args.loss_beta * kl_loss
                + args.loss_gamma * diff_loss
                + args.loss_delta * cl_loss
                + args.loss_alpha * dis_loss
        )

        total_loss.backward()
        optimizer.step()
        total_epoch_loss += total_loss.item()
        total_diff_loss += diff_loss.item()

    scheduler.step()
    avg_loss = total_epoch_loss / len(train_loader)
    avg_diff_loss = total_diff_loss / len(train_loader)
    valid_results = evaluate(test_loader, valid_y_data, train_data_ori, topn_list)
    print(f"Epoch {epoch}/{args.epochs} | Total: {avg_loss:.4f} | DiffLoss: {avg_diff_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    print(
        f"Valid: R@10: {valid_results[1][0]:.4f} | "
        f"R@20: {valid_results[1][1]:.4f} | "
        f"N@10: {valid_results[2][0]:.4f} | "
        f"N@20: {valid_results[2][1]:.4f}"
    )

    current_valid_recall_20 = valid_results[1][1]

    if current_valid_recall_20 > best_valid_recall_20:
        best_valid_recall_20 = current_valid_recall_20
        best_epoch = epoch
        patience_counter = 0
        final_test_results = evaluate(test_loader, test_y_data, mask_tv, topn_list)
        print("*** Best Valid Recall@20 Found! Test evaluated. ***")

        print(
            f"Test : R@10: {final_test_results[1][0]:.4f} | "
            f"R@20: {final_test_results[1][1]:.4f} | "
            f"N@10: {final_test_results[2][0]:.4f} | "
            f"N@20: {final_test_results[2][1]:.4f}"
        )

    else:
        patience_counter += 1

    if patience_counter >= args.patience:
        print(f"Early stop at {epoch}")
        break


print(f"\nFinal Best Test Results (at Epoch {best_epoch}):")
print(f"Recall@10: {final_test_results[1][0]:.4f} | NDCG@10: {final_test_results[2][0]:.4f}")
print(f"Recall@20: {final_test_results[1][1]:.4f} | NDCG@20: {final_test_results[2][1]:.4f}")
