import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score, silhouette_score
from scipy.optimize import linear_sum_assignment
import random
import os
import time
import torch
import scipy
import sklearn
import numpy
import argparse
import yaml

from dataset.data import load
from model.AttributeRandomWalk import RandomWalkPriorLearner
from model.HGNN import HGNN



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cluster_metrics(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    map_dict = {row: col for row, col in zip(row_ind, col_ind)}
    y_pred_aligned = np.array([map_dict.get(x, x) for x in y_pred])
    acc = w[row_ind, col_ind].sum() / y_pred.size
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred_aligned, average='macro')
    return acc, f1, nmi, ari


def dmon_regularization(probs):
    n_nodes = probs.shape[0]
    n_clusters = probs.shape[1]
    c_t_c = torch.matmul(probs.t(), probs)
    c_t_c = c_t_c / n_nodes
    norm_fro = torch.norm(c_t_c, p='fro')
    loss = norm_fro - (1.0 / n_clusters ** 0.5)
    return loss


def compute_honorcut_losses_refined(S, Pi_prior, d_pi_stable):
    device = S.device
    num_clusters = S.shape[1]

    DS = S * d_pi_stable.unsqueeze(1)
    vol_mat = torch.mm(S.t(), DS)
    if Pi_prior.is_sparse:
        AS = torch.sparse.mm(Pi_prior, S)
    else:
        AS = torch.mm(Pi_prior, S)
    StAS = torch.mm(S.t(), AS)
    cut_mat = vol_mat - StAS
    identity = torch.eye(num_clusters).to(device)
    loss_c = torch.trace(torch.linalg.pinv(vol_mat + 1e-4 * identity) @ cut_mat)
    return loss_c


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def contrastive_loss(emb, adj, batch_size, tau=0.5):
    device = emb.device
    num_nodes = emb.shape[0]
    if num_nodes > batch_size:
        batch_idx = torch.randperm(num_nodes)[:batch_size].to(device)
    else:
        batch_idx = torch.arange(num_nodes).to(device)

    emb = F.normalize(emb, p=2, dim=1)
    anchor_emb = emb[batch_idx]
    sim_mat = torch.mm(anchor_emb, emb.t())
    sim_mat = torch.div(sim_mat, tau)
    adj_batch = torch.index_select(adj, 0, batch_idx)
    pos_mask = adj_batch.to_dense() > 0
    pos_mask = pos_mask.float()
    exp_sim = torch.exp(sim_mat)
    sum_exp = exp_sim.sum(dim=1, keepdim=True)
    log_prob = sim_mat - torch.log(sum_exp + 1e-12)
    has_pos = pos_mask.sum(1) > 0
    if has_pos.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    loss = - (log_prob * pos_mask).sum(dim=1)
    loss = loss[has_pos].mean()
    return loss


def train_hyper(args):
    setup_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading dataset: {args.dataset}...")
    dataset = load(args.dataset)

    features = dataset['features']
    if sp.issparse(features): features = features.toarray()
    H_raw = dataset['adj']
    labels = dataset['labels']
    y_true = np.argmax(labels, axis=1) if labels.ndim == 2 else labels

    if H_raw.shape[0] != features.shape[0]:
        if H_raw.shape[1] == features.shape[0]: H_raw = H_raw.T

    x = torch.from_numpy(features).float().to(device)
    n_clusters = len(np.unique(y_true))
    x_norm = F.normalize(x, p=2, dim=1)

    if not sp.issparse(H_raw):
        H_scipy = sp.csc_matrix(H_raw)
    else:
        H_scipy = H_raw.tocsc()
    H_coo = H_scipy.tocoo()
    indices = torch.from_numpy(np.vstack((H_coo.row, H_coo.col)).astype(np.int64))
    values = torch.from_numpy(H_coo.data.astype(np.float32))
    H_incidence = torch.sparse_coo_tensor(indices, values, H_coo.shape).to(device)

    deg_v = np.array(H_scipy.sum(1)).flatten()
    deg_e = np.array(H_scipy.sum(0)).flatten()
    dv_inv_sqrt = torch.from_numpy(np.power(deg_v + 1e-12, -0.5)).float().to(device)
    de_inv = torch.from_numpy(np.power(deg_e + 1e-12, -1.0)).float().to(device)

    if torch.cuda.is_available(): torch.cuda.synchronize()
    start_time = time.time()

    Structure = RandomWalkPriorLearner(steps=args.rw_steps, alpha=args.alpha_appnp, beta=args.beta).to(device)
    Structure.compute_static_prior(x, H_incidence, n_clusters)
    A_target, Z_target, svd_dim = Structure()


    model = HGNN(x.shape[1], args.hidden, n_clusters, svd_dim=svd_dim, dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    print(f"\n>>> Start Training...")
    Q_target = None

    if A_target.is_sparse:
        d_pi = torch.sparse.mm(A_target, torch.ones(x.shape[0], 1).to(device)).squeeze()
    else:
        d_pi = A_target.sum(dim=1)
    d_pi_stable = d_pi + 1e-4 * d_pi.mean()

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        logits, final_emb, x_rec, z_contrast = model(x, H_incidence, dv_inv_sqrt, de_inv)

        curr_temp = max(0.1, 1.0 * (0.995 ** epoch))
        S = F.softmax(logits / curr_temp, dim=1)

        if epoch % 10 == 0 or Q_target is None:
            with torch.no_grad():
                Q_target = target_distribution(S)

        loss_c = compute_honorcut_losses_refined(S, A_target, d_pi_stable)
        loss_balance = dmon_regularization(S)
        loss_rec = 1.0 - F.cosine_similarity(x_rec, x_norm, dim=1).mean()
        loss_kl = F.kl_div(S.log(), Q_target, reduction='batchmean')


        loss = args.gamma_ncut * loss_c + \
               args.gamma_balance * loss_balance + \
               args.gamma_rec * loss_rec + \
               args.kl * loss_kl

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) == args.epochs:
            y_pred = torch.argmax(S, dim=1).cpu().numpy()
            acc, f1, nmi, ari = cluster_metrics(y_true, y_pred)

    if torch.cuda.is_available(): torch.cuda.synchronize()
    end_time = time.time()

    print(f"\n====================================")
    print(f"Final Epoch | ACC: {acc:.4f} | NMI: {nmi:.4f} | F1: {f1:.4f} | ARI: {ari:.4f}")
    print(f"Effective Time: {end_time - start_time:.2f}s")
    print(f"====================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LNAHC Hypergraph Clustering")
    parser.add_argument('--dataset', type=str, default='cocitation/cora', help='Name of the dataset to run. Include: '
                                    'cocitation/cora, cocitation/citeseer, cocitation/pubmed, 20newsW100, Mushroom, NTU2012, ModelNet40')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML configuration file')
    cmd_args = parser.parse_args()

    if not os.path.exists(cmd_args.config):
        raise FileNotFoundError(f"can't find yaml file: {cmd_args.config}")

    with open(cmd_args.config, 'r', encoding='utf-8') as f:
        all_configs = yaml.safe_load(f)

    if cmd_args.dataset not in all_configs:
        raise ValueError(f"can't find parameter set for '{cmd_args.dataset}' in {cmd_args.config}")

    dataset_config = all_configs[cmd_args.dataset]


    class ConfigArgs:
        def __init__(self, dataset_name, config_dict):
            self.dataset = dataset_name
            for key, value in config_dict.items():
                if key in ['lr', 'weight_decay']:
                    value = float(value)
                setattr(self, key, value)


    args = ConfigArgs(cmd_args.dataset, dataset_config)

    print(f" Loaded Configuration for: {args.dataset}")


    train_hyper(args)