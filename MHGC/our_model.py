import math
from typing import List
import sys

import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import torch.nn as nn
import scipy.io as sio
import random
import torch.nn.functional as F
from setup import args
import scipy.stats as stats
from torch.distributions.beta import Beta


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_z, n_dec_1, n_dec_2):
        super(AE, self).__init__()
        self.enc_1 = nn.Linear(n_enc_1, n_enc_2)
        self.enc_2 = nn.Linear(n_enc_2, n_z)
        self.z_layer = nn.Linear(n_z, n_z)

        self.dec_1 = nn.Linear(n_z, n_dec_1)
        self.dec_2 = nn.Linear(n_dec_1, n_dec_2)
        self.x_bar_layer = nn.Linear(n_dec_2, n_enc_1)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        z = self.z_layer(enc_h2)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        x_bar = self.x_bar_layer(dec_h2)

        return x_bar, z


class Model(nn.Module):

    def __init__(self, n_in, n_h, n_clusters, activation='prelu', negsamp_round_patch=args.negsamp_ratio_patch,
                 negsamp_round_context=args.negsamp_ratio_context, readout=args.readout):
        super(Model, self).__init__()
        self.read_mode = readout
        self.ae1 = nn.Linear(n_in, n_h)
        self.ae2 = nn.Linear(n_h, n_h)
        self.gcn_context = GCN(n_h, n_h, activation)
        self.gcn_patch = GCN(n_h, n_h, activation)
        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()

        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, n_h))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.weight = nn.Parameter(torch.Tensor(1, ))
        self.weight.data = torch.tensor(0.50000).to(args.device)

    def forward(self, seq1, adj, sparse=False, msk=None, samp_bias1=None, samp_bias2=None):
        seq1 = seq1.squeeze(0)
        adj = adj.squeeze(0)
        z = self.ae1(seq1)
        # z = self.ae2(z)
        # z = self.ae2(z)
        h_1 = self.gcn_context(z, adj, sparse)
        # h_2 = self.gcn_patch(z, adj, sparse)

        # similarity = z @ self.cluster_layer.T
        # # similarity = similarity.pow((args.v + 1.0) / 2.0)
        # # similarity = (similarity.t() / torch.sum(similarity, 1)).t()
        # similarity = F.softmax(similarity, dim=1)

        similarity = F.cosine_similarity(z.unsqueeze(2), self.cluster_layer.T.unsqueeze(0), dim=1)
        similarity = F.softmax(similarity, dim=-1)

        # q = 1.0 / \
        #     (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / args.v)
        # q = q.pow((args.v + 1.0) / 2.0)
        # q = (q.t() / torch.sum(q, 1)).t()

        # node_embed = self.weight * h_1 + (1 - self.weight) * h_2

        return h_1, similarity


class GCN(nn.Module):

    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.in_features = in_ft
        self.out_features = out_ft
        self.weight = nn.Parameter(torch.FloatTensor(in_ft, out_ft))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, seq, adj, sparse=False):
        support = torch.mm(seq, self.weight)
        output = torch.spmm(adj, support)
        output = torch.tanh(output)
        return output


class AvgReadout(nn.Module):

    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)


class MaxReadout(nn.Module):

    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq, 1).values


class MinReadout(nn.Module):

    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values


class WSReadout(nn.Module):

    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq, sim)
        out = torch.sum(out, 1)
        return out


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_epoch(model: Model, optimizer, features, adj, adj_hat, subgraphs, epoch, bmm_model):
    # train mode
    model.train()

    num_nodes = features.size(1)
    feature_size = features.size(2)

    all_index = list(range(num_nodes))
    random.shuffle(all_index)

    num_batches = math.ceil(num_nodes / args.batch_size)

    # node embedding
    Zs = []
    all_loss = []

    for batch_index in range(num_batches):
        # last batch
        if (batch_index == num_batches - 1):
            index = all_index[batch_index * args.batch_size:]
        else:
            index = all_index[batch_index *
                              args.batch_size: (batch_index + 1) * args.batch_size]

        current_batch_size = len(index)

        node_embed, q = model(features, adj)
        node_embed_hat, q_hat = model(features, adj_hat)

        bf = []
        bf_hat = []
        node_index = []
        for i in index:
            cur_feat = node_embed[subgraphs[i], :].unsqueeze(0)
            cur_feat_hat = node_embed_hat[subgraphs[i], :].unsqueeze(0)
            bf.append(cur_feat)
            bf_hat.append(cur_feat_hat)
            node_index.append(subgraphs[i][-1])

        bf = torch.cat(bf)
        bf_hat = torch.cat(bf_hat)
        subgraph_embed = torch.sum(bf, dim=1)
        subgraph_hat_embed = torch.sum(bf_hat, dim=1)
        node_embed_index = node_embed[node_index, :]
        node_embed_hat_index = node_embed_hat[node_index, :]

        q = q[node_index, :]
        q_hat = q_hat[node_index, :]

        p = target_distribution(q)
        p_hat = target_distribution(q_hat)

        N = node_embed_index.size(0) * 2
        if epoch < args.epoch_start:
            ns_loss, nn_loss, ss_loss, nn_sim = loss_cal(model, subgraph_embed, subgraph_hat_embed, node_embed_index, node_embed_hat_index)
        else:
            _, _, _, nn_sim = loss_cal(model, subgraph_embed, subgraph_hat_embed, node_embed_index, node_embed_hat_index)

            q_sim = q.unsqueeze(1) * (q[:, None] / p).log()
            q_sim = torch.sum(q_sim, dim=-1)

            q_sim_hat = q_hat.unsqueeze(1) * (q_hat[:, None] / p_hat).log()
            q_sim_hat = torch.sum(q_sim_hat, dim=-1)

            q_sim = torch.cat([torch.cat([q_sim, q_sim], dim=1), torch.cat([q_sim_hat, q_sim_hat], dim=1)], dim=0)

            pseudo_labels = torch.argmax(q, dim=1)
            pseudo_labels_hat = torch.argmax(q_hat, dim=1)
            Q = torch.eq(pseudo_labels.unsqueeze(1), pseudo_labels.unsqueeze(0)).int()
            Q_hat = torch.eq(pseudo_labels_hat.unsqueeze(1), pseudo_labels_hat.unsqueeze(0)).int()
            Q = torch.cat([torch.cat([Q, Q], dim=1), torch.cat([Q_hat, Q_hat], dim=1)], dim=0)

            nn_sim = (nn_sim - nn_sim.min()) / (nn_sim.max() - nn_sim.min())

            if epoch == args.epoch_start:
                N_sel = 100
                index_fit = np.random.randint(0, N / 2, N_sel)
                sim_fit = q_sim[:, index_fit]
                sim_fit = (sim_fit + 1) / 2
                bmm_model.fit(sim_fit.flatten())
            q_sim_norm = q_sim.view(N, -1)
            q_sim_norm = (q_sim_norm + 1) / 2
            B = bmm_model.posterior(q_sim_norm, 0)
            B = torch.where(B < (1 - args.hard_ratio), torch.ones([N, N]).cuda(), torch.zeros([N, N]).cuda())
            B = torch.where(B == 0, B, torch.abs(Q - nn_sim) ** args.tao)
            B = torch.exp(B)
            ns_loss, nn_loss, ss_loss, nn_sim = loss_cal(model, subgraph_embed, subgraph_hat_embed, node_embed_index, node_embed_hat_index, B)

        Z = node_embed_index
        Zs.append(Z)
        # make cluster loss to the same order of magnitude
        loss_clu = F.kl_div(q.log(), p, reduction='batchmean')
        loss_ortho = torch.mean(F.pairwise_distance(q.T @ q, torch.eye(q.T.shape[0]).cuda(), p=2))

        # tgt_sim_t = torch.einsum('ij->ji', [tgt_sim])
        # sim_matrix = torch.einsum('ij,jk->ik', tgt_sim, tgt_sim_t)
        # loss_ortho = torch.mean(F.pairwise_distance(sim_matrix, torch.eye(tgt_sim.shape[0]).to('cuda'), p=2))

        loss = args.beta * ns_loss + (1 - args.beta) * nn_loss + args.alpha * ss_loss + args.gamma * loss_clu  # total loss

        all_loss.append(loss.item())

        # optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        Zs = torch.cat(Zs)
        scatter_index = torch.tensor(all_index).unsqueeze(
            1).expand_as(Zs).to(args.device)
        Z = torch.zeros_like(Zs).to(args.device).scatter_(0, scatter_index, Zs)
        loss = sum(all_loss) / len(all_loss)
        return Z, loss


def loss_cal(model, subgraph_embed, subgraph_hat_embed, node_embed_index, node_embed_hat_index, B=1):
    node_num = node_embed_index.size(0)
    mask = torch.ones([node_num * 2, node_num * 2]) - torch.eye(node_num * 2)

    # node-node similarity
    node_embed_index = F.normalize(node_embed_index, dim=1, p=2)
    node_embed_hat_index = F.normalize(node_embed_hat_index, dim=1, p=2)
    subgraph_embed = F.normalize(subgraph_embed, dim=1, p=2)
    subgraph_hat_embed = F.normalize(subgraph_hat_embed, dim=1, p=2)
    nn_sim = torch.cat([torch.cat([node_embed_index @ node_embed_index.T, node_embed_index @ node_embed_hat_index.T], dim=1),
                        torch.cat([node_embed_hat_index @ node_embed_hat_index.T, node_embed_hat_index @ node_embed_index.T], dim=1)], dim=0)

    # node-subgraph contrast loss
    ns_sim = torch.cat([torch.cat([node_embed_index @ subgraph_embed.T, node_embed_index @ subgraph_hat_embed.T], dim=1),
                        torch.cat([node_embed_hat_index @ subgraph_hat_embed.T, node_embed_hat_index @ subgraph_embed.T], dim=1)], dim=0)
    pos_neg = mask.cuda() * torch.exp(ns_sim * B)
    pos = torch.cat([torch.diag(ns_sim, node_num), torch.diag(ns_sim, -node_num)], dim=0)
    pos = torch.exp(pos)
    neg = (torch.sum(pos_neg, dim=1) - pos)
    ns_loss = (-torch.log(pos / (pos + neg))).sum() / (2 * node_num)

    # node-node contrast loss
    pos_neg = mask.cuda() * torch.exp(nn_sim * B)
    pos = torch.cat([torch.diag(nn_sim, node_num), torch.diag(nn_sim, -node_num)], dim=0)
    pos = torch.exp(pos)
    neg = (torch.sum(pos_neg, dim=1) - pos)
    nn_loss = (-torch.log(pos / (pos + neg))).sum() / (2 * node_num)

    # subgraph-subgraph contrast loss
    ss_sim = torch.cat([torch.cat([subgraph_embed @ subgraph_embed.T, subgraph_embed @ subgraph_hat_embed.T], dim=1),
                        torch.cat([subgraph_hat_embed @ subgraph_hat_embed.T, subgraph_hat_embed @ subgraph_embed.T], dim=1)], dim=0)
    pos_neg = mask.cuda() * torch.exp(ss_sim * B)
    pos = torch.cat([torch.diag(ss_sim, node_num), torch.diag(ss_sim, -node_num)], dim=0)
    pos = torch.exp(pos)
    neg = (torch.sum(pos_neg, dim=1) - pos)
    ss_loss = (-torch.log(pos / (pos + neg))).sum() / (2 * node_num)

    return ns_loss, nn_loss, ss_loss, nn_sim


def aug_random_edge(A, drop_percent=0.2):
    """
    randomly delect partial edges and
    randomly add the same number of edges in the graph
    """
    # input_adj = sp.csr_matrix(A.numpy(force=True))
    input_adj = sp.csr_matrix(A.numpy())

    percent = drop_percent / 2
    row_idx, col_idx = input_adj.nonzero()
    num_drop = int(len(row_idx) * percent)

    edge_index = [i for i in range(len(row_idx))]
    edges = dict(zip(edge_index, zip(row_idx, col_idx)))
    drop_idx = random.sample(edge_index, k=num_drop)

    list(map(edges.__delitem__, filter(edges.__contains__, drop_idx)))

    new_edges = list(zip(*list(edges.values())))
    new_row_idx = new_edges[0]
    new_col_idx = new_edges[1]
    data = np.ones(len(new_row_idx)).tolist()

    new_adj = sp.csr_matrix(
        (data, (new_row_idx, new_col_idx)), shape=input_adj.shape)

    row_idx, col_idx = (new_adj.todense() - 1).nonzero()
    no_edges_cells = list(zip(row_idx, col_idx))
    add_idx = random.sample(no_edges_cells, num_drop)
    new_row_idx_1, new_col_idx_1 = list(zip(*add_idx))
    row_idx = new_row_idx + new_row_idx_1
    col_idx = new_col_idx + new_col_idx_1
    data = np.ones(len(row_idx)).tolist()

    new_adj = sp.csr_matrix((data, (row_idx, col_idx)), shape=input_adj.shape)

    return new_adj


def our_normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


# polyfill for random_walk_with_restart in dgl 0.4.1
rwr_cache = None
cached_a = None


def build_rwr_cache(A):
    global rwr_cache
    global cached_a
    cached_a = A
    rwr_cache = dict()
    for node in range(A.size(0)):
        row = A[node]
        mask = row > 0
        rwr_cache[node] = torch.nonzero(mask)


def random_walk_with_restart(A, nodes: List[int], restart_prob: float, max_nodes_per_seed: int):
    global rwr_cache
    if not rwr_cache or cached_a is not A:
        build_rwr_cache(A)

    traces = []
    for node in nodes:
        start_node = torch.tensor([node])
        current_node = start_node
        trace = []
        while len(trace) < max_nodes_per_seed:
            if (random.random() < restart_prob):
                current_node = start_node
            neighbours = rwr_cache[current_node.item()]
            if len(neighbours) != 0:
                current_node = random.choice(rwr_cache[current_node.item()])
            trace.append(current_node)
        traces.append(trace)
    return traces


def generate_rwr_subgraph(A):
    all_idx = list(range(A.size(0)))
    reduced_size = args.subgraph_size - 1
    traces = random_walk_with_restart(
        A, all_idx, restart_prob=1, max_nodes_per_seed=args.subgraph_size * 3)
    subv = []

    for i, trace in enumerate(traces):
        subv.append(torch.unique(torch.cat(trace), sorted=False).tolist())
        retry_time = 0
        while len(subv[i]) < reduced_size:
            cur_trace = random_walk_with_restart(
                A, [i], restart_prob=0.9, max_nodes_per_seed=args.subgraph_size * 5)
            subv[i] = torch.unique(
                torch.cat(cur_trace[0]), sorted=False).tolist()
            retry_time += 1
            if (len(subv[i]) <= 2) and (retry_time > 10):
                subv[i] = (subv[i] * reduced_size)
        subv[i] = subv[i][:reduced_size]
        subv[i].append(i)

    return subv


def weighted_mean(x, w):
    return torch.sum(w * x) / torch.sum(w)


def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar) ** 2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) / x_bar
    return alpha, beta


class BetaMixture1D(object):
    def __init__(self, max_iters,
                 alphas_init,
                 betas_init,
                 weights_init):
        self.alphas = alphas_init
        self.betas = betas_init
        self.weight = weights_init
        self.max_iters = max_iters
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        x_cpu = x.cpu().detach().numpy()
        alpha_cpu = self.alphas.cpu().detach().numpy()
        beta_cpu = self.betas.cpu().detach().numpy()
        return torch.from_numpy(stats.beta.pdf(x_cpu, alpha_cpu[y], beta_cpu[y])).to(x.device)

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return self.weighted_likelihood(x, 0) + self.weighted_likelihood(x, 1)

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r = torch.cat((self.weighted_likelihood(x, 0).view(1, -1), self.weighted_likelihood(x, 1).view(1, -1)), 0)
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(0)
        return r

    def fit(self, x):
        eps = 1e-12
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):
            # E-step
            r = self.responsibilities(x)
            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            if self.betas[1] < 1:
                self.betas[1] = 1.01
            self.weight = r.sum(1)
            self.weight /= self.weight.sum()
        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)
