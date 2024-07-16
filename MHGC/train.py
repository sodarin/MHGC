from sklearn.cluster import KMeans

from utils import *
from tqdm import tqdm
from torch import optim
from setup import setup_args
from model import hard_sample_aware_network
from our_model import train_epoch, Model, aug_random_edge, our_normalize_adj, generate_rwr_subgraph, BetaMixture1D
import scipy.sparse as sp
import itertools

dataset_name = "bat"
result_filename = f'result_{dataset_name}_bmm_2.csv'

if __name__ == '__main__':
    torch.set_num_threads(1)

    # for dataset_name in ["bat", "eat", "uat"]:
    # for dataset_name in ["citeseer", "amap"]:
    # for dataset_name in ["cora"]:
    # setup hyper-parameter
    args = setup_args(dataset_name)
    # gamma_list = [0.01, 0.1, 1]
    # gamma_list = [1, 3, 5]
    # gamma_list = [2, 3, 4]
    # gamma_list = [0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
    gamma_list = [4]
    # subgraph_size_list = [4, 8, 12, 16, 20]
    # subgraph_size_list = [12, 14, 16, 18, 20]
    subgraph_size_list = [4]
    # lr_list = [1e-3, 1e-4, 1e-5]
    # lr_list = [5e-4, 1e-4, 5e-5]
    lr_list = [0.001]
    # lr_list = [1e-5, 1e-6]
    # dims_list = [100, 500, 1500]
    # dims_list = [60, 80, 100, 120]
    dims_list = [1000]

    alpha_list = [0.1]
    beta_list = [0.2]

    # prop_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    prop_list = [0.2]
    t_list = [2, 4, 6, 8]

    for args.gamma, args.subgraph_size, args.lr, args.dims, args.alpha, args.beta, prop_rate, args.t in itertools.product(
            gamma_list, subgraph_size_list,
            lr_list, dims_list, alpha_list,
            beta_list, prop_list, t_list):

        args.acc = args.nmi = args.ari = args.f1 = 0

        # record results
        file = open(result_filename, "a+")
        print(args.dataset, file=file)
        print(f'gamma: {args.gamma}, subgraph_size: {args.subgraph_size}, lr: {args.lr}, dims: {args.dims}, alpha: {args.alpha}, beta: {args.beta}, '
              f'prop_rate: {prop_rate}, t: {args.t}', file=file)
        print(dataset_name)
        print(f'gamma: {args.gamma}, subgraph_size: {args.subgraph_size}, lr: {args.lr}, dims: {args.dims}, alpha: {args.alpha}, beta: {args.beta}, '
              f'prop_rate: {prop_rate}, t: {args.t}')
        print("ACC,   NMI,   ARI,   F1", file=file)
        file.close()
        acc_list = []
        nmi_list = []
        ari_list = []
        f1_list = []

        # ten runs with different random seeds
        for args.seed in range(args.runs):
            # record results

            # fix the random seed
            setup_seed(args.seed)

            # load graph data
            X, y, A, node_num, cluster_num = load_graph_data(
                dataset_name, show_details=False)

            # apply the laplacian filtering
            X_filtered = laplacian_filtering(A, X, args.t)
            # X_filtered = torch.from_numpy(X)

            # test
            args.acc, args.nmi, args.ari, args.f1, y_hat, center = phi(
                X_filtered, y, cluster_num)

            # our model
            model = Model(X.shape[1], args.dims, cluster_num)

            # adam optimizer
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

            # positive and negative sample pair index matrix
            mask = torch.ones([node_num * 2, node_num * 2]
                              ) - torch.eye(node_num * 2)

            # graph data augmentation
            adj_edge_modification = aug_random_edge(A, prop_rate)

            adj = our_normalize_adj(A)
            adj = (adj + sp.eye(adj.shape[0])).todense()
            adj_hat = our_normalize_adj(adj_edge_modification)
            adj_hat = (adj_hat + sp.eye(adj_hat.shape[0])).todense()
            # adj_1 = A
            # adj_hat_1 = adj_edge_modification.todense()

            features = X_filtered.unsqueeze(0).to(args.device)
            adj = torch.FloatTensor(adj[np.newaxis]).to(args.device)
            adj_hat = torch.FloatTensor(adj_hat[np.newaxis]).to(args.device)

            # build subgraphs
            subgraphs = generate_rwr_subgraph(A)

            # load data to device
            A, model, X_filtered, mask = map(lambda x: x.to(
                args.device), (A, model, X_filtered, mask))

            alphas_init = torch.tensor([1, 2], dtype=torch.float64).cuda()
            betas_init = torch.tensor([2, 1], dtype=torch.float64).cuda()
            weights_init = torch.tensor([1 - args.weight_init, args.weight_init], dtype=torch.float64).cuda()
            bmm_model = BetaMixture1D(args.iters, alphas_init, betas_init, weights_init)

            # data = features.squeeze(0)
            # with torch.no_grad():
            #     x_bar, z = model.ae(data)
            #
            # kmeans = KMeans(n_clusters=cluster_num, n_init=20)
            # y_pred = kmeans.fit_predict(z.data.cpu().numpy())
            # y_pred_last = y_pred
            # model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).cuda()

            # training
            progress_bar = tqdm(range(100))
            for epoch in progress_bar:
                Z, loss = train_epoch(
                    model, optimizer, features, adj, adj_hat, subgraphs, epoch, bmm_model)
                progress_bar.set_description(f'loss: {loss:4f}')

                # testing
                if epoch % 1 == 0:
                    # evaluation mode
                    model.eval()

                    acc, nmi, ari, f1, P, center = phi(Z, y, cluster_num)

                    # recording
                    if acc >= args.acc:
                        args.acc, args.nmi, args.ari, args.f1 = acc, nmi, ari, f1

            print("Training complete")

            # record results
            file = open(result_filename, "a+")
            print("{:.2f}, {:.2f}, {:.2f}, {:.2f}".format(
                args.acc, args.nmi, args.ari, args.f1), file=file)
            file.close()
            acc_list.append(args.acc)
            nmi_list.append(args.nmi)
            ari_list.append(args.ari)
            f1_list.append(args.f1)

        # record results
        acc_list, nmi_list, ari_list, f1_list = map(
            lambda x: np.array(x), (acc_list, nmi_list, ari_list, f1_list))
        file = open(result_filename, "a+")
        print("{:.2f}, {:.2f}".format(
            acc_list.mean(), acc_list.std()), file=file)
        print("{:.2f}, {:.2f}".format(
            nmi_list.mean(), nmi_list.std()), file=file)
        print("{:.2f}, {:.2f}".format(
            ari_list.mean(), ari_list.std()), file=file)
        print("{:.2f}, {:.2f}".format(f1_list.mean(), f1_list.std()), file=file)
        file.close()
