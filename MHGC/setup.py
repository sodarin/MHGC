from opt import args


def setup_args(dataset_name="cora"):
    args.dataset = dataset_name
    args.device = "cuda:0"
    args.acc = args.nmi = args.ari = args.f1 = 0

    args.negsamp_ratio_patch = 6
    args.negsamp_ratio_context = 1
    args.alpha = 0.1
    args.beta = 0.2
    args.gamma = 0.1
    args.readout = 'avg'
    args.subgraph_size = 4
    args.v = 1.0
    args.batch_size = 1000

    if args.dataset == 'cora':
        args.t = 2
        args.lr = 1e-3
        args.n_input = 500
        args.dims = 1500

    elif args.dataset == 'citeseer':
        args.t = 2
        args.lr = 1e-3
        args.n_input = 500
        args.dims = 1500

    elif args.dataset == 'amap':
        args.t = 8
        args.lr = 1e-5
        args.n_input = -1
        args.dims = 500

    elif args.dataset == 'bat':
        args.t = 8
        args.lr = 1e-3
        args.n_input = -1
        args.dims = 1500

    elif args.dataset == 'eat':
        args.t = 8
        args.lr = 1e-4
        args.n_input = -1
        args.dims = 1500

    elif args.dataset == 'uat':
        args.t = 6
        args.lr = 1e-4
        args.n_input = -1
        args.dims = 500

    # other new datasets
    else:
        args.t = 2
        args.lr = 1e-3
        args.n_input = 500
        args.dims = 1500

    print("---------------------")
    print("runs: {}".format(args.runs))
    print("dataset: {}".format(args.dataset))
    print("---------------------")

    return args
