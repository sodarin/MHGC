import argparse

parser = argparse.ArgumentParser()

# dataset
parser.add_argument('--device', type=str, default="cpu", help='device.')
parser.add_argument('--dataset', type=str, default='cora', help='dataset name')
parser.add_argument('--cluster_num', type=int,
                    default=7, help='cluster number')

# pre-process
parser.add_argument('--n_input', type=int, default=500,
                    help='input feature dimension')
parser.add_argument('--t', type=int, default=2,
                    help="filtering time of Laplacian filters")

# network
parser.add_argument('--dims', type=int, default=[1500], help='hidden unit')
parser.add_argument('--negsamp_ratio_patch', type=int, default=6)
parser.add_argument('--negsamp_ratio_context', type=int, default=1)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--beta', type=float, default=0.2)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--readout', default='avg')
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--v', type=float, default=1.0)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--hard_ratio', type=float, default=0.5)
parser.add_argument('--tao', type=int, default=2)
parser.add_argument('--epoch_start', type=int, default=10)
parser.add_argument('--iters', type=int, default=10)
parser.add_argument('--weight_init', type=float, default=0.05)

# training
parser.add_argument('--runs', type=int, default=5, help='runs')
parser.add_argument('--epochs', type=int, default=100, help='training epoch')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--acc', type=float, default=0, help='acc')
parser.add_argument('--nmi', type=float, default=0, help='nmi')
parser.add_argument('--ari', type=float, default=0, help='ari')
parser.add_argument('--f1', type=float, default=0, help='f1')

args = parser.parse_args()
