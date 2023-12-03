import argparse
from dgl.data import register_data_args


def get_my_args():
    parser = argparse.ArgumentParser(description='PyTorch')
    # register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
    parser.add_argument("--lr_c", type=float, default=1, help="learning rate")
    parser.add_argument("--lr_p", type=float, default=0.01, help="proto learning rate")
    # parser.add_argument("--lr_temperature", type=float, default=1, help="temperature learning rate")
    parser.add_argument("--seed", type=int, default=200, help="random seed")
    parser.add_argument("--n-hidden", type=int, default=128, help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=2, help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument("--aggregator-type", type=str, default="gcn")
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--sample-list', type=list, default=[4, 4])

    parser.add_argument("--n-epochs", type=int, default=200, help="number of training epochs")
    parser.add_argument("--file-id", type=str, default='128')
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--half', type=bool, default=False)
    parser.add_argument('--mask_rate', type=float, default=0.5)
    parser.add_argument('--center_num', type=int, default=3)
    parser.add_argument('--dataset_dir', type=str, default='./dataset/')
    parser.add_argument("--gpu", type=int, default=1, help="gpu")
    parser.add_argument("--dataset", type=str, default='cora', help="data",choices=['cora','citeseer','pubmed','CoraFull', 'AmazonCoBuyComputer','AmazonCoBuyPhoto','ogbn-arxiv', 'reddit'])
    parser.add_argument("--save_flag", type=bool, default=False)
    parser.add_argument("--datasets_list", type=str,  help="data_list",
                        default=['cora', 'citeseer', 'pubmed', 'CoraFull', 'AmazonCoBuyComputer', 'AmazonCoBuyPhoto',
                                 'ogbn-arxiv', 'reddit'])

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    pass
