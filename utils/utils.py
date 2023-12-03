import random
import dgl
from dgl.data import register_data_args, citation_graph
from mpl_toolkits.mplot3d import Axes3D
from ogb.nodeproppred import DglNodePropPredDataset
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from config.get_args import get_my_args
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


def node_mask(train_mask, mask_rate):
    mask_rate = int(mask_rate * 10)
    count = 0
    for i in range(train_mask.shape[0]):
        if train_mask[i]:
            count = count + 1
            # 连续mask 10*mask_rate个训练样本
            if count <= mask_rate:
                train_mask[i] = False
                count = count + 1
            if count == 10:
                count = 0
    return train_mask


def load_cite(args):
    if args.dataset == 'cora':
        return citation_graph.CoraGraphDataset(raw_dir=args.dataset_dir, verbose=False)
    elif args.dataset == 'citeseer':
        return citation_graph.CiteseerGraphDataset(raw_dir=args.dataset_dir, verbose=False)
    elif args.dataset == 'pubmed':
        return citation_graph.PubmedGraphDataset(raw_dir=args.dataset_dir, verbose=False)
    elif args.dataset is not None and args.dataset.startswith('reddit'):
        # self_loop=('self-loop' in args.dataset),
        return dgl.data.RedditDataset(raw_dir=args.dataset_dir, verbose=False)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))


def my_load_data(args):
    if args.dataset == 'cora' or args.dataset == 'citeseer' or args.dataset == 'pubmed' or args.dataset == 'reddit':

        data = load_cite(args)
        g = data[0]
        features = g.ndata['feat']
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        if args.half:
            train_mask = node_mask(train_mask, args.mask_rate)
        else:
            train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        in_feats = features.shape[1]
        n_classes = data.num_classes
        n_edges = data.graph.number_of_edges()
    elif args.dataset == 'Fraud_yelp' or args.dataset == 'Fraud_amazon':
        if args.dataset == 'Fraud_yelp':
            data = dgl.data.FraudDataset('yelp')
        else:
            data = dgl.data.FraudDataset('amazon')
        g = data[0]
        g = dgl.to_homogeneous(g, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
        features = g.ndata['feature'].to(torch.float32)
        labels = g.ndata['label'].view(1, -1)
        train_mask = g.ndata['train_mask']
        if args.half:
            train_mask = node_mask(train_mask, args.mask_rate)
        else:
            train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        in_feats = features.shape[1]
        n_classes = data.num_classes
        n_edges = data.graph.number_of_edges()
    elif args.dataset == 'CoraFull':
        data = dgl.data.CoraFullDataset(raw_dir=args.dataset_dir)
        g = data[0]
        features = g.ndata['feat']
        labels = g.ndata['label']
        ind = torch.Tensor(random.choices([0, 1, 2], weights=[0.3, 0.1, 0.6], k=features.shape[0]))
        g.ndata['train_mask'] = (ind == 0)
        g.ndata['val_mask'] = (ind == 1)
        g.ndata['test_mask'] = (ind == 2)
        train_mask = g.ndata['train_mask']
        if args.half:
            train_mask = node_mask(train_mask, args.mask_rate)
        else:
            train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        in_feats = features.shape[1]
        n_classes = data.num_classes
        n_edges = g.number_of_edges()
    elif args.dataset == 'AmazonCoBuyComputer' or args.dataset == 'AmazonCoBuyPhoto' or args.dataset == 'CoauthorCS':
        data = None
        if args.dataset == 'AmazonCoBuyComputer':
            data = dgl.data.AmazonCoBuyComputerDataset(raw_dir=args.dataset_dir)
        elif args.dataset == 'AmazonCoBuyPhoto':
            data = dgl.data.AmazonCoBuyPhotoDataset(raw_dir=args.dataset_dir)
        elif args.dataset == 'CoauthorCS':
            data = dgl.data.CoauthorCSDataset(raw_dir=args.dataset_dir)
        g = data[0]
        features = g.ndata['feat']
        labels = g.ndata['label']
        ind = torch.Tensor(random.choices([0, 1, 2], weights=[0.1, 0.3, 0.6], k=features.shape[0]))
        g.ndata['train_mask'] = (ind == 0)
        g.ndata['val_mask'] = (ind == 1)
        g.ndata['test_mask'] = (ind == 2)
        train_mask = g.ndata['train_mask']
        if args.half:
            train_mask = node_mask(train_mask, args.mask_rate)
        else:
            train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        in_feats = features.shape[1]
        n_classes = data.num_classes
        n_edges = g.number_of_edges()
    elif args.dataset == 'ogbn-arxiv':
        dataset = DglNodePropPredDataset(name='ogbn-arxiv')

        split_idx = dataset.get_idx_split()
        g, labels = dataset[0]
        g = dgl.add_reverse_edges(g)
        features = g.ndata['feat']
        g.ndata['label'] = labels.view(-1, )
        ind = torch.zeros(labels.shape, dtype=bool)
        ind[split_idx['train']] = True
        g.ndata['train_mask'] = ind.view(-1, )
        ind = torch.zeros(labels.shape, dtype=bool)
        ind[split_idx['valid']] = True
        g.ndata['val_mask'] = ind.view(-1, )
        ind = torch.zeros(labels.shape, dtype=bool)
        ind[split_idx['test']] = True
        g.ndata['test_mask'] = ind.view(-1, )
        train_mask = g.ndata['train_mask']
        if args.half:
            train_mask = node_mask(train_mask, args.mask_rate)
        else:
            train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        in_feats = features.shape[1]
        n_classes = dataset.num_classes
        n_edges = g.number_of_edges()
        labels = labels.view(-1, )
    else:
        g = None
        features = None
        labels = None
        train_mask = None
        val_mask = None
        test_mask = None
        in_feats = None
        n_classes = None
        n_edges = None

    return g, features, labels, train_mask, val_mask, test_mask, in_feats, n_classes, n_edges


def evaluate(model, graph, nid, batch_size, device, sample_list):
    args = get_my_args()
    sampler = dgl.dataloading.MultiLayerNeighborSampler(sample_list)
    valid_dataloader = dgl.dataloading.NodeDataLoader(graph, nid, sampler, batch_size=batch_size, shuffle=False,
                                                      drop_last=False, num_workers=args.num_workers, device=device)
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for input_nodes, output_nodes, mfgs in valid_dataloader:
            inputs = mfgs[0].srcdata['feat']
            lab = mfgs[-1].dstdata['label']
            labels.append(lab.cpu().numpy())
            pred = model(mfgs)
            predictions.append(pred.argmax(1).cpu().numpy())
            # predictions.append(model(mfgs).argmax(1).cpu().numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        accuracy = accuracy_score(labels, predictions)
        f_score = f1_score(labels, predictions, average='weighted')
    return accuracy, f_score


def constraint(device, proto):
    if isinstance(proto, list):
        total = 0
        for p in proto:
            total = total + torch.norm(torch.mm(p, p.T) - torch.eye(p.shape[0]).to(device))
        return total / len(proto)
    else:
        return torch.norm(torch.mm(proto, proto.T) - torch.eye(proto.shape[0]).to(device))


def get_device():
    args = get_my_args()
    if args.gpu < 0:
        device = 'cpu'
    else:
        device = 'cuda:' + str(args.gpu)
        torch.cuda.set_device(args.gpu)
    return device


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_init_info(args):
    g, features, labels, train_mask, val_mask, test_mask, in_feats, n_classes, n_edges = my_load_data(args)
    print(f"""----{args.dataset} Data statistics------'
    NumNodes: {features.shape[0]}
    NumEdges:{n_edges}
    NumFeats:{features.shape[1]}
    NumClasses:{n_classes}
    NumTrainingSamples:{train_mask.int().sum().item()}
    NumTestSamples:{test_mask.int().sum().item()}
    NumValidationSamples:{val_mask.int().sum().item()}
        """)
    device = get_device()
    # train_mask = train_mask
    # val_mask = val_mask
    # test_mask = test_mask

    g = dgl.remove_self_loop(g)
    n_edges = g.number_of_edges()
    train_nid = torch.LongTensor(np.nonzero(train_mask)).squeeze()
    val_nid = torch.LongTensor(np.nonzero(val_mask)).squeeze()
    test_nid = torch.LongTensor(np.nonzero(test_mask)).squeeze()
    return g, features, labels, in_feats, n_classes, n_edges, train_nid, val_nid, test_nid, device


def save_res(acc_all, num, args, dir_name):
    data = pd.DataFrame(acc_all)
    file_name = ''
    if args.half and args.mask_rate == 0.3:
        file_name = 'few_shot30'
    elif args.half and args.mask_rate == 0.5:
        file_name = 'few_shot50'
    elif not args.half:
        file_name = 'full'
    else:
        file_name = '其它'
    print(file_name, ',', num)
    if not os.path.exists(f'./res/{dir_name}/{file_name}/' + args.dataset):
        os.makedirs(f'./res/{dir_name}/{file_name}/' + args.dataset + '/')
    data.to_csv(f'./res/{dir_name}/{file_name}/' + args.dataset + f'/self_{num}.csv', index=False, header=False)


class Prototype(nn.Module):
    def __init__(self, h_feats, num_classes):
        super(Prototype, self).__init__()
        self.proto = nn.Parameter(torch.randn(num_classes, h_feats))
        # self.lin = nn.Linear(h_feats, h_feats)
        # self.fc = nn.Linear(h_feats, num_classes)

    def setInitPara(self, dd):
        self.proto.data.copy_(dd)

    def getDistance(self, x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        # print(x.shape,y.shape) # torch.Size([way*qshots, dim]) torch.Size([way, dim])
        assert d == y.size(1)
        # torch.Size([qnums, snums, dim])
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        return torch.sqrt(torch.pow(x - y, 2).sum(2))

    def forward(self, features):
        # h = self.lin(features)
        # h = self.lin(features)
        h = torch.cdist(features, self.proto)
        # x=self.getDistance(features,self.proto)
        return -h




if __name__ == '__main__':
    pass
