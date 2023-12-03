import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import SAGEConv
from config import get_args
from utils import utils
from utils.utils import *
import warnings
import random

warnings.filterwarnings("ignore")

class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        if dropout == False:
            self.dropout = False
        else:
            self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.layers.append(SAGEConv(in_feats, n_hidden,aggregator_type))
        for i in range(n_layers - 1):
            # n_hidden, n_hidden,aggregator_type
            self.layers.append(SAGEConv(n_hidden, n_hidden,aggregator_type))
        self.fc = nn.Linear(n_hidden, n_classes,bias=True)

    def forward(self, graph, output_nodes=None, Train_flag=False):
        inputs = graph[0].srcdata['feat']
        h=None
        if not self.dropout:
            h = inputs
        else:
            h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(graph[l], h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                if self.dropout:
                    h = self.dropout(h)
        h = self.activation(h)
        h = self.fc(h)
        return h



def main(args, num=0):
    utils.seed_torch(args.seed)
    g, features, labels, in_feats, n_classes, n_edges, train_nid, val_nid, test_nid, device = utils.get_init_info(args)
    sampler = dgl.dataloading.MultiLayerNeighborSampler(args.sample_list)
    train_dataloader = dgl.dataloading.NodeDataLoader(g, train_nid, sampler, device=device,
                                                      batch_size=args.batch_size, shuffle=True, drop_last=False,
                                                      num_workers=0)
    model = GraphSAGE(in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout, args.aggregator_type)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    acc_all = []
    loss_all = []
    f_score_all = []
    for epoch in range(args.n_epochs):
        model.train()
        acc,f_score = utils.evaluate(model, g, test_nid, args.batch_size, device, args.sample_list)
        acc_all.append(acc)
        f_score_all.append(f_score)
        t0 = time.time()
        for step, (input_nodes, output_nodes, mfgs) in enumerate(train_dataloader):
            inputs = mfgs[0].srcdata['feat']
            lab = mfgs[-1].dstdata['label']
            logits = model(mfgs, inputs)
            loss = F.cross_entropy(logits, lab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch {:03d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} ".format(epoch, time.time() - t0,
                                                                                      loss.item(), acc))
        loss_all.append(loss.item())
    if args.save_flag:
        save_res(acc_all, num, args, 'gs-acc-few')
        # save_res(loss_all, num, args, 'gs-loss_256')
        # save_res(f_score_all, num, args, 'gs-f1-score_256')
if __name__ == '__main__':
    arg = get_args.get_my_args()
    seed_torch(arg.seed)
    # datasets=['cora','citeseer','pubmed','CoraFull','AmazonCoBuyComputer','AmazonCoBuyPhoto','ogbn-arxiv','reddit']
    # datasets = ['cora', 'pubmed', 'CoraFull', 'ogbn-arxiv', 'reddit']
    for data in arg.datasets_list[:-1]:
        arg.dataset = data
        arg.seed = random.randint(3, 200)
        seed_torch(arg.seed)
        print(arg)
        print(f'half:{arg.half},mask_rate:{arg.mask_rate}')
        print(arg.dataset)
        for i in range(10):
            main(arg,i)
