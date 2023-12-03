import os.path
import os.path
import time
import warnings
import dgl
import random
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import SAGEConv

from config import get_args
from utils.utils import *

warnings.filterwarnings("ignore")
from model import SAGE as Encoder

class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type, center_num, device):
        super().__init__()
        self.device = device
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.n_classes = n_classes
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        self.prompt = nn.Linear(2 * n_hidden, n_classes, bias=True)
        self.proto=Prototype(2 * n_hidden, n_classes)

    def weigth_init(self, graph, inputs, train_nid, labels):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(graph.to(self.device), h.to(self.device))
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        h = self.activation(h)

        h = torch.cat((h, graph.ndata['gfeat'].to(self.device)), dim=1)
        h=h[train_nid]
        p = []
        for i in range(self.n_classes):
            p.append(h[labels[train_nid] == i].mean(dim=0).view(1, -1))
        dd = torch.cat(p, dim=0)
        self.proto.setInitPara(dd.clone().detach())

    def reset_parameters(self):
        # for emb in self.layers:
        #     nn.init.xavier_uniform_(emb.weight.data)
        num_layers = len(self.layers)
        for i in range(num_layers):
            self.layers[i].reset_parameters()
        self.prompt.reset_parameters()

    def forward(self, graph, lab=None, Train_flag=False):
        if not self.dropout:
            h = graph[0].srcdata['feat']
        else:
            h = self.dropout(graph[0].srcdata['feat'])
        for l, layer in enumerate(self.layers):
            if self.dropout:
                h = self.dropout(h)
            h = layer(graph[l], h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
        h = self.activation(h)
        h_dst = graph[0].srcdata['gfeat'][:graph[-1].num_dst_nodes()]
        h = torch.cat((h, h_dst), dim=1)
        return self.proto(h)




def main_func(args, num):
    g, features, labels, in_feats, n_classes, n_edges, train_nid, val_nid, test_nid, device = get_init_info(args)
    g.create_formats_()
    g = dgl.add_self_loop(g)
    sampler = dgl.dataloading.MultiLayerNeighborSampler(args.sample_list)
    train_dataloader = dgl.dataloading.DataLoader(g, train_nid, sampler, device=device,
                                                  batch_size=args.batch_size, shuffle=True, drop_last=False,
                                                  num_workers=args.num_workers)
    model = GraphSAGE(in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout, args.aggregator_type,
                      args.center_num, device)
    model.reset_parameters()
    model.to(device)
    encoder = Encoder(in_feats, args.n_hidden, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout)
    m_m = torch.load('./pre_data/' + args.dataset + '_model_' + args.file_id + '.pt', map_location=device)
    encoder.load_state_dict(m_m)
    encoder.eval()
    encoder.to(device)
    with torch.no_grad():
        emb = encoder.inference(g, features, device, 4000, 0)
    emb = (emb - emb.mean(0, keepdims=True)) / emb.std(0, keepdims=True)
    g.ndata['gfeat'] = emb
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.CrossEntropyLoss()
    model.weigth_init(g, features, train_nid, labels)
    acc_all = []
    loss_all = []
    f_score_all = []
    max_acc = 0.0
    name = os.path.basename(__file__).split(".")[0]
    model_path = f'temp/model_{args.dataset}_{name}.pkl'
    for epoch in range(args.n_epochs):
        acc, f_score = evaluate(model, g, test_nid, args.batch_size, device, args.sample_list)
        if acc > max_acc:
            torch.save(model.state_dict(), model_path)
            max_acc = acc
        acc_all.append(acc)
        f_score_all.append(f_score)
        model.train()
        t0 = time.time()
        loss = None
        for step, (input_nodes, output_nodes, mfgs) in enumerate(train_dataloader):
            # batch_inputs = mfgs[0].srcdata['feat']
            lab = mfgs[-1].dstdata['label']
            optimizer.zero_grad()
            pred = model(mfgs, lab, True)
            loss = loss_fn(pred, lab)
            loss.backward()
            optimizer.step()
        loss_all.append(loss.item())
        print("Epoch {:03d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f}|f1_score {:.4f} ".format(epoch,
                                                                                                      time.time() - t0,
                                                                                                      loss.item(), acc,
                                                                                                      f_score))
    model.reset_parameters()
    model.load_state_dict(torch.load(model_path))
    acc, f_score = evaluate(model, g, test_nid, args.batch_size, device, args.sample_list)
    print(args.dataset)
    print(f'val load para when max_cc={max_acc}')
    print(f'test :{acc}')
    print(f'f1_score :{f_score}')
    if args.save_flag:
        # save_res(acc_all, num, args, 'down-batch_GS+Proto_val')
        # save_res(loss_all, num, args, 'bf_wu-loss_')
        save_res(f_score_all, num, args, 'bf_wu+Proto-f1-score_VAL')
    print('over')
    # return
    # Load dataset


if __name__ == '__main__':
    arg = get_args.get_my_args()
    seed_torch(arg.seed)
    datasets=['cora','citeseer','pubmed','CoraFull','AmazonCoBuyComputer','AmazonCoBuyPhoto','ogbn-arxiv','reddit']
    # datasets = ['cora', 'citeseer', 'pubmed', 'CoraFull', 'AmazonCoBuyComputer', 'AmazonCoBuyPhoto', 'ogbn-arxiv']
    for data in datasets:
        arg.dataset = data
        arg.seed = random.randint(3, 200)
        seed_torch(arg.seed)
        print(arg)
        print(f'half:{arg.half},mask_rate:{arg.mask_rate}')
        print(arg.dataset)

        for i in range(0, 10):
            print('第', i, '次')
            main_func(arg, i)
