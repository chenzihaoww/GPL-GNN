import warnings
import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import SAGEConv
import time
from config import get_args
from tqdm.auto import trange
import os

from utils import utils

warnings.filterwarnings("ignore")


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type, center_num):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.n_classes = n_classes
        self.center_num = center_num
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden,aggregator_type))
        self.prompt = nn.Linear(n_hidden, self.center_num, bias=False)
        self.fea=None
        self.pp = nn.ModuleList()
        for i in range(self.center_num):
            self.pp.append(nn.Linear(2 * n_hidden, n_classes, bias=False))
    def forward(self, graph, output_nodes=None, Train_flag=False):
        if self.dropout == False:
            h = graph[0].srcdata['feat']
        else:
            h = self.dropout(graph[0].srcdata['feat'])
            # print(h.shape)
        for l, layer in enumerate(self.layers):
            h_dst = h[:graph[l].num_dst_nodes()]
            h = layer(graph[l], (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.activation(h)
                if self.dropout != False:
                    h = self.dropout(h)
        h = self.activation(h)
        h_dst = self.activation(h_dst)
        neighbor = h_dst
        h = torch.cat((h, neighbor), dim=1)
        if not Train_flag:
            self.fea=h
        out= self.pp[0](h)
        return out
def evaluate(model, graph, nid, batch_size, device, sample_list,n=0):
    args = utils.get_my_args()
    sampler = dgl.dataloading.MultiLayerNeighborSampler(sample_list)
    valid_dataloader = dgl.dataloading.NodeDataLoader(graph, nid, sampler, batch_size=batch_size, shuffle=False,
                                                      drop_last=False, num_workers=args.num_workers, device=device)
    model.eval()
    predictions = []
    labels = []
    fea=[]
    with torch.no_grad():
        for input_nodes, output_nodes, mfgs in valid_dataloader:
            # inputs = mfgs[0].srcdata['feat']
            labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
            predictions.append(model(mfgs).argmax(1).cpu().numpy())
            # predictions.append(model(mfgs).argmax(1).cpu().numpy())
            # fea=model.fea
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        accuracy = utils.accuracy_score(labels, predictions)
    return accuracy

def save_res(acc_all, num,args):
    data = pd.DataFrame(acc_all)
    file_name = ''
    if args.half and args.mask_rate == 0.3:
        file_name = 'few_shot30'
    elif args.half and args.mask_rate == 0.5:
        file_name = 'few_shot50'
    elif not args.half:
        file_name = 'full'
    print(file_name, ',', num)
    if not os.path.exists(f'./res/baseline/{file_name}/' + args.dataset):
        os.makedirs(f'./res/baseline/{file_name}/' + args.dataset + '/')
    data.to_csv(f'./res/baseline/{file_name}/' + args.dataset + f'/self_{num}.csv', index=False)
    return
def main(args,num):
    utils.seed_torch(args.seed)
    g, features, labels, in_feats, n_classes, n_edges, train_nid, val_nid, test_nid, device = utils.get_init_info(args)
    sampler = dgl.dataloading.MultiLayerNeighborSampler(args.sample_list)
    train_dataloader = dgl.dataloading.DataLoader(g, train_nid, sampler, device=device,
                                                  batch_size=args.batch_size, shuffle=True, drop_last=False,
                                                  num_workers=0)

    model = GraphSAGE(in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout, args.aggregator_type,args.center_num)
    # model.load_parameters(args)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    acc_all = []
    loss_all = []
    max_acc = 0.0
    fea=None
    labels=None
    for epoch in range(args.n_epochs):
        acc = evaluate(model, g, test_nid, args.batch_size, device, args.sample_list,epoch)
        if acc > max_acc:
            # torch.save(model.state_dict(), f'temp/model_{args.dataset}.pkl')
            max_acc = acc
        acc_all.append(acc)
        model.train()
        t0 = time.time()
        loss = None
        for step, (input_nodes, output_nodes, mfgs) in enumerate(train_dataloader):
            lab = mfgs[-1].dstdata['label']
            logits = model(mfgs, lab, True)
            loss = F.cross_entropy(logits, lab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # fea = model.fea
            # labels=lab
        print("Epoch {:03d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} ".format(epoch, time.time() - t0,loss.item(), acc))
    # model_state = torch.load(f'temp/model_{args.dataset}.pkl')
    # model.load_state_dict(model_state)
    # evaluate(model, g, test_nid, args.batch_size, device, args.sample_list, args.n_epochs)
    # print(args.dataset)
    print(f'val load para when max_cc={max_acc}')
    save_res(acc_all, num, args)

if __name__ == '__main__':

    result = []
    args = get_args.get_my_args()
    print(args)
    val_all = []
    for i in trange(1):
        print('第', i, '次')
        main(args,i)