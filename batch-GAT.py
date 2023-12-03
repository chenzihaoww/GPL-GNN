import random

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import get_args
from utils.utils import evaluate, seed_torch, get_init_info
from dgl.nn.pytorch.conv import GATConv
import time
import os
import pandas as pd


class GAT(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, heads, activation, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        # F.elu
        self.layers.append(GATConv(in_feats, n_hidden, heads[0],
                                   feat_drop=dropout,
                                   attn_drop=dropout, activation=activation))

        self.layers.append(
            GATConv(
                n_hidden * heads[0],
                n_hidden,
                heads[1],
                feat_drop=dropout,
                attn_drop=dropout,
                activation=activation,
            )
        )
        self.fc = nn.Linear(n_hidden, n_classes, bias=True)

    def forward(self, g,inputs=None):
        if inputs is None:
            inputs = g[0].srcdata['feat']
        h = inputs
        for l, layer in enumerate(self.layers):
            h = layer(g[l], h)
            if l != len(self.layers) - 1:
                h = h.flatten(1)
            else:  # last layer
                h = h.mean(1)
        out = self.fc(h)
        return out


def save_res(acc_all, num, args):
    data = pd.DataFrame(acc_all)
    file_name = ''
    if args.half and args.mask_rate == 0.3:
        file_name = 'few_shot30'
    elif args.half and args.mask_rate == 0.5:
        file_name = 'few_shot50'
    elif not args.half:
        file_name = 'full'
    print(file_name, ',', num)
    if not os.path.exists(f'./res/GAT/{file_name}/' + args.dataset):
        os.makedirs(f'./res/GAT/{file_name}/' + args.dataset + '/')
    data.to_csv(f'./res/GAT/{file_name}/' + args.dataset + f'/self_{num}.csv', index=False, header=None)


def evaluate_full_map(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train_full_map(g, features, labels, train_mask, val_mask, model, optimizer):
    loss_fcn = nn.CrossEntropyLoss()

    # training loop
    for epoch in range(200):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate_full_map(g, features, labels, val_mask, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )


def main_func(args, num=0):
    g, features, labels, in_feats, n_classes, n_edges, train_nid, val_nid, test_nid, device = get_init_info(args)
    g = dgl.add_self_loop(g)
    # create GAT model
    model = GAT(in_feats, args.n_hidden, n_classes, [8, 1], F.relu, args.dropout).to(device)
    model.to(device)
    sampler = dgl.dataloading.MultiLayerNeighborSampler(args.sample_list)
    train_dataloader = dgl.dataloading.DataLoader(g, train_nid, sampler, device=device,
                                                  batch_size=args.batch_size, shuffle=True, drop_last=False,
                                                  num_workers=args.num_workers)

    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # training loop
    model.train()
    acc_all = []
    max_acc = 0.0

    for epoch in range(args.n_epochs):
        acc,_ = evaluate(model, g, test_nid, args.batch_size, device, args.sample_list)
        if acc > max_acc:
            # torch.save(model.state_dict(), f'temp/model_{args.dataset}.pkl')
            max_acc = acc
        acc_all.append(acc)
        model.train()
        t0 = time.time()
        loss = None
        for step, (input_nodes, output_nodes, mfgs) in enumerate(train_dataloader):
            batch_inputs = mfgs[0].srcdata['feat']
            lab = mfgs[-1].dstdata['label']
            optimizer.zero_grad()
            pred = model(mfgs, batch_inputs)
            loss = F.cross_entropy(pred, lab)
            loss.backward()
            optimizer.step()
        print("Epoch {:03d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} ".format(epoch, time.time() - t0,
                                                                                      loss.item(), acc))
    print(f'val load para when max_cc={max_acc}')
    # print(f'test :{acc}')
    if args.save_flag:
        save_res(acc_all, num, args)
    print('over')


if __name__ == '__main__':
    arg = get_args.get_my_args()
    datasets = ['cora', 'citeseer', 'pubmed', 'CoraFull', 'AmazonCoBuyComputer', 'AmazonCoBuyPhoto', 'ogbn-arxiv',
                'reddit']
    for data in datasets:
        arg.dataset = data
        arg.seed = random.randint(3, 200)
        seed_torch(arg.seed)
        print(arg)
        print(arg.dataset)
        for i in range(0, 1):
            print('第', i, '次')
            main_func(arg, i)
