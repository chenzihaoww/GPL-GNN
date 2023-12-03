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
        self.n_hidden=n_hidden
        # ,feat_drop=dropout,activation=activation
        # ,feat_drop=dropout,activation=activation
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        self.prompt = nn.Linear(2 * n_hidden, n_classes, bias=True)

    def weigth_init(self, graph, inputs, labels, train_nid):
        pass

    def reset_parameters(self):
        # for emb in self.layers:
        #     nn.init.xavier_uniform_(emb.weight.data)
        num_layers = len(self.layers)
        for i in range(num_layers):
            self.layers[i].reset_parameters()
        self.prompt.reset_parameters()

    def forward(self, graph, batch_inputs=None, Train_flag=False):
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
        out = self.prompt(h)
        return out

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
    # model.reset_parameters()
    model.to(device)
    encoder = Encoder(in_feats, args.n_hidden, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout)
    m_m = torch.load('./pre_data/' + args.dataset + '_model_' + args.file_id + '.pt', map_location=device)
    encoder.load_state_dict(m_m)
    encoder.eval()
    encoder.to(device)

    with torch.no_grad():
        emb = encoder.inference(g, features, device, 4000, 0)
    # g_embeds = graph_power(emb, g)
    # embeds = emb + g_embeds
    # embeds = sk_prep.normalize(X=embeds.cpu().numpy(), norm="l2")
    # embeds = torch.FloatTensor(embeds).cuda()
    emb = (emb - emb.mean(0, keepdims=True)) / emb.std(0, keepdims=True)
    g.ndata['gfeat'] = emb
    # lr = sklearn.linear_model.LogisticRegression(multi_class='multinomial', max_iter=10000)
    # lr.fit(emb[train_nid].cpu(), labels[train_nid].cpu())
    # pred = lr.predict(emb)
    # f1_micro_eval = sklearn.metrics.f1_score(labels[test_nid].cpu(), pred[test_nid], average='micro')
    # print('f1_micro_eval:', f1_micro_eval)
    # 参数迁移
    # model.weigth_init(g, features, labels, train_nid)
    # model.load_parameters(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
            batch_inputs = mfgs[0].srcdata['feat']
            lab = mfgs[-1].dstdata['label']
            optimizer.zero_grad()
            pred = model(mfgs, batch_inputs, True)
            loss = F.cross_entropy(pred, lab)
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
        save_res(acc_all, num, args, 'bf_wu-acc-few')
        # save_res(loss_all, num, args, 'bf_wu-loss_')
        # save_res(f_score_all, num, args, 'bf_wu-f1-score_VAL')
    print('over')
    # return


if __name__ == '__main__':
    arg = get_args.get_my_args()
    seed_torch(arg.seed)
    datasets=['cora','citeseer','pubmed','CoraFull','AmazonCoBuyComputer','AmazonCoBuyPhoto','ogbn-arxiv','reddit']
    # datasets = ['cora']
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
