import torch as th
import torch.nn as nn
import torch.functional as F
import dgl
import dgl.nn as dglnn
import sklearn.linear_model as lm
import sklearn.metrics as skm
import tqdm
import torch, gc
from dgl.nn.pytorch.conv import SAGEConv,GATConv

class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, classes, n_layers, activation, dropout, aggregator_type='gcn'):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.classes = classes
        self.layers = nn.ModuleList()
        if n_layers > 1:
            self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
            for i in range(1, n_layers - 1):
                self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
            self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type))
        else:
            self.layers.append(SAGEConv(in_feats, n_classes, aggregator_type))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def get_e(self):
        return self.embedding_x

    def get_pre(self):
        return self.pre

    def forward(self, blocks, x):
        h = self.dropout(x)
        # print(blocks)
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        self.embedding_x = h
        # h= self.prompt(h)
        return h

    def inference(self, g, x, device, batch_size, num_workers):
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g, th.arange(g.num_nodes()).to(g.device),
                sampler,
                device=device if num_workers == 0 else None,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers)
            for input_nodes, output_nodes, blocks in dataloader:  # tqdm.tqdm(dataloader):
                block = blocks[0]
                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)
                y[output_nodes] = h.cpu()
                # gc.collect()
                # torch.cuda.empty_cache()

            x = y
        return y


def compute_acc_unsupervised(emb, labels, train_nids, val_nids, test_nids):
    """
    Compute the accuracy of prediction given the labels.
    """
    # 设置一个LR模型来检测acc
    emb = emb.cpu().numpy()
    labels = labels.cpu().numpy()
    train_nids = train_nids.cpu().numpy()
    train_labels = labels[train_nids]
    val_nids = val_nids.cpu().numpy()
    val_labels = labels[val_nids]
    test_nids = test_nids.cpu().numpy()
    test_labels = labels[test_nids]

    emb = (emb - emb.mean(0, keepdims=True)) / emb.std(0, keepdims=True)

    lr = lm.LogisticRegression(multi_class='multinomial', max_iter=10000)
    lr.fit(emb[train_nids], train_labels)

    pred = lr.predict(emb)
    f1_micro_eval = skm.f1_score(val_labels, pred[val_nids], average='micro')
    f1_micro_test = skm.f1_score(test_labels, pred[test_nids], average='micro')
    return f1_micro_eval, f1_micro_test
