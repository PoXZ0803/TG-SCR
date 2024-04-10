from __future__ import print_function

import os, tqdm, torch, sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils.datasets import DFGDataset, collate_fn, myDataset
import dgl
from dgl.nn.pytorch import GraphConv, SAGEConv
import utils.metrics as metrics
import matplotlib.pyplot as plt
from utils.plotlib import plot_one_loss, plot_two_loss
import time
import math
import argparse

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--city', default='beijing')
    parse.add_argument('--cuda_num', default=0)
    parse.add_argument('--model', default='gcn')
    args = parse.parse_args()
    return args

args = parse_args()

class config:
    cuda_num = 0
    cityname = 'beijing'
    epoch_num = 500
    batch_size = 64
    interval = 10
    lr = 1e-3
    input_dims = 24
    hidden_dims = 256
    output_dims = 24

g = dgl.load_graphs(f'./dataset/{args.city}_graph.bin')[0][0]
# a = torch.tensor(g.ndata['x_count'], dtype=torch.int32)
# g.ndata['x_count'] = a
# a = torch.tensor(g.ndata['y'], dtype=torch.int32)
# g.ndata['y'] = a
# print(g.ndata['x_count'].dtype)
# print(g.ndata['y'].dtype)
g_train = dgl.node_subgraph(g, range(0, int(g.num_nodes() * 0.6)))
g_val = dgl.node_subgraph(g, range(int(g.num_nodes() * 0.6), int(g.num_nodes() * 0.8)))
g_test = dgl.node_subgraph(g, range(int(g.num_nodes() * 0.8), g.num_nodes()))
print(g_train)
print(g_val)
print(g_test)
time.sleep(100)
g_train = dgl.add_self_loop(g_train)
g_val = dgl.add_self_loop(g_val)
g_test = dgl.add_self_loop(g_test)
train_nids = np.load(f'./dataset/gnns/{args.city}_train_indice.npz')['arr_0']
val_nids = np.load(f'./dataset/gnns/{args.city}_val_indice.npz')['arr_0']
test_nids = np.load(f'./dataset/gnns/{args.city}_test_indice.npz')['arr_0']


sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
train_dataloader = dgl.dataloading.DataLoader(g_train, train_nids, sampler, batch_size=64, shuffle=True, drop_last=False)
val_dataloader = dgl.dataloading.DataLoader(g_val, val_nids, sampler, batch_size=64, shuffle=True, drop_last=False)
test_dataloader = dgl.dataloading.DataLoader(g_test, test_nids, sampler, batch_size=64, shuffle=True, drop_last=False)
input_nodes, output_nodes, blocks = next(iter(train_dataloader))


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_dim1, hidden_dim2, out_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_dim1)
        self.conv2 = GraphConv(hidden_dim1, hidden_dim2)
        self.fc = torch.nn.Linear(hidden_dim2, out_feats)
        self.activate = torch.nn.Sigmoid()

    def forward(self, blocks, input):
        h = self.conv1(blocks[0], input)
        h = self.activate(h)
        h = self.conv2(blocks[1], h)
        h = self.activate(h)
        h = self.fc(h)
        return h

model = GCN(config.input_dims, 48, 64, config.output_dims)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_dim1, hidden_dim2, out_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_dim1, aggregator_type='mean')
        self.conv2 = SAGEConv(hidden_dim1, hidden_dim2, aggregator_type='mean')
        self.fc = torch.nn.Linear(hidden_dim2, out_feats)
        self.activate = torch.nn.Sigmoid()

    def forward(self, blocks, input):
        h = self.conv1(blocks[0], input)
        h = self.activate(h)
        h = self.conv2(blocks[1], h)
        h = self.activate(h)
        h = self.fc(h)
        return h

# model = GraphSAGE(config.input_dims, 48, 64, config.output_dims)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f'params number: {pytorch_total_params}')

model.criterion = nn.MSELoss()
mseloss = nn.MSELoss()

val_epoch_list = []
tr_mse = []
val_mse = []

# '''
for epoch in tqdm.tqdm(range(config.epoch_num), ncols=70, ascii=True):
    model.train()
    for input_nodes, output_nodes, blocks in train_dataloader:
        # print(input_nodes)
        # print(output_nodes)
        # print(blocks)
        # print(blocks[0].srcdata['x_count'].shape)
        # print(blocks[-1].dstdata['y'].shape)
        # time.sleep(1)
        # blocks = [b.to(torch.device('cuda')) for b in blocks]
        input_features = blocks[0].srcdata['x_count']
        output_labels = blocks[-1].dstdata['y']
        output_predictions = model(blocks, input_features)
        loss = model.criterion(output_predictions, output_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % config.interval == config.interval-1:
        model.eval()
        val_epoch_list.append(epoch)
        
        train_mse_one_epoch = []
        for val_input_nodes, val_output_nodes, val_blocks in train_dataloader:
            val_input_features = val_blocks[0].srcdata['x_count']
            val_output_labels = val_blocks[-1].dstdata['y']
            output_predictions = model(val_blocks, val_input_features)
            one_mse = mseloss(output_predictions, val_output_labels)
            train_mse_one_epoch.append(one_mse.item())
        MSE = np.mean(train_mse_one_epoch)
        tr_mse.append(MSE)
        print(f'training  EPOCH: {epoch}, MSE: {MSE}')
            
        val_mse_one_epoch = []
        for val_input_nodes, val_output_nodes, val_blocks in val_dataloader:
            val_input_features = val_blocks[0].srcdata['x_count']
            val_output_labels = val_blocks[-1].dstdata['y']
            output_predictions = model(val_blocks, val_input_features)
            one_mse = mseloss(output_predictions, val_output_labels)
            val_mse_one_epoch.append(one_mse.item())
        MSE = np.mean(val_mse_one_epoch)
        val_mse.append(MSE)
        print(f'training  EPOCH: {epoch}, MSE: {MSE}')
    
np.savez(f'./results/{args.city}/val_epoch_{args.model}.npz', val_epoch_list)
np.savez(f'./results/{args.city}/train_mse_{args.model}.npz', tr_mse)
np.savez(f'./results/{args.city}/val_mse_{args.model}.npz',val_mse)
torch.save(model, f'./results/{args.city}/{args.city}_best_{args.model}.pt')
# '''

# model test
tr_mse = np.load(f'./results/{args.city}/train_mse_{args.model}.npz')['arr_0']
val_epoch_list = np.load(f'./results/{args.city}/val_epoch_{args.model}.npz')['arr_0']
val_mse = np.load(f'./results/{args.city}/val_mse_{args.model}.npz')['arr_0']
best_net = torch.load(f'./results/{args.city}/{args.city}_best_{args.model}.pt')

test_pred_total = []
test_real_total = []
for test_input_nodes, test_output_nodes, test_blocks in test_dataloader:
    test_input_features = test_blocks[0].srcdata['x_count']
    test_output_labels = test_blocks[-1].dstdata['y']
    output_predictions = model(test_blocks, test_input_features)
    test_pred_total.append(output_predictions.detach().cpu().numpy())
    test_real_total.append(test_output_labels.detach().cpu().numpy())
test_pred_total = np.concatenate(test_pred_total, axis=0)
test_real_total = np.concatenate(test_real_total, axis=0)
    
test_rmse = metrics.get_RMSE(test_pred_total, test_real_total)
test_mae = metrics.get_MAE(test_pred_total, test_real_total)
test_mape = metrics.get_MAPE(test_pred_total, test_real_total)

print(round(test_rmse, 4), round(test_mae, 4), round(test_mape, 4))
with open(f"./results/{args.city}/test_{args.model}.txt", mode='a', encoding='utf-8') as f:
    f.writelines(str(round(test_rmse, 4)) + ' ' + str(round(test_mae, 4)) + ' ' + str(round(test_mape, 4)) + '\n')
plot_one_loss(val_epoch_list, tr_mse, 'mse_loss', f'{args.model}', args.city)
print('finish!')
