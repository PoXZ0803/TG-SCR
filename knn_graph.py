from __future__ import print_function

import os, tqdm, torch, sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import dgl
import utils.metrics as metrics
from utils.plotlib import plot_one_loss, plot_two_loss
import time
import math
import argparse
from dgl.data.utils import generate_mask_tensor
from utils.models import GCN, GraphSAGE, GAT, GIN

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--city', default='shenzhen')
    parse.add_argument('--cuda_num', default=0)
    parse.add_argument('--model', default='gcn')
    parse.add_argument('--k', default=20)
    args = parse.parse_args()
    return args

args = parse_args()

class config:
    epoch_num = 2000
    batch_size = 64
    interval = 50
    lr = 1e-2
    input_dims = 24
    hidden_dims = 256
    output_dims = 24

g = dgl.load_graphs(f'./dataset/knn_graphs/{args.city}_knn_graph.bin')[0][0]
# train_nids = range(0, int(g.num_nodes() * 0.6))
# val_nids = range(int(g.num_nodes() * 0.6), int(g.num_nodes() * 0.8))
# test_nids = range(int(g.num_nodes() * 0.8), g.num_nodes())

# train_nids = np.load(f'./dataset/gnns/{args.city}_train_indice_knn.npz')['arr_0']
# val_nids = np.load(f'./dataset/gnns/{args.city}_val_indice_knn.npz')['arr_0']
# test_nids = np.load(f'./dataset/gnns/{args.city}_test_indice_knn.npz')['arr_0']
input_features = g.ndata['x_count']
real_traffic = g.ndata['y']

def get_mask(g):
    train_mask = [0] * g.num_nodes()
    val_mask = [0] * g.num_nodes()
    test_mask = [0] * g.num_nodes()
    for index in train_nids:
        train_mask[index] = 1
    for index in val_nids:
        val_mask[index] = 1
    for index in test_nids:
        test_mask[index] = 1
    return generate_mask_tensor(np.array(train_mask)), generate_mask_tensor(np.array(val_mask)), generate_mask_tensor(np.array(test_mask))
        

train_mask, val_mask, test_mask = get_mask(g)
input_features = g.ndata['x_count'].float()
real_traffic = g.ndata['y']
train_real_traffic = real_traffic[train_mask]
val_real_traffic = real_traffic[val_mask]
test_real_traffic = real_traffic[test_mask]


if args.model == 'gcn':
    model = GCN(config.input_dims, 48, config.output_dims)
    model.optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
elif args.model == 'graphsage':
    model = GraphSAGE(config.input_dims, 48, config.output_dims)
    model.optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
elif args.model == 'gat':
    model = GAT(g, in_dim = config.input_dims, hidden_dim=8, out_dim=config.output_dims, num_heads=8)
    model.optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
elif args.model == 'gin':
    model = GIN(config.input_dims, 48, config.output_dims)
    model.optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f'params number: {pytorch_total_params}')

model.criterion = nn.MSELoss()
mseloss = nn.MSELoss()

val_epoch_list = []
tr_mse = []
val_mse = []

for epoch in tqdm.tqdm(range(config.epoch_num), ncols=70, ascii=True):
    model.train()
    if args.model == 'gat':
        output_predictions = model(input_features)
    else:
        output_predictions = model(g, input_features)
    train_output_predictions = output_predictions[train_mask]
    loss = model.criterion(train_output_predictions, train_real_traffic)
    model.optimizer.zero_grad()
    loss.backward()
    model.optimizer.step()
    if epoch % config.interval == config.interval-1:
        model.eval()
        val_epoch_list.append(epoch)
        
        if args.model == 'gat':
            output_predictions = model(input_features)
        else:
            output_predictions = model(g, input_features)
        train_output_predictions = output_predictions[train_mask]
        print(train_output_predictions)
        print(train_real_traffic)
        MSE = model.criterion(train_output_predictions, train_real_traffic)
        MSE = MSE.detach().numpy()
        tr_mse.append(MSE)
        print(f'training  EPOCH: {epoch}, MSE: {MSE}')
            
        if args.model == 'gat':
            output_predictions = model(input_features)
        else:
            output_predictions = model(g, input_features)
        val_output_predictions = output_predictions[val_mask]
        MSE = model.criterion(val_output_predictions, val_real_traffic)
        MSE = MSE.detach().numpy()
        val_mse.append(MSE)
        print(f'validation EPOCH: {epoch}, MSE: {MSE}')
    
np.savez(f'./results/{args.city}/val_epoch_{args.model}_knn.npz', val_epoch_list)
np.savez(f'./results/{args.city}/train_mse_{args.model}_knn.npz', tr_mse)
np.savez(f'./results/{args.city}/val_mse_{args.model}_knn.npz',val_mse)
torch.save(model, f'./results/{args.city}/{args.city}_best_{args.model}_knn.pt')

tr_mse = np.load(f'./results/{args.city}/train_mse_{args.model}_knn.npz')['arr_0']
val_epoch_list = np.load(f'./results/{args.city}/val_epoch_{args.model}_knn.npz')['arr_0']
val_mse = np.load(f'./results/{args.city}/val_mse_{args.model}_knn.npz')['arr_0']
best_net = torch.load(f'./results/{args.city}/{args.city}_best_{args.model}_knn.pt')

if args.model == 'gat':
    output_predictions = model(input_features)
else:
    output_predictions = model(g, input_features)
test_output_predictions = output_predictions[test_mask]
test_output_predictions = test_output_predictions.detach().numpy()
test_real_traffic = test_real_traffic.detach().numpy()
    
test_rmse = metrics.get_RMSE(test_output_predictions, test_real_traffic)
test_mae = metrics.get_MAE(test_output_predictions, test_real_traffic)
test_mape = metrics.get_MAPE(test_output_predictions, test_real_traffic)

print(round(test_rmse, 4), round(test_mae, 4), round(test_mape, 4))
with open(f"./results/{args.city}/test_{args.model}_knn_{args.k}.txt", mode='a', encoding='utf-8') as f:
    f.writelines(str(round(test_rmse, 4)) + ' ' + str(round(test_mae, 4)) + ' ' + str(round(test_mape, 4)) + '\n')
plot_one_loss(val_epoch_list, tr_mse, 'mse_loss', f'{args.model}_knn', args.city)
print('finish!')
