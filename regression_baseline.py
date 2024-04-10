import os, torch, sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils.datasets import DFGDataset, collate_fn, myDataset
import dgl
import utils.metrics as metrics
import matplotlib.pyplot as plt
import time
import math
import argparse
from dgl.data.utils import generate_mask_tensor
from xgboost.sklearn import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--city', default='beijing')
    parse.add_argument('--cuda_num', default=0)
    parse.add_argument('--model', default='xgboost')
    args = parse.parse_args()
    return args

args = parse_args()

poi = np.load(f'./dataset/{args.city}_selected_poi.npz')['arr_0']     
traffic = np.load(f'./dataset/{args.city}_selected_traffic.npz')['arr_0']

g = dgl.load_graphs(f'./dataset/{args.city}_graph.bin')[0][0]
train_nids = np.load(f'./dataset/gnns/{args.city}_train_indice.npz')['arr_0']
val_nids = np.load(f'./dataset/gnns/{args.city}_val_indice.npz')['arr_0']
test_nids = np.load(f'./dataset/gnns/{args.city}_test_indice.npz')['arr_0']
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
real_traffic = g.ndata['y'].float()
train_real_traffic = real_traffic[train_mask]
val_real_traffic = real_traffic[val_mask]
test_real_traffic = real_traffic[test_mask]

train_poi = poi[0:int(poi.shape[0] * 0.6), :]
test_poi = poi[int(poi.shape[0] * 0.8):, :]
train_traffic = traffic[0:int(poi.shape[0] * 0.6), :]
test_traffic = traffic[int(poi.shape[0] * 0.8):, :]

model = MultiOutputRegressor(XGBRegressor(max_depth=6, learning_rate=0.05, n_estimators=100))
model = MultiOutputRegressor(SVR(kernel='rbf'))
model = MultiOutputRegressor(DecisionTreeRegressor(random_state=0))
model = MultiOutputRegressor(RandomForestRegressor(random_state=0))


model.fit(train_poi, train_traffic)
test_traffic_pred = model.predict(test_poi)

test_rmse = metrics.get_RMSE(test_traffic_pred, test_traffic)
test_mae = metrics.get_MAE(test_traffic_pred, test_traffic)
test_mape = metrics.get_MAPE(test_traffic_pred, test_traffic)
print(round(test_rmse, 4), round(test_mae, 4), round(test_mape, 4))