import pynndescent
import numpy as np
import time
import torch
import dgl
import argparse
def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--n_neighbors', default=10)
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

time1 = time.time()
city = 'shenzhen'
poi_data = np.load(f'./dataset/{city}_selected_poi.npz')['arr_0']
traffic_data = np.load(f'./dataset/{city}_selected_traffic.npz')['arr_0']
index = pynndescent.NNDescent(data=poi_data, n_neighbors=int(args.n_neighbors))
graph_tuple = index.neighbor_graph
node_array = graph_tuple[0]
edge_weight_array = graph_tuple[1]
print(node_array.shape)

start_list = []
end_list = []
weight_list = []
for i in range(np.size(node_array, 0)):
    start_list = start_list + [i] * (np.size(node_array, 1) - 1)
    end_list = end_list + node_array[i, 1:].tolist()
    weight_list = weight_list + edge_weight_array[i, 1:].tolist()
    # print(start_list)
    # print(end_list)
    # time.sleep(30)
g = dgl.graph(data=(start_list, end_list))
g.ndata['x_count'] = torch.FloatTensor(poi_data)
g.ndata['y'] = torch.FloatTensor(traffic_data)
weight = np.array(weight_list).reshape(-1, 1)
print(weight.shape)
g.edata['weight'] = torch.FloatTensor(weight)
g = dgl.to_bidirected(g, copy_ndata=True)
print(g)
dgl.save_graphs(f'./dataset/knn_graphs/{city}_knn_graph.bin', g)
