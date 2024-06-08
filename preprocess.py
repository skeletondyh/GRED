import os
import numpy as np
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.datasets import ZINC
from torch_geometric.transforms import ToDense
from functools import partial
from multiprocessing import Pool
from scipy.sparse.csgraph import floyd_warshall

splits = ['train', 'val', 'test']
metadata = {
    'MNIST': {'max_num_nodes': 75},
    'CIFAR10': {'max_num_nodes': 150},
    'PATTERN': {'max_num_nodes': 188},
    'CLUSTER': {'max_num_nodes': 190},
    'ZINC': {'max_num_nodes': 38}
}

func_sp = partial(floyd_warshall, directed=False, unweighted=True)

def process(name):
    for split in splits:
        dataset = GNNBenchmarkDataset(root='./data', name=name, split=split,
                                      transform=ToDense(num_nodes=metadata[name]['max_num_nodes']))
        keys = dataset[0].keys
        dataset_as_dict = {key: [] for key in keys}
        for g in dataset:
            for key in keys:
                dataset_as_dict[key].append(g[key].numpy())
        
        adjs = dataset_as_dict.pop('adj')
        with Pool(25) as p: #! adjust according to your machine
            dist = p.map(func_sp, adjs)
        dist = np.stack(dist)
        dist = np.where(np.isfinite(dist), dist, -1).astype(np.int32)
        dist_mask = np.stack([(dist == k) for k in range(dist.max() + 1)], axis=1)
        
        if name in ['MNIST', 'CIFAR10']:
            np.savez(f'./data/{name}/{split}.npz',
                     x = np.concatenate([
                         np.stack(dataset_as_dict['x']),
                         np.stack(dataset_as_dict['pos'])
                     ], axis=-1),
                     y = np.concatenate(dataset_as_dict['y']).astype(np.int32),
                     node_mask = np.stack(dataset_as_dict['mask']))
        elif name in ['PATTERN', 'CLUSTER']:
            np.savez(f'./data/{name}/{split}.npz',
                     x = np.stack(dataset_as_dict['x']),
                     y = np.stack(dataset_as_dict['y']).astype(np.int32),
                     node_mask = np.stack(dataset_as_dict['mask']))
        np.save(f'./data/{name}/{split}_dist_mask', dist_mask)

def process_zinc():
    if not os.path.exists('./data/ZINC'):
        os.mkdir('./data/ZINC')
    for split in splits:
        dataset = ZINC(root='./data/ZINC', subset=True, split=split,
                       transform=ToDense(num_nodes=metadata['ZINC']['max_num_nodes']))
        keys = dataset[0].keys
        dataset_as_dict = {key: [] for key in keys}
        for g in dataset:
            for key in keys:
                dataset_as_dict[key].append(g[key].numpy())
        
        adjs = dataset_as_dict.pop('adj')
        with Pool(25) as p:
            dist = p.map(func_sp, adjs)
        dist = np.stack(dist)
        dist = np.where(np.isfinite(dist), dist, -1).astype(np.int32)
        dist_mask = np.stack([(dist == k) for k in range(dist.max() + 1)], axis=1)

        np.savez(f'./data/ZINC/subset/{split}.npz',
                 x = np.stack(dataset_as_dict['x']).squeeze().astype(np.int32),
                 y = np.concatenate(dataset_as_dict['y']),
                 node_mask = np.stack(dataset_as_dict['mask']))
        np.save(f'./data/ZINC/subset/{split}_dist_mask', dist_mask)
        np.save(f'./data/ZINC/subset/{split}_edge_attr', np.stack(adjs).astype(np.int32))


if __name__ == '__main__':
    if not os.path.exists('./data'):
        os.mkdir('./data')
    for name in ['MNIST', 'CIFAR10', 'PATTERN', 'CLUSTER']:
        print(f'Processing {name}...')
        process(name)
    print(f'Processing ZINC...')    
    process_zinc()