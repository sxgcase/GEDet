import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import json
import os
from networkx.readwrite import json_graph as jg



def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index




def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.toarray()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict




def dumpJSON(destDirect, datasetName, graph, idMap, classMap, features):
    print("Dumping into JSON files...")
    # Turn graph into data
    dataG = jg.node_link_data(graph)
    # print(graph.number_of_edges())
    # Make names
    json_G_name = destDirect + '/' + datasetName + '-G.json'
    json_ID_name = destDirect + '/' + datasetName + '-id_map.json'
    json_C_name = destDirect + '/' + datasetName + '-class_map.json'
    npy_F_name = destDirect + '/' + datasetName + '-feats'

    # Dump graph into json file
    with open(json_G_name, 'w') as outputFile:
        json.dump(dataG, outputFile)

    # Dump idMap into json file
    with open(json_ID_name, 'w') as outputFile:
        json.dump(idMap, outputFile)

    # Dump classMap into json file
    with open(json_C_name, 'w') as outputFile:
        json.dump(classMap, outputFile)

    # Save features as .npy file
    print("Saving features as numpy file...")
    np.save(npy_F_name, features)

    print("all part finished")

