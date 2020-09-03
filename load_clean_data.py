import json
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import scipy.sparse as sp
import os
import random
from matplotlib import pyplot as plt

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)



def z_score(x, mean, std):
    '''
    Z-score normalization function: $z = (X - \mu) / \sigma $,
    where z is the z-score, X is the value of the element,
    $\mu$ is the population mean, and $\sigma$ is the standard deviation.
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score normalized array.
    '''
    return (x - mean) / std


def load_clean_data(path,prefix, normalize=True):
    embedding_path = 'processed/'

    G_data = json.load(open(path + prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)

    #directly load the saved adj from graph augmentation
    adj_updated = sp.load_npz(embedding_path+'adj_matrix.npz')



    print("The number of nodes")
    nodes_num = G.number_of_nodes()
    print(nodes_num)

    if isinstance(G.nodes()[0], int):
        conversion = lambda n: int(n)
    else:
        conversion = lambda n: n

    # load the original feats
    feats_original = sp.load_npz(embedding_path + 'clean_feats.npz')
    # change the csr_matrix to ndarray
    feats_original = feats_original.toarray()
    # load the original embeddings
    embeds_clean = np.load(embedding_path + 'clean_embeddings.npy')
    feats = np.concatenate((feats_original, embeds_clean), axis=1)

    #perform z_score if necessary
    #x_stats = {'mean': np.mean(feats), 'std': np.std(feats)}
    #feats = z_score(feats, x_stats['mean'], x_stats['std'])




    id_map = json.load(open(path+prefix + "-id_map.json"))
    id_map = {conversion(k): int(v) for k, v in id_map.items()}

    walks = []
    class_map = json.load(open(path + prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n: n
    else:
        lab_conversion = lambda n: int(n)

    class_map = {conversion(k): lab_conversion(v) for k, v in class_map.items()}


    # generate y_train, y_val, y_test ndarray
    y_train = np.array([0, 0])
    y_val = np.array([0, 0])
    y_test = np.array([0, 0])
    # use only 10% nodes as training
    semi_threshold = int(round(nodes_num * 0.6 * 0.1))
    idx_train = range(semi_threshold)
    idx_val = []
    idx_test = []
    for node in G.nodes():
        if G.node[node]['test'] == False and G.node[node]['val'] == False and node in idx_train:
            print("Train,currrent n is %d" % node)
            train_label = G.node[node]['label']
            train_label = np.array(train_label)
            y_train = np.vstack((y_train, train_label))
            y_val = np.vstack((y_val, [0, 0]))
            y_test = np.vstack((y_test, [0, 0]))
        elif G.node[node]['test'] == False and G.node[node]['val'] == False and node not in idx_train:
            print("current n is %d" %node)
            y_train = np.vstack((y_train, [0, 0]))
            y_val = np.vstack((y_val, [0, 0]))
            y_test = np.vstack((y_test, [0, 0]))
        elif G.node[node]['test'] == False and G.node[node]['val'] == True:
            print("Validation, current n is %d" % node)
            validation_label = G.node[node]['label']
            validation_label = np.array(validation_label)
            y_val = np.vstack((y_val, validation_label))
            y_train = np.vstack((y_train, [0, 0]))
            y_test = np.vstack((y_test, [0, 0]))
            idx_val.append(node)
        elif G.node[node]['test'] == True and G.node[node]['val'] == False:
            print("Test, current n is %d" % node)
            test_label = G.node[node]['label']
            test_label = np.array(test_label)
            y_test = np.vstack((y_test, test_label))
            y_train = np.vstack((y_train, [0, 0]))
            y_val = np.vstack((y_val, [0, 0]))
            idx_test.append(node)

    print("training label shape is")
    print(y_train.shape)
    y_train = np.delete(y_train, 0, axis=0)
    y_val = np.delete(y_val, 0, axis=0)
    y_test = np.delete(y_test, 0, axis=0)

    # generate train_mask, val_mask and test_mask
    train_mask = sample_mask(idx_train, len(G.node))
    val_mask = sample_mask(idx_val, len(G.node))
    test_mask = sample_mask(idx_test, len(G.node))

    # check how many train_mask is true:
    train_true_num = np.count_nonzero(train_mask)
    # Similarly for val_mask, test_mask
    val_true_num = np.count_nonzero(val_mask)
    test_true_num = np.count_nonzero(test_mask)

    # print the anormaly ground truth number
    anormaly_count_gt = 0
    anormaly_count_vl = 0
    anormaly_count_tn = 0
    for node in G.nodes():
        if G.node[node]['test'] == True:
            if G.node[node]['label'] == [0, 1]:
                anormaly_count_gt += 1
        if G.node[node]['val'] == True:
            if G.node[node]['label'] == [0, 1]:
                anormaly_count_vl += 1
        if G.node[node]['val'] != True and G.node[node]['test'] != True and node in idx_train:
            if G.node[node]['label'] == [0, 1]:
                anormaly_count_tn += 1
    print("anormaly in test data is %d" % (anormaly_count_gt))
    print("anormaly in validation data is %d" % (anormaly_count_vl))
    print("anormaly in training data is %d" % (anormaly_count_tn))

    node_degrees = list(G.degree().values())
    print("the maximum degree of the graph is %d" % max(node_degrees))

    ## Remove all nodes that do not have val/test annotations
    broken_count = 0
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))


    feats = sp.csr_matrix(feats)

    return adj_updated, feats, y_train, y_val, y_test, train_mask, val_mask, test_mask
