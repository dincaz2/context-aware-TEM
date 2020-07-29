from time import time

import numpy as np
from scipy.sparse import csc_matrix


def node_col_mapping(train_node_pred, valid_node_pred, test_node_pred):
    min_mapping = list()
    base_mapping = list()
    raw_num, col_num = train_node_pred.shape
    
    for i in range(col_num):
        min_node = min(np.min(train_node_pred[:,i]), np.min(valid_node_pred[:,i]), np.min(test_node_pred[:,i]))
        max_node = max(np.max(train_node_pred[:,i]), np.max(valid_node_pred[:,i]), np.max(test_node_pred[:,i]))
        if i == 0:
            base_node = max_node - min_node + 1
        else:
            base_node = max_node - min_node + base_mapping[i-1] + 1
        
        min_mapping.append(min_node)
        base_mapping.append(base_node)
    return min_mapping, base_mapping


def node_encoding(raw_node_pred, min_mapping, base_mapping):
    multi_node_pred = np.zeros(shape=raw_node_pred.shape)
    raw_num, col_num = raw_node_pred.shape
    
    for i in range(col_num):
        min_node = min_mapping[i]
        if i == 0:
            base_code = 0
        else:
            base_code = base_mapping[i-1]
            
        multi_node_pred[:,i] = raw_node_pred[:,i] - min_node + base_code
    return multi_node_pred


def node_onehot_csc(multi_node_pred):
    t1 = time()
    csc_rows = list()
    csc_cols = list()
    
    for pairs_index, fields in enumerate(multi_node_pred):
        for field_index in fields:
            csc_rows.append(pairs_index)
            csc_cols.append(field_index)
    csc_data = np.ones((len(csc_rows),), dtype=np.int)
    num_rows = max(csc_rows) + 1
    num_cols = max(csc_cols) + 1
    
    onehot_node_pred = csc_matrix((csc_data, (csc_rows, csc_cols)), shape=(num_rows, num_cols))
    print('convert to one hot csc matrix@%.4f; shape=(%d,%d);' % (time() - t1, num_rows, num_cols, ))
    return onehot_node_pred