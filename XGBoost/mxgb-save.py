# pandas
# import pandas as pd
# from pandas import Series, DataFrame
from time import time
import numpy as np
# import scipy as sp
# import matplotlib.pyplot as plt
# import seaborn as sns
# import zipfile
import xgboost as xgb
from sklearn.metrics import log_loss
# from xgboost import XGBClassifier, plot_importance
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC, LinearSVC
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
import argparse
import json
import os.path

# from final_load_mxgb import FinalLoad
# from metrics import *
# from factory import *
import warnings

from MXGB.factory import ensureDir
from MXGB.final_load_mxgb import FinalLoad
from MXGB.metrics import eval_model_pro

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='Run XGBoost.')
    parser.add_argument('--out_dir', nargs='?', default='MXGB',
                        help='Output directory in (FM or NFM)')
    parser.add_argument('--city', nargs='?', default='London',
                        help='City Name in (London, New_York_City, Singapore)')
    parser.add_argument('--cate', nargs='?', default='Attractions',
                        help='Category in (Attractions, Restaurant)')
    parser.add_argument('--data_type', nargs='?', default='attr',
                        help='Data Type in (attr, id_attr, id+attr).')

    parser.add_argument('--batch_size', type=int, default=5120,
                        help='Batch size.')
    parser.add_argument('--num_boost_round', type=int, default=500)

    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--min_child_weight', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.)

    parser.add_argument('--booster', nargs='?', default='gbtree',
                        help='Specify which booster to use: (gbtree, gblinear or dart.)')
    parser.add_argument('--n_jobs', type=int, default=4)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--colsample_bytree', type=float, default=0.8)
    parser.add_argument('--reg_alpha', type=float, default=0.,
                        help='L1 regularization term on weights')
    parser.add_argument('--reg_lambda', type=float, default=1.,
                        help='L2 regularization term on weights')
    parser.add_argument('--scale_pos_weight', type=float, default=1,
                        help='Balancing of positive and negative weights.')
    parser.add_argument('--base_score', type=float, default=0.5,
                        help='The initial prediction score of all instances, global bias.')
    parser.add_argument('--merge', type=int, default=0)
    parser.add_argument('--save_flag', type=int, default=0,
                        help='Whether save the model & params')
    return parser.parse_args()


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


def save2txt(mapping, mapping_path):
    fout = open(mapping_path, 'w')
    for code in mapping:
        fout.write('%d\n' % code)


def save2json(params, params_path):
    with open(params_path, 'w') as fout:
        json.dump(params, fout)

def load2json(params_path):
    try:
        with open(params_path, 'r') as fin:
            data = json.load(fin)
        return data['valid_log_loss'], data['test_log_loss']
    except Exception:
        return 10, 10

def evaluate(y_true, y_pred, K, row_len):
    tmp_true = np.array(y_true.tolist())
    tmp_pred = np.array(y_pred.tolist())

    tmp_true = tmp_true.flatten()
    tmp_pred = tmp_pred.flatten()

    test_hit, test_ndcg = eval_model_pro(tmp_true, tmp_pred, K=K, row_len=row_len)
    return test_hit, test_ndcg

if __name__ == '__main__':
    args = parse_args()

    # fetching the data for training, validation, and testing.
    my_load = FinalLoad(args.city, args.cate, args.data_type)

    x_train, y_train = my_load._fetch_batch_data(flag='train')
    x_valid, y_valid = my_load._fetch_batch_data(flag='valid')
    x_test, y_test = my_load._fetch_batch_data(flag='test')

    train_dm = xgb.DMatrix(x_train, label=y_train)
    valid_dm = xgb.DMatrix(x_valid, label=y_valid)
    test_dm = xgb.DMatrix(x_test, label=y_test)

    evallist = [(train_dm, 'train'), (valid_dm, 'valid'), (test_dm, 'test')]

    ind_params = {'learning_rate': args.learning_rate,

                  'max_depth': args.max_depth,
                  'scale_pos_weight': args.scale_pos_weight,
                  'min_child_weight': args.min_child_weight,
                  
                  'subsample': args.subsample,
                  'colsample_bytree': args.colsample_bytree,
                  'gamma': args.gamma,

                  'reg_alpha': args.reg_alpha,
                  'reg_lambda': args.reg_lambda,

                  'objective': 'binary:logistic',
                  'seed': 2017,
                  'silent': True,
                  'nthread': 4,
                  'base_score': 0.5,
                  'eval_metric': 'logloss',}

    t1 = time()
    xgb_model = xgb.train(ind_params, train_dm, args.num_boost_round, evallist, early_stopping_rounds=10, verbose_eval=20)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # only output the probability prediction.
    y_train_pred = xgb_model.predict(train_dm)
    y_valid_pred = xgb_model.predict(valid_dm)
    y_test_pred = xgb_model.predict(test_dm)

    train_log_loss = log_loss(y_train, y_train_pred)
    valid_log_loss = log_loss(y_valid, y_valid_pred)
    test_log_loss = log_loss(y_test, y_test_pred)

    valid_hit, valid_ndcg = evaluate(y_valid, y_valid_pred, K=10, row_len=51)
    test_hit, test_ndcg = evaluate(y_test, y_test_pred, K=10, row_len=51)

    final_result_str = '[%.4f] train=[%.6f], valid=[%.6f, %.6f, %.6f], test=[%.6f, %.6f, %.6f]' % (time()-t1, train_log_loss, valid_log_loss, valid_hit, valid_ndcg, test_log_loss, test_hit, test_ndcg)
    print(final_result_str)


    # if args.save_flag == 0:
        
    save_path = 'Output/'+args.out_dir+'/'+args.city+'_'+args.cate+'_'+args.data_type+'.txt'
    ensureDir(save_path)
    f = open(save_path, 'a')

    f.write("XGboost: num_boost_round=%d, max_depth=%d, min_child_weight=%d, scale_pos_weight=%.4f, learning_rate=%.4f, gamma=%.4f, subsample=%.4f, colsample_bytree=%.4f, %s\n" 
              %(args.num_boost_round, args.max_depth, args.min_child_weight, args.scale_pos_weight, args.learning_rate, args.gamma, args.subsample, args.colsample_bytree, final_result_str))

    f.close()
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # output the prediction over leaf nodes.
    # elif args.save_flag == 1:
    save_path = '../Data/Model/%s_%s_%s/%d_%d/' % (args.city, args.cate, args.data_type, args.num_boost_round, args.max_depth)
    ensureDir(save_path)

    min_mapping_path = '%smin_mapping.txt' % save_path
    base_mapping_path = '%sbase_mapping.txt' % save_path
    model_path = '%smodel.dump' % save_path
    params_path = '%sparams.json' % save_path

    best_valid_log_loss, best_test_log_loss = load2json(params_path)
    print('best: valid=[%.6f], test=[%.6f]' % (best_valid_log_loss, best_test_log_loss))
    
    if best_valid_log_loss > valid_log_loss:
        ind_params['train_log_loss'] = train_log_loss
        ind_params['valid_log_loss'] = valid_log_loss
        ind_params['test_log_loss'] = test_log_loss

        ind_params['num_boost_round'] = args.num_boost_round

        ind_params['valid_hit'] = valid_hit
        ind_params['valid_ndcg'] = valid_ndcg
        ind_params['test_hit'] = test_hit
        ind_params['test_ndcg'] = test_ndcg

        y_train_pred = xgb_model.predict(train_dm, ntree_limit=0, pred_leaf=True)
        y_valid_pred = xgb_model.predict(valid_dm, ntree_limit=0, pred_leaf=True)
        y_test_pred = xgb_model.predict(test_dm, ntree_limit=0, pred_leaf=True)

        min_mapping, base_mapping = node_col_mapping(y_train_pred, y_valid_pred, y_test_pred)



        save2txt(min_mapping, min_mapping_path)
        save2txt(base_mapping, base_mapping_path)
        save2json(ind_params, params_path)

        xgb_model.save_model(model_path)

        print('save model & params done.')
    else:
        print('suboptimal model not saved')