from time import time
import numpy as np
import scipy as sp
import xgboost as xgb
import random
from sklearn.metrics import log_loss
from scipy.sparse import csc_matrix

import argparse

from load_raw import FinalLoad
from metrics import eval_model
from factory import *
import json
import re
import argparse
from math import ceil


def parse_args():
	parser = argparse.ArgumentParser(description='Run XGBoost.')
	parser.add_argument('--city', nargs='?', default='New_York_City',
						help='City Name in (London, New_York_City, Singapore)')
	parser.add_argument('--cate', nargs='?', default='Restaurant',
						help='Category in (Attractions, Restaurant)')
	parser.add_argument('--data_type', nargs='?', default='attr',
						help='Data Type in (attr, id_attr, id+attr).')

	parser.add_argument('--batch_size', type=int, default=5120,
						help='Batch size.')
	parser.add_argument('--batch_neg_sample', type=int, default=4,
						help='Batch size.')
	parser.add_argument('--num_boost_round', type=int, default=500)

	parser.add_argument('--max_depth', type=int, default=6)
	parser.add_argument('--min_child_weight', type=int, default=1)
	parser.add_argument('--learning_rate', type=float, default=0.2)
	parser.add_argument('--gamma', type=float, default=0.)

	parser.add_argument('--booster', nargs='?', default='gbtree',
						help='Specify which booster to use: (gbtree, gblinear or dart.)')
	parser.add_argument('--n_jobs', type=int, default=4)
	parser.add_argument('--subsample', type=float, default=1.)
	parser.add_argument('--colsample_bytree', type=float, default=1.)
	parser.add_argument('--reg_alpha', type=float, default=0.,
						help='L1 regularization term on weights')
	parser.add_argument('--reg_lambda', type=float, default=1.,
						help='L2 regularization term on weights')
	parser.add_argument('--scale_pos_weight', type=float, default=1,
						help='Balancing of positive and negative weights.')
	parser.add_argument('--base_score', type=float, default=0.5,
						help='The initial prediction score of all instances, global bias.')
	parser.add_argument('--merge', type=int, default=0)
	parser.add_argument('--save_flag', type=int, default=1,
						help='Whether save the model & params')
	return parser.parse_args()

class NodeGenerator(object):
	def __init__(self, args):
		self.all_neg_sampling = 50

		city = args.city
		cate = args.cate
		data_type = args.data_type

		self.mapping_matrix, self.weight_matrix, self.xgb_model = self._load_xgb_settings(args)
		self.ui_train, self.ui_valid, self.ui_test, self.raw_train, self.raw_valid, self.raw_test, self.node_train, self.node_valid, self.node_test, self.y_train, self.y_valid, self.y_test = self._load_raw_features(args)
		
		print(self.ui_train.shape, self.raw_train.shape, self.node_train.shape, self.y_train.shape)
		
		print('ui_M=%d; raw_M=%d; nodes_M=%d' % (self.ui_M, self.raw_M, self.nodes_M))

	def _load_xgb_settings(self, args):
		t1 = time()
		load_path = '../Data/Model/%s_%s_%s/%d_%d/' % (args.city, args.cate, args.data_type, args.num_boost_round, args.max_depth)
		
		model_path = '%smodel.dump' % load_path
		params_path = '%sparams.json' % load_path

		ind_params = self._json2dict(params_path)

		xgb_model = xgb.Booster(ind_params)
		xgb_model.load_model(model_path)

		mapping_matrix, weight_matrix = self._node_weight_mapping(xgb_model, args.max_depth)

		print('model & params load done@%4.fs' % (time() - t1))

		return mapping_matrix, weight_matrix, xgb_model


	def _node_weight_mapping(self, xgb_model, max_depth):
		tree_list = xgb_model.get_dump()
		mapping_matrix = np.zeros([pow(2, max_depth+1), len(tree_list)]) - 1

		pattern = re.compile(r'[0-9]+:leaf=[-0-9\.]+')
		weight_list = list()
		mapping_count = 0
		
		for i, tree in enumerate(tree_list):
			for node in re.findall(pattern, tree):
				j = int(node.split(':leaf=')[0])
				weight_list.append(float(node.split(':leaf=')[1]))
				mapping_matrix[j,i] = mapping_count
				mapping_count += 1

		self.nodes_M = mapping_count
		
		weight_matrix = np.array(weight_list)
		return mapping_matrix, weight_matrix

	def _load_raw_features(self, args):
		t1 = time()
		data_load = FinalLoad(args.city, args.cate, args.data_type)
		self.raw_M = data_load.mask_field_id + 1
		self.ui_M = data_load.user_num + data_load.item_num
		# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		ui_train, raw_train, x_train, y_train = data_load._fetch_raw_data(flag='train')
		ui_valid, raw_valid, x_valid, y_valid = data_load._fetch_raw_data(flag='valid')
		ui_test, raw_test, x_test, y_test = data_load._fetch_raw_data(flag='test')

		dm_train = xgb.DMatrix(x_train, label=y_train)
		dm_valid = xgb.DMatrix(x_valid, label=y_valid)
		dm_test = xgb.DMatrix(x_test, label=y_test)
		# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		node_train = self.xgb_model.predict(dm_train, ntree_limit=0, pred_leaf=True)
		node_valid = self.xgb_model.predict(dm_valid, ntree_limit=0, pred_leaf=True)
		node_test = self.xgb_model.predict(dm_test, ntree_limit=0, pred_leaf=True)

		# node_train = self._node_encoding(node_train, self.mapping_matrix)
		# node_valid = self._node_encoding(node_valid, self.mapping_matrix)
		# node_test = self._node_encoding(node_test, self.mapping_matrix)
		# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		yy_train_pred = self.xgb_model.predict(dm_train, ntree_limit=0)
		yy_valid_pred = self.xgb_model.predict(dm_valid, ntree_limit=0)
		yy_test_pred = self.xgb_model.predict(dm_test, ntree_limit=0)
		# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		train_log_loss = log_loss(y_train, yy_train_pred)
		valid_log_loss = log_loss(y_valid, yy_valid_pred)
		test_log_loss = log_loss(y_test, yy_test_pred)

		print('xgb out done@%.4f; train=[%.6f]; valid=[%.6f], test=[%.6f]' % (time()-t1, train_log_loss, valid_log_loss, test_log_loss))
		return ui_train, ui_valid, ui_test, raw_train, raw_valid, raw_test, node_train, node_valid, node_test, y_train, y_valid, y_test


	def _fetch_flag_data(self, flag):
		if flag == 'train':
			ui_data = self.ui_train
			raw_data = self.raw_train
			node_data = self.node_train
			y_data = self.y_train
		elif flag == 'valid':
			ui_data = self.ui_valid
			raw_data = self.raw_valid
			node_data = self.node_valid
			y_data = self.y_valid
		else:
			ui_data = self.ui_test
			raw_data = self.raw_test
			node_data = self.node_test
			y_data = self.y_test

		return ui_data, raw_data, node_data, y_data

	def _check_batch_loop(self, flag, batch_size):
		ui_data, raw_data, node_data, y_data = self._fetch_flag_data(flag)
		num_instances = len(y_data)
		num_loop = num_instances / batch_size
		return num_instances, num_loop

	def _generate_train_batch(self, batch_size, batch_neg_sample):

		all_instances = self.ui_train.shape[0]
		all_pos_index = list(np.array(range(all_instances))[0::self.all_neg_sampling+1])

		per_batch_size = ceil(batch_size / (batch_neg_sample + 1))
		all_loop = ceil(len(all_pos_index) / per_batch_size)
		all_batchs = np.zeros([all_loop, batch_size])

		for i in range(all_loop):
			pos_batch = np.array(random.sample(list(all_pos_index), per_batch_size))
			tmp_batch = pos_batch

			for j in range(batch_neg_sample):
				neg_batch = pos_batch + np.random.randint(low=1, high=self.all_neg_sampling, size=per_batch_size)
				tmp_batch = np.concatenate([tmp_batch, neg_batch])

			if len(tmp_batch) != batch_size:
				tmp_batch = np.concatenate([tmp_batch, np.array(random.sample(all_pos_index, batch_size-len(tmp_batch)))])

			np.random.shuffle(tmp_batch)
			all_batchs[i, :] = tmp_batch
		all_batchs = all_batchs.astype(int)
		# print('all batches@(%d ,%d)' % (all_batchs.shape[0], all_batchs.shape[1]))
		return all_batchs, all_loop

	def _fetch_train_batch_data(self, all_batchs, flag, current_loop):
		batch_index = all_batchs[current_loop, :]
		# batch_index = batch_index.tolist()

		ui_data, raw_data, node_data, y_data = self._fetch_flag_data(flag)

		ui_data = ui_data[batch_index]
		raw_data = raw_data[batch_index]
		node_data = node_data[batch_index]

		node_data = self._node_encoding(node_data, self.mapping_matrix)

		y_data = y_data[batch_index]

		return ui_data, raw_data, node_data, y_data

	def _fetch_batch_data(self, flag, batch_size, current_loop):
		t1 = time()
		ui_data, raw_data, node_data, y_data = self._fetch_flag_data(flag)
		num_instances, num_loop = self._check_batch_loop(flag, batch_size)

		if current_loop < num_loop - 1:
			batch_index = range(current_loop * batch_size, (current_loop + 1) * batch_size)
		elif current_loop == num_loop - 1:
			batch_index = range(current_loop * batch_size, num_instances)
		
		ui_data = ui_data[batch_index]
		raw_data = raw_data[batch_index]
		node_data = node_data[batch_index]

		node_data = self._node_encoding(node_data, self.mapping_matrix)

		y_data = y_data[batch_index]
		return ui_data, raw_data, node_data, y_data

	def _json2dict(self, params_path):
		with open(params_path, 'r') as fin:
			data = json.load(fin)

			data.pop('train_log_loss', None)
			data.pop('valid_log_loss', None)
			data.pop('test_log_loss', None)
		return data

	def _node_encoding(self, raw_node_pred, mapping_matrix):
		t1 = time()

		multi_node_pred = np.zeros(shape=raw_node_pred.shape)
		raw_num, col_num = raw_node_pred.shape
		
		for i in range(col_num):
			multi_node_pred[:,i] = mapping_matrix[raw_node_pred[:,i],i]
		
		# print('encode raw node @%.4fs' % (time() - t1))
		return multi_node_pred

if __name__ == '__main__':
	args = parse_args()
	node_generator = NodeGenerator(args)

	batch_size = 5120

	all_batchs, train_num_loop = node_generator._generate_train_batch(batch_size, args.batch_neg_sample)

	t1 = time()
	for i in range(train_num_loop):
		x_ui, x_raw, x_node, y_list = node_generator._fetch_train_batch_data(all_batchs, flag='train', current_loop=i)
		print(np.sum(y_list))
	print('fetch train batch@%.4f' % (time() - t1))

	for flag in ['valid', 'test']:
		t1 = time()
		num_instances, num_loop = node_generator._check_batch_loop(flag=flag, batch_size=batch_size)
		
		for i in range(num_loop):
			x_ui, x_raw, x_node, y_list = node_generator._fetch_batch_data(flag=flag, batch_size=batch_size, current_loop=i)
		print('fetch %s batch@%.4f' % (flag, time() - t1))

