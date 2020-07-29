# import pandas as pd
import json
import os
import ast
# from factory import *
import operator
import random
import time
import numpy as np
from scipy.sparse import csc_matrix
from random import shuffle

class FinalLoad(object):
    def __init__(self, city, cate, data_type):
        self.data_type = data_type
        self.split_str = '../Data/Split-final/' + city + '_' + cate
        self.map_str = '../Data/Map/' + city + '_' + cate

        self.pos_str = '../Data/Comp/' + city + '_' + cate

        self.train_neg_sampling_ratio = 50
        self.test_neg_sampling_ratio = 50
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.field_dict = self._get_field_dict(self.map_str + '_field_dict.json')
        self.user_num = self.field_dict['user_num']
        self.item_num = self.field_dict['item_num']
        self.field_num = self.field_dict['field_num']
        self.mask_field_id = self.field_num + self.user_num + self.item_num
        # print(self.field_num, self.user_num, self.item_num, self.mask_field_id)
        self.move_field_id = -1
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # the original id matrix before padding with the field ids.
        self.train_id = np.loadtxt(self.split_str + '_Train.split').astype(int)
        # np.random.shuffle(self.train_id)
        self.valid_id = np.loadtxt(self.split_str + '_Valid.split').astype(int)
        self.test_id = np.loadtxt(self.split_str + '_Test.split').astype(int)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.user_map = np.loadtxt(self.map_str + '_User.map').astype(int)
        self.item_map = np.loadtxt(self.map_str + '_Item.map').astype(int)


    def _fetch_raw_data(self, flag='train'):
        t1 = time.time()
        if flag == 'train':
            data_id = self.train_id
            existing_id = list()
        elif flag == 'valid':
            data_id = self.valid_id
            existing_id = self.train_id
        else:
            data_id = self.test_id
            existing_id = self.train_id

        ui_list, comp_list, y_list = self._concate_map_id(data_id, existing_id, self.user_map, self.item_map, self.field_num,
                                                 self.user_num, self.mask_field_id)
        # if flag == 'train':
        #     move_field_id = self.mask_field_id
        #     csc_comp_list, move_field_id = self._convert_onehot_csc(comp_list, move_field_id)
        #     self.move_field_id = move_field_id
        # else:
        #     move_field_id = self.move_field_id
        #     csc_comp_list, move_field_id = self._convert_onehot_csc(comp_list, move_field_id)

        print('%s dataset fetch batch data@%.4f' % (flag, time.time() - t1))
        return ui_list, comp_list, comp_list, y_list

    def _concate_map_id(self, data_id, existing_id, user_map, item_map, field_num, user_num, mask_field_id):
        t1 = time.time()
        uid_list = data_id[:, 0]
        iid_list = data_id[:, 1]
        y_list = data_id[:, 2]

        ufield_list = user_map[uid_list]
        ifield_list = item_map[iid_list]

        keep_id, data_id = self._padding_id(data_id, existing_id, field_num, user_num, mask_field_id)

        if self.data_type == 'id_attr':
            comp_list = np.concatenate((data_id[:, 0:2], ufield_list, ifield_list), axis=1)
        elif self.data_type == 'attr':
            comp_list = np.concatenate((ufield_list, ifield_list), axis=1)

        # print('\tconcatenate the matrices of id and attributes@%.4f s; shape=(%d,%d); min-max=(%d,%d)'
        #       % (time.time()-t1, comp_list.shape[0], comp_list.shape[1], np.min(comp_list), np.max(comp_list)))
        return keep_id, comp_list, y_list

    def _padding_id(self, data_id, existing_id, field_num, user_num, mask_field_id):
        t1 = time.time()

        result_id = np.zeros(data_id.shape)
        keep_id = np.zeros([data_id.shape[0], 2])

        result_id[:, 2] = data_id[:, 2]

        keep_id[:, 0] = data_id[:, 0]
        keep_id[:, 1] = data_id[:, 1] + user_num

        ex_uid_set = set([])
        ex_iid_set = set([])

        if len(existing_id) != 0:
            ex_uid_set = set(existing_id[:, 0])
            ex_iid_set = set(existing_id[:, 1])

        for i, pair_ids in enumerate(data_id):
            uid = pair_ids[0]
            iid = pair_ids[1]
            # uid in the list of existing id needs to be changed to uid + field_num
            # uid not in the list needs to be set as mask_field_id.
            if len(ex_uid_set) != 0 and uid not in ex_uid_set:
                result_id[i, 0] = mask_field_id
            else:
                result_id[i, 0] = uid + field_num
            # analogously with uid
            if len(ex_iid_set) != 0 and iid not in ex_iid_set:
                result_id[i, 1] = mask_field_id
            else:
                result_id[i, 1] = iid + field_num + user_num

        # print('padding the id fields@%.4f s' % (time.time() - t1))
        return keep_id, result_id


    def _convert_onehot_csc(self, comp_list, move_field_id):
        t1 = time.time()

        csc_rows = list()
        csc_cols = list()

        for pairs_index, org_pairs in enumerate(comp_list):
            temps = org_pairs[org_pairs < move_field_id]

            for field_index in temps:
                # if field_index not in self.remove_field_ids:
                csc_rows.append(pairs_index)
                csc_cols.append(field_index)

        csc_data = np.ones((len(csc_rows),), dtype=np.int)
        num_rows = max(csc_rows) + 1
        num_cols = max(csc_cols) + 1

        csc_comp_list = csc_matrix((csc_data, (csc_rows, csc_cols)), shape=(num_rows, num_cols))
        # print('convert to one hot csc matrix@%.4f; shape=(%d,%d); max popularity: %d' % (time.time() - t1, num_rows, num_cols, max(csc_data)))
        return csc_comp_list, num_cols


    def _get_field_dict(self, path):
        with open(path, 'r') as json_str:
            field_dict = json.load(json_str)
        return field_dict


if __name__ == '__main__':
    # city = 'London'
    city = 'New_York_City'
    cate = 'Restaurant'
    data_type = 'attr'

    my_load = FinalLoad(city, cate, data_type)

    train_ui, train_list, train_comp_list, train_y_list = my_load._fetch_raw_data(flag='train')

    test_ui, test_list, test_comp_list, test_y_list = my_load._fetch_raw_data(flag='test')