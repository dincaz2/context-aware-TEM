import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import json
from collections import Counter
import random
from tqdm import tqdm


class Args():
    def __init__(self, city = 'London', cate = 'Attractions'):
        self.city = city
        self.cate = cate


def read_csv(args):
    csv_path = f'../Data/Raw/{args.city}_{args.cate}_Complete_Review.csv'
    df = pd.read_csv(csv_path, delimiter='\t', index_col=0)
    return df


def split_by_season(df):
    df = df.copy()

    df['rdatetime'] = pd.to_datetime(df['rtime'], infer_datetime_format=True)
    df['rseason'] = df['rdatetime'].map(lambda datetime: 'Summer' if 4 <= datetime.date().month <= 9 else 'Winter')
    df['iid_context'] = df.apply(lambda row: f'{row["iid"]}_{row["rseason"]}', axis=1)

    #     lbe = LabelEncoder()
    #     df['iid_context'] = lbe.fit_transform(df['iid_context'])

    return df


def select_features(df, user_features, item_features, label):
    df = df.copy()

    df['rdatetime'] = pd.to_datetime(df['rtime'], infer_datetime_format=True)
    df.sort_values('rdatetime', inplace=True)

    df[label] = df['rrate'].fillna('0.0').map(lambda rate: 1 if rate == '5.0' else 0)

    return df[user_features + item_features + ['uid_index', 'iid', 'iid_context', label]]


def process_df(args, df):
    print('Started processing df...')
    df = df.copy()

    item_one_hot_columns = ['irating']
    user_one_hot_columns = ['uage', 'ugender', 'ulevel', 'ucountry', 'ucity']
    compound_columns = ['ustyle', 'iattribute', 'itag']

    if args.cate == 'Restaurant':
        item_one_hot_columns.append('iprice')

    print('Processing one hot features...')
    for column in item_one_hot_columns:
        df[column] = df[column].map(str)
    #     one_hot_columns += almost_one_hot_columns

    item_dummies = pd.get_dummies(df[item_one_hot_columns])
    df = df.drop(columns=item_one_hot_columns).join(item_dummies)
    item_features = item_dummies.columns.tolist()

    user_dummies = pd.get_dummies(df[user_one_hot_columns])
    df = df.drop(columns=user_one_hot_columns).join(user_dummies)
    user_features = user_dummies.columns.tolist()

    print('Processing compound features... ', end='')
    for column in compound_columns:
        print(column, end=' ')

        # eval lists. Handle nan values as well
        df[column] = df[column].map(lambda x: [] if type(x) is float else eval(x))
        # get unique values
        c = Counter()
        df[column].map(lambda arr: list(c.update([element]) for element in arr))
        # values = list(c.keys())
        if column == 'iattribute':
            values = [v for v in values if
                      not (v.startswith('trkP') or v.startswith('overrideIndex') or v[0].isnumeric())]

        if column[0] == 'u':
            user_features += [f'{column}_{value}' for value in values]
        elif column[0] == 'i':
            item_features += [f'{column}_{value}' for value in values]

        # create column for each value
        for value in values:
            df[f'{column}_{value}'] = df[column].map(lambda arr: 1 if value in arr else 0)

    print('\nLabel encoding ids...')
    for column in ['iid_context', 'uid_index']:
        lbe = LabelEncoder()
        df[column] = lbe.fit_transform(df[column])

    #     print('Inferring datetime and sorting...')
    #     df['rdatetime'] = pd.to_datetime(df['rtime'], infer_datetime_format=True)
    #     df.sort_values('rdatetime', inplace=True)

    #     print('Setting label...')
    #     df['label'] = df['rrate'].fillna('0.0').map(lambda rate: 1 if rate == '5.0' else 0)

    print('Done processing!')

    return df.drop(columns=compound_columns), user_features, item_features


def write_field_dict(args, user_num, item_num, field_num=0):
    field_dict = {'user_num': user_num, 'item_num': item_num, 'field_num': field_num}

    params_path = f'../Data/Map/{args.city}_{args.cate}_field_dict.json'
    with open(params_path, 'w') as fout:
        json.dump(field_dict, fout)


def write_iid2context_maps(args, df):
    context2iid_map = {iid_context: group['iid'].tolist()[0] for iid_context, group in df.groupby('iid_context')}
    context2iid_map_path = f'../Data/Map/{args.city}_{args.cate}_context2iid_dict.json'
    with open(context2iid_map_path, 'w') as fout:
        json.dump(context2iid_map, fout)

    iid2contexts_map = {iid: group['iid_context'].unique().tolist() for iid, group in df.groupby('iid')}
    iid2contexts_map_path = f'../Data/Map/{args.city}_{args.cate}_iid2contexts_dict.json'
    with open(iid2contexts_map_path, 'w') as fout:
        json.dump(iid2contexts_map, fout)


def write_map(args, df, name, column, features):
    path = f'../Data/Map/{args.city}_{args.cate}_{name}.map'

    df = df[[column] + features]

    #     ids = df[column].unique().tolist()
    #     num_ids = np.max(ids)
    #     missing_ids = [id for id in range(num_ids) if id not in ids]
    #     missing_df = pd.DataFrame([[id] + [None]*len(features) for id in missing_ids], columns=[column]+features)
    #     df = df.append(missing_df)

    df = df.drop_duplicates(subset=[column]).sort_values(column)[features]
    arr = np.array(df)
    np.savetxt(path, arr, fmt='%d')


def split_pos_df_tvt(df):
    df = df[['uid_index', 'iid_context', 'label']]

    pos_instances = df[df['label'] == 1]
    neg_instances = df[df['label'] == 0]

    pos_uids_groups = pos_instances.groupby('uid_index')
    test = None
    train_val = None
    for uid, group in tqdm(pos_uids_groups):
        if group.shape[0] < 14:
            continue
        split_index = int(group.shape[0] * 0.8)
        #         split_index = -1
        #         print(f'df shape: {group.shape[0]}, split_index: {split_index}')
        user_train_val = group[:split_index]
        user_test = group[split_index:]
        if test is None:
            test = user_test
            train_val = user_train_val
        else:
            test = test.append(user_test)
            train_val = train_val.append(user_train_val)

    train, valid = train_test_split(train_val, test_size=0.08)
    #     train = train.append(neg_instances).sample(frac=1)

    return train, valid, test


def populate_with_uninteracted(df, uid_iids_map, all_iids):
    new_df = None
    for index, row in tqdm(list(df.iterrows())):
        user = row['uid_index']
        user_uninteracted_iids = random.sample(all_iids - uid_iids_map[user], 50)

        row_df = pd.DataFrame(
            data={'uid_index': [user] * 51, 'iid_context': [row['iid_context']] + user_uninteracted_iids,
                  'label': [1] + [0] * 50})
        if new_df is None:
            new_df = row_df
        else:
            new_df = new_df.append(row_df)

    return new_df


def write_split(args, df, name):
    path = f'../Data/Split-final/{args.city}_{args.cate}_{name}.split'
    #     df = df[['uid_index', 'iid', 'label']]
    np.savetxt(path, df, fmt='%d')


def main(args):
    label = 'label'
    basic_user_features = ['uage', 'ugender', 'ulevel', 'ustyle', 'ucountry', 'ucity']
    basic_item_features = ['irating', 'iattribute', 'itag']
    if args.cate == 'Restaurant':
        basic_item_features.append('iprice')

    df = read_csv(args)
    df2 = split_by_season(df)
    df3 = select_features(df2, basic_user_features, basic_item_features, label)

    df4, user_features, item_features = process_df(args, df3)
    write_field_dict(args, len(user_features), len(item_features))
    write_map(args, df4, 'User', 'uid_index', user_features)
    write_map(args, df4, 'Item', 'iid_context', item_features)
    write_iid2context_maps(args, df4)

    uid_iids_map = {uid: set(group['iid_context'].tolist()) for uid, group in df4.groupby('uid_index')}
    all_iids = set(df4['iid_context'].unique().tolist())

    train, valid, test = split_pos_df_tvt(df4)
    train = populate_with_uninteracted(train, uid_iids_map, all_iids)
    valid = populate_with_uninteracted(valid, uid_iids_map, all_iids)
    test = populate_with_uninteracted(test, uid_iids_map, all_iids)

    write_split(args, train, 'Train')
    write_split(args, valid, 'Valid')
    write_split(args, test, 'Test')


if __name__ == '__main__':
    args = Args()
    main(args)