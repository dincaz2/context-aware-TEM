import sys
import json
import pickle
# from TEM import ours-batch

class Args():
    def __init__(self, city = 'London', cate = 'Attractions'):
        self.city = city
        self.cate = cate

def load_json(path):
    with open(path, 'r') as json_str:
        field_dict = json.load(json_str)
    return field_dict

def get_opposite_context_iid(context2iid_map, iid2contexts_map, orig_context_iid):
    original_iid = str(context2iid_map[orig_context_iid])
    context_iids = iid2contexts_map[original_iid]
    for context_iid in context_iids:
        if context_iid != orig_context_iid:
            return context_iid
    return orig_context_iid

def pickle_load_model(args):
    model_path = f'../Data/Model/{args.city}_{args.cate}_attr/tem_model.dump'
    with open(model_path, 'rb') as file:
        return pickle.load(file)

def is_context_feature_important(args, uid, iid, threshold):
    # mappers from context iid to original iid and from original iid to its contexts iids
    map_str = f'../Data/Map/{args.city}_{args.cate}'
    context2iid_map = load_json(map_str + '_context2iid_dict.json')
    iid2contexts_map = load_json(map_str + '_iid2contexts_dict.json')

    other_iid = get_opposite_context_iid(context2iid_map, iid2contexts_map, iid)

    loaded_model = pickle_load_model(args)

    initial_pred = loaded_model.predict(uid, iid)
    new_pred = loaded_model.predict(uid, other_iid)

    return abs(new_pred - initial_pred) > threshold

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Use this script to find if context feature is important in this interaction')
        print('python context-importance.py <uid> <iid> <pred-label> [<threshold>]')
    else:
        uid = sys.argv[1]
        iid = sys.argv[2]
        threshold = float(sys.argv[3]) if len(sys.argv) == 4 else 1.1
        if is_context_feature_important(Args(), uid, iid, threshold):
            print('Context is important for this user-item interaction')
        else:
            print('Context is not important for this user-item interaction')