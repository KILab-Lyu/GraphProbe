import numpy as np
from torch_geometric.data import Data, Dataset
import json
from time import time
import os
import torch

time0 = time()
class YelpOriginDataset(Dataset):
    def __init__(self, root=None, transform=None, pre_transform=None):
        super(YelpOriginDataset, self).__init__(root, transform, pre_transform)

        self.data_root_path = '../Data/Yelp'
        user_id_set = set()
        with open(os.path.join(self.data_root_path, "yelp_academic_dataset_user.json"), encoding='utf8') as json_file:
            for line in json_file:
                line_contents = json.loads(line)
                if line_contents['review_count'] > 40:
                    user_id_set.add(line_contents['user_id'])

        user_num = len(user_id_set)
        user_id_dict = dict(zip(user_id_set, np.arange(user_num)))
        print(f'{user_num} user read! time:{time() - time0: .3}')

        feature = []
        item_categories = set()
        item_id_set = set()
        with open(os.path.join(self.data_root_path, "yelp_academic_dataset_business.json"), encoding='utf8') as json_file:
            for line in json_file:
                line_contents = json.loads(line)
                if line_contents['review_count'] > 20 and line_contents['categories'] is not None:
                    item_id_set.add(line_contents['business_id'])
                    feat = line_contents['categories'].split(',')
                    item_categories.update(feat)
                    feature.append(feat)

        item_num = len(item_id_set)
        item_id_dict = dict(zip(item_id_set, np.arange(item_num)+user_num))
        print(f'{item_num} item_feature read! time:{time() - time0: .3}')

        item_feature = []
        feat_dim = len(item_categories)
        item_categories = dict(zip(item_categories, np.arange(feat_dim)))
        for i, c in enumerate(feature):
            empty = np.zeros(feat_dim)
            loc = list(map(lambda x: item_categories[x], c))
            empty[loc] = 1
            item_feature.append(empty)
        item_feature = torch.from_numpy(np.stack(item_feature, axis=1)).t()

        edge = []
        missed_node = []
        with open(os.path.join(self.data_root_path, "yelp_academic_dataset_tip.json"), encoding='utf8') \
                as json_file:
            for line in json_file:
                line_contents = json.loads(line)
                user = user_id_dict.get(line_contents['user_id'], None)
                item = item_id_dict.get(line_contents['business_id'], None)
                if user is None or item is None:
                    missed_node.append((user, item))
                else:
                    edge.append([user, item])

        reviews = torch.tensor(edge, dtype=torch.long).t()
        print(f"{len(reviews)} review file read! {len(missed_node)} reviews missed! time:{time() - time0: .4}")



yelp = YelpOriginDataset()