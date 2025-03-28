import argparse
import os
from GE_train import GE_training
from Utility.constant import *
from Datasets import select_data
from Utility.parser import args
from probe.CentProbe import train_CentProbe
from probe.DistProbe import train_DistProbe
from probe.GstructProbe import cal_GstructProbe
from probe.CategoryProbe import category_probe
import sys

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        # self.log = open(filename, "a")
        self.log = open(filename, "w")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass
    def reset(self):
        self.log.close()
        sys.stdout=self.terminal
def train_GE_model():
    for i in range(len(List_permit_args)):
        param = List_permit_args[i]
        # if param['data'] != args.dataset: continue
        if param['meta_feat']:
            node_emb_dim = None
        else:
            node_emb_dim = args.node_emb_dim
        dataset = select_data(param['data'], node_feat_dim=node_emb_dim)


        for ge in List_GE_Model:
            log_root = os.path.join("results", args.dataset)
            if not os.path.exists(log_root):
                os.makedirs(log_root)
            log_path = os.path.join("results", args.dataset, "{}.txt".format(ge))
            sys.stdout = Logger(log_path)

            param['ge_model'] = ge
            if args.test_one == True:
                if ge != args.test_model:
                    continue
            GE_training(param, dataset)

            sys.stdout.reset()
def train_cal_Probe():
    for cutoff in range(2,args.cutoff):
        # cutoff = 5
        Dict_probe_score = {(arg['data'], arg['downstreamTask'], args.node_emb_dim): {} for arg in List_permit_args}
        for probe in Dict_probe_emb.keys():
            if probe == "CentProbe" and cutoff > 2: continue
            list_data_task = Dict_probe_emb[probe]
            for id_permit_args in list_data_task:
                permit_args = List_permit_args[id_permit_args]
                data_name = permit_args['data']
                if permit_args['data'] != args.dataset: continue
                downstreamTask = permit_args['downstreamTask']
                feat = 'meta' if args.meta_emb else 'rand' + str(args.node_emb_dim)
                data = select_data(data_name, args.meta_emb)
                for ge_model in List_GE_Model:
                    if args.test_one == True:
                        if ge_model != args.test_model:
                            continue
                    args.ge_model = ge_model
                    emb = torch.load(os.path.join(EMB_ROOT, data_name, f"nemb-{feat}-{ge_model}-{downstreamTask}.pt"))
                    probe_score = probing(probe, emb, data, cutoff)
                    if isinstance(probe_score, dict):
                        Dict_probe_score[(data_name, downstreamTask, args.node_emb_dim)][ge_model] = probe_score
                    else:
                        Dict_probe_score[(data_name, downstreamTask, args.node_emb_dim)][ge_model] = probe_score

                    print(data_name, downstreamTask, args.node_emb_dim, ge_model, probe_score)
def probing(probe, emb, dataset, cutoff):
    probe_score = None
    if probe == 'CentProbe':
        probe_score = train_CentProbe(emb, dataset)
    elif probe == 'DistProbe':
        probe_score = train_DistProbe(emb, dataset, cutoff)
    elif probe == 'GstructProbe':
        probe_score = cal_GstructProbe(emb, dataset)
    elif probe == 'ClusterProbe':
        probe_score = category_probe(emb, dataset, probe)
    elif probe == 'ContrastiveProbe':
        probe_score = category_probe(emb, dataset, probe)
    return probe_score




if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--Embedding', type=int, default=0)
    # parser.add_argument('--Probe', type=int, default=0)
    # args = parser.parse_args()
    train_GE_model()

    #
    train_cal_Probe()


