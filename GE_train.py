import os

import psutil
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, \
    accuracy_score, ndcg_score
from time import time

from torch_geometric.nn import Node2Vec
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import DataLoader

from Datasets import select_data, apply_node_perturbation_movielens
from Utility.parser import args
from Utility.helper import Stop_trick, select_GE_model, Update_cal, select_GETest_model
from Utility.metrics import BPR_loss
from Utility.constant import *
from DownstreamTask import *
from Datasets import select_data, apply_node_perturbation


def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024 

def get_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.synchronize() 
        return torch.cuda.memory_allocated() / 1024 / 1024 
    else:
        return 0

def backward(opt1, opt2, train_loss):
    train_loss.backward(retain_graph=False)
    opt1.step(), opt2.step()
    opt1.zero_grad(), opt2.zero_grad()

def GE_training(params, dataset=None, model=None, dataset_disturb=None):

    data_name = params['data']
    learn_mode = params['mode']
    downstreamTask = params['downstreamTask']

    device = torch.device(args.device)
    # device = torch.device('cpu')
    node_emb_dim = args.node_emb_dim
    feat = 'meta' if params['meta_feat'] else 'rand'+str(node_emb_dim)

    args.dataset = data_name
    print("data is", data_name)
    print('task is', 'inductive' if learn_mode else 'transductive')

    print("downstreamTask is", downstreamTask)
    print(f'feature use {feat}')
    print('embedding dim is', node_emb_dim)
    print('device is {}'.format(args.device))

    if dataset is None:
        dataset = select_data(data_name, feat)
    if dataset_disturb is not None:
        if data_name == "movielens":
            dataset = apply_node_perturbation_movielens(dataset,perturbation_ratio=0.001)
        elif data_name == "cora" or data_name == "citeseer":
            dataset = apply_node_perturbation(dataset,perturbation_ratio=0.001, data_name=data_name)
        # elif data_name is "mutag" or "proteins":
        #     dataset = apply_node_perturbation_proteins_mutag(dataset,perturbation_ratio=0.01)
    node_feat_dim = dataset.data.num_node_features
    if model is None:
        ge_model = params['ge_model']
        print("GE_Model is", ge_model)
        args.ge_model = ge_model
        if not args.ge_test:
            GE_model = select_GE_model(ge_model, node_feat_dim, node_emb_dim, device,dataset.data.edge_index)
        else:
            GE_model = select_GETest_model(ge_model, node_feat_dim, node_emb_dim, device, dataset.data.edge_index)
    else:
        ge_model = model.__name__
        print("GE_Model is {}".format(ge_model))
        GE_model = model(in_channels=node_feat_dim, hidden_channels=256, out_channels=node_emb_dim).to(device)
        # params['node_class_num'] =

    if downstreamTask == "NodeClassifyTask":
        DownStreamTask = NodeClassifyTask(node_emb_dim, params['node_class_num'], multi_label=params['multilabel'])

    elif downstreamTask == "LinkPredTask" or downstreamTask == 'GraphReconsTask':
        DownStreamTask = LinkPredTask(node_emb_dim)

    elif downstreamTask == "GraphClassifyTask":
        DownStreamTask = GraphClassifyTask(node_emb_dim, params['graph_class_num'])

    DownStreamTask = DownStreamTask.to(device=device)

    opt1 = torch.optim.Adam(GE_model.parameters(), lr=args.lr)
    opt2 = torch.optim.Adam(DownStreamTask.parameters(), lr=args.lr)

    logger = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
    }
    if downstreamTask == "NodeClassifyTask" and learn_mode == TRANSDUCTIVE:
        result, metric_val, memory_usage = NC_transductive(dataset, GE_model, DownStreamTask, opt1, opt2, device, logger)

    elif downstreamTask == "GraphReconsTask" and learn_mode == TRANSDUCTIVE:
        result, metric_val, memory_usage = GR_transductive(dataset, GE_model, DownStreamTask, opt1, opt2, device, logger)

    elif downstreamTask == "NodeClassifyTask" and learn_mode == INDUCTIVE:
        result, metric_val, memory_usage = NC_inductive(dataset, GE_model, DownStreamTask, opt1, opt2, device, logger, params['multilabel'])

    elif downstreamTask == "LinkPredTask" and learn_mode == INDUCTIVE:
        result, metric_val, memory_usage = LP_inductive(dataset, GE_model, DownStreamTask, opt1, opt2, device, logger)

    elif downstreamTask == "GraphClassifyTask" and learn_mode == INDUCTIVE:
        result, metric_val, memory_usage = GC_inductive(dataset, GE_model, DownStreamTask, opt1, opt2, device, logger)

    if dataset_disturb is None:
        for name in result.keys():
            if name == 'nemb' or name == 'gemb':
                torch.save(result[name], os.path.join(EMB_ROOT, params['data'], f"{name}-{feat}-{ge_model}-{downstreamTask}.pt"))
            else:
                torch.save(result[name], os.path.join(EMB_ROOT, params['data'], f"{name}.pt"))

    print('=======================================================================================')
    # return result, predictions, labels
    return result, metric_val, memory_usage


def NC_transductive(dataset, GE_model, DownStreamTask, opt1, opt2, device, logger):
    data = dataset.data.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    stop_trick = Stop_trick(max_epoch=args.epochs, patience=args.patience)

    node_feat = data.x
    edge_index = data.edge_index

    time0 = time()
    final_usage = 0
    for epoch in range(args.epochs + 1):
        initial_memory = get_memory_usage()
        initial_gpu_memory = get_gpu_memory()
        time1 = time()
        GE_model.train(), DownStreamTask.train()
        if args.ge_model == "Node2Vec":
            node_emb = GE_model.embedding.weight
        elif args.ge_model == "DeepWalk":
            node_emb = GE_model.embedding.weight
        elif args.ge_model == "GCL":
            node_emb = GE_model(data.x, data.edge_index, data.batch)[0]
        # elif args.ge_model == "SGL":
        #     node_emb = GE_model(node_feat, pos_edge_index, batch)
        else:
            node_emb = GE_model(node_feat, edge_index)

        pred = DownStreamTask(node_emb)
        pred = pred.squeeze(dim=0)

        train_label = data.y[data.train_mask]
        train_pred = pred[data.train_mask]
        train_loss = loss_fn(train_pred, train_label)
        train_acc = accuracy_score(train_label.detach().cpu(), torch.argmax(train_pred.detach().cpu(), dim=1))


        backward(opt1, opt2, train_loss)
        final_memory = get_memory_usage()
        final_gpu_memory = get_gpu_memory()
        gpu_usage = final_gpu_memory - initial_gpu_memory
        memory_usage = final_memory - initial_memory
        final_usage = max(gpu_usage+memory_usage, final_usage)

        with torch.no_grad():
            val_label = data.y[data.val_mask]
            val_pred = pred[data.val_mask]
            val_loss = loss_fn(val_pred, val_label)
            val_acc = accuracy_score(val_label.cpu(), torch.argmax(val_pred.cpu(), dim=1))
            if epoch % 100 == 0:
                print(f'epoch:{epoch}',
                      f'acc_train:{train_acc:.4}',
                      f'loss_train:{train_loss.item():.4}',
                      f'acc_val:{val_acc:.4}',
                      f'val_loss:{val_loss.item():.4}',
                      )

        stop_flag, best_epoch, reason, best_model = stop_trick.should_stop_training(val_loss, epoch, GE_model, DownStreamTask)
        if stop_flag:
            print(f'best epoch:{best_epoch}, stop reason:{reason}')
            break

    GE_model.load_state_dict(best_model[0])
    DownStreamTask.load_state_dict(best_model[1])

    with torch.no_grad():
        GE_model.eval(), DownStreamTask.eval()
        if args.ge_model == "Node2Vec":
            node_emb = GE_model.embedding.weight
        elif args.ge_model == "DeepWalk":
            node_emb = GE_model.node_embed.weight
        elif args.ge_model == "GCL":
            node_emb = GE_model(data.x, data.edge_index, data.batch)[0]
        # elif args.ge_model == "SGL":
        #     node_emb = GE_model(node_feat, pos_edge_index, batch)
        else:
            node_emb = GE_model(node_feat, edge_index)
        pred = DownStreamTask(node_emb)
        pred = pred.squeeze(dim=0)
        test_label = data.y[data.test_mask].cpu()
        test_pred = pred[data.test_mask].cpu()

        metric_val = {}
        test_pred = torch.argmax(test_pred, dim=1)
        metric_val['accuracy'] = accuracy_score(test_label, test_pred)
        metric_val['f1'] = f1_score(test_label, test_pred, average='macro')
        metric_val['recall'] = recall_score(test_label, test_pred, average='macro')
        metric_val['precision'] = precision_score(test_label, test_pred, average='macro')
        print(metric_val)

    result = {}
    result['nemb'] = node_emb
    result['edge_index'] = edge_index
    result['node_class'] = data.y
    return result, metric_val, final_usage

def GR_transductive(dataset, GE_model, DownStreamTask, opt1, opt2, device, logger, full=False):
    data = dataset.data.to(device)

    train_ratio = 0.7
    val_ratio = (1 - train_ratio) / 3
    test_ratio = (1 - train_ratio) / 3 * 2
    transform = RandomLinkSplit(num_val=val_ratio, num_test=test_ratio, is_undirected=True)
    train_data, val_data, test_data = transform(data)
    train_pos_edge_index = train_data.edge_index.t()[train_data.edge_label == 1.].t()

    loss_fn = BPR_loss()
    stop_trick = Stop_trick(max_epoch=args.epochs, patience=args.patience)

    final_usage = 0
    for epoch in range(args.epochs + 1):
        initial_memory = get_memory_usage()
        initial_gpu_memory = get_gpu_memory()
        GE_model.train(), DownStreamTask.train()
        node_emb = GE_model(data.x, train_pos_edge_index)

        train_pred = DownStreamTask(node_emb, train_data.edge_label_index)
        train_label = train_data.edge_label
        train_loss = loss_fn(train_pred, train_label)
        train_auc = roc_auc_score(train_label.detach().cpu(), train_pred.detach().cpu(), average='macro')

        backward(opt1, opt2, train_loss)
        final_memory = get_memory_usage()
        final_gpu_memory = get_gpu_memory()
        gpu_usage = final_gpu_memory - initial_gpu_memory
        memory_usage = final_memory - initial_memory
        final_usage = max(gpu_usage+memory_usage, final_usage)

        with torch.no_grad():
            GE_model.eval(), DownStreamTask.eval()
            val_pred = DownStreamTask(node_emb, val_data.edge_label_index)
            val_label = val_data.edge_label
            val_loss = loss_fn(val_pred, val_label)
            val_auc = roc_auc_score(val_label.cpu(), val_pred.cpu(), average='macro')
            if epoch % 100 == 0:
                print(f'epoch:{epoch}',
                      f'auc_train:{train_auc:.4}',
                      f'loss_train:{train_loss.item():.4}',
                      f'auc_val:{val_auc:.4}',
                      f'val_loss:{val_loss.item():.4}',
                      )

        stop_flag, best_epoch, reason, best_model = stop_trick.should_stop_training(val_loss, epoch, GE_model, DownStreamTask)
        if stop_flag:
            print(f'best epoch:{best_epoch}, stop reason:{reason}')
            break

    GE_model.load_state_dict(best_model[0])
    DownStreamTask.load_state_dict(best_model[1])

    with torch.no_grad():
        GE_model.eval(), DownStreamTask.eval()
        test_pred = DownStreamTask(node_emb, test_data.edge_label_index)
        test_pred = test_pred.cpu()
        test_score = (test_pred > 0.5)
        test_label = test_data.edge_label.cpu()

    metric_val = {}
    metric_val['auc'] = roc_auc_score(test_label, test_pred, average='macro')
    metric_val['f1'] = f1_score(test_label, test_score, average='macro')
    metric_val['recall'] = recall_score(test_label, test_score, average='macro')
    metric_val['precision'] = precision_score(test_label, test_score, average='macro')
    metric_val['ap'] = average_precision_score(test_label, test_pred, average='macro')
    print(metric_val)

    result = {}
    result['nemb'] = node_emb
    result['edge_index'] = data.edge_index
    result['node_class'] = data.y
    return result, metric_val, final_usage

def NC_inductive(dataset, GE_model, DownStreamTask, opt1, opt2, device, logger, multilabel):
    if multilabel:
        loss_fn = torch.nn.BCEWithLogitsLoss()
        f1 = 'f1'
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
        f1 = 'f1s'

    stop_trick = Stop_trick(max_epoch=args.epochs, patience=args.patience)

    train_batch_num = int(dataset.len() * 0.6)
    val_batch_num = int(dataset.len() * 0.2)
    final_usage = 0
    for epoch in range(args.epochs):
        initial_memory = get_memory_usage()
        initial_gpu_memory = get_gpu_memory()
        GE_model.train(), DownStreamTask.train()
        train_update_cal = Update_cal(loss_fn)
        for batch in range(train_batch_num):
            data = dataset[batch].to(device)
            node_emb = GE_model(data.x, data.edge_index)
            train_pred = DownStreamTask(node_emb)
            train_label = data.y
            train_update_cal.update(train_label, train_pred)

        train = train_update_cal.cal([f1, 'loss'])
        backward(opt1, opt2, train['loss'])


        final_memory = get_memory_usage()
        final_gpu_memory = get_gpu_memory()
        gpu_usage = final_gpu_memory - initial_gpu_memory
        memory_usage = final_memory - initial_memory
        final_usage = max(gpu_usage+memory_usage, final_usage)


        with torch.no_grad():
            GE_model.eval(), DownStreamTask.eval()
            val_update_cal = Update_cal(loss_fn)
            for batch in range(train_batch_num, train_batch_num + val_batch_num):
                data = dataset[batch].to(device)
                node_emb = GE_model(data.x, data.edge_index)
                val_pred = DownStreamTask(node_emb)
                val_label = data.y
                val_update_cal.update(val_label, val_pred)

            val = val_update_cal.cal([f1, 'loss'])
            if epoch % 200 == 0:
                print(f'epoch:{epoch}',
                      f"f1_train:{train['f1']:.4}",
                      f"loss_train:{train['loss'].item():.4}",
                      f"f1_val:{val['f1']:.4}",
                      f"loss_val:{val['loss']:.4}",
                      )

            stop_flag, best_epoch, reason, best_model = stop_trick.should_stop_training(val['loss'], epoch, GE_model, DownStreamTask)
            if stop_flag:
                print(f'best epoch:{best_epoch}, stop reason:{reason}')
                break

    GE_model.load_state_dict(best_model[0])
    DownStreamTask.load_state_dict(best_model[1])

    predictions = []
    labels = []

    with torch.no_grad():
        test_update_cal = Update_cal()
        GE_model.eval()
        DownStreamTask.eval()
        for batch in range(train_batch_num + val_batch_num, dataset.len()):
            data = dataset[batch].to(device)
            node_emb = GE_model(data.x, data.edge_index)
            test_pred = DownStreamTask(node_emb)
            test_label = data.y

            test_update_cal.update(test_label, test_pred)

            if multilabel:
                preds = torch.sigmoid(test_pred) >= 0.5
                preds = preds.int().cpu().tolist()
            else:
                preds = test_pred.argmax(dim=1).cpu().tolist()

            labels_batch = test_label.cpu().tolist()

            labels.extend(labels_batch)
            predictions.extend(preds)

        metric_val = test_update_cal.cal([f1, "acc"])
        print(metric_val)

    with torch.no_grad():
        data = dataset.data.cpu()
        GE_model = GE_model.cpu()
        result = {}
        result['nemb'] = GE_model(data.x, data.edge_index)
        result['edge_index'] = data.edge_index
        result['node_class'] = data.y
    return result, metric_val, final_usage

def LP_inductive(dataset, GE_model, DownStreamTask, opt1, opt2, device, logger):
    loss_fn = BPR_loss()
    stop_trick = Stop_trick(max_epoch=args.epochs, patience=args.patience)

    batch_num = len(dataset)
    train_batch_num = int(batch_num * 0.7)

    add_model = ""
    if args.ge_test == True:
        add_model = "GCN"
    if 1:
    # if os.path.exists("models/GE_model/{}/{}-meta{}.pth".format(args.dataset,args.ge_model+add_model, str(args.meta_emb))) == False:
        final_usage = 0
        for epoch in range(args.epochs + 1):
            initial_memory = get_memory_usage()
            initial_gpu_memory = get_gpu_memory()
            GE_model.train(), DownStreamTask.train()
            train_update_cal = Update_cal()
            for batch in range(train_batch_num):
                data = dataset[batch].to(device)
                node_feat = data.x
                pos_edge_index = data.edge_index
                neg_edge_index = data['neg_edge_index']
                edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
                train_label = torch.cat([torch.ones(pos_edge_index.shape[1]), torch.zeros(neg_edge_index.shape[1])]).to(device)

                if args.ge_model == "Node2Vec":
                    node_emb = GE_model.embedding.weight
                elif args.ge_model == "DeepWalk":
                    node_emb = GE_model.embedding.weight
                elif args.ge_model == "GCL":
                    node_emb = GE_model(data.x, data.edge_index, data.batch)[0]
                # elif args.ge_model == "SGL":
                #     node_emb = GE_model(node_feat, pos_edge_index, batch)
                else:
                    node_emb = GE_model(node_feat, pos_edge_index)

                train_pred = DownStreamTask(node_emb, edge_index)
                train_loss = loss_fn(train_pred, train_label)
                backward(opt1, opt2, train_loss)

                final_memory = get_memory_usage()
                final_gpu_memory = get_gpu_memory()
                gpu_usage = final_gpu_memory - initial_gpu_memory
                memory_usage = final_memory - initial_memory
                final_usage = max(gpu_usage + memory_usage, final_usage)
                train_update_cal.update(train_label, train_pred)

            train_auc = train_update_cal.cal(['auc'])['auc']

            with torch.no_grad():
                GE_model.eval(), DownStreamTask.eval()
                val_update_cal = Update_cal(loss_fn)
                for batch in range(train_batch_num, batch_num):
                    data = dataset[batch].to(device)
                    node_feat = data.x
                    pos_edge_index = data.edge_index
                    neg_edge_index = data['neg_edge_index']
                    edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
                    val_label = torch.cat([torch.ones(pos_edge_index.shape[1]), torch.zeros(neg_edge_index.shape[1])]).to(device)

                    if args.ge_model == "Node2Vec":
                        node_emb = GE_model.embedding.weight
                    elif args.ge_model == "DeepWalk":
                        node_emb = GE_model.embedding.weight
                    elif args.ge_model == "GCL":
                        node_emb = GE_model(data.x, data.edge_index, data.batch)[0]
                    else:
                        node_emb = GE_model(node_feat, pos_edge_index)
                    val_pred = DownStreamTask(node_emb, edge_index)
                    val_update_cal.update(val_label, val_pred)

                val = val_update_cal.cal(['auc', 'loss'])
                if epoch % 100 == 0:
                    print(f'epoch:{epoch}',
                          f'auc_train:{train_auc:.4}',
                          f'loss_train:{train_loss.item():.4}',
                          f"auc_val:{val['auc']:.4}",
                          f"val_loss:{val['loss']:.4}",
                          )

                stop_flag, best_epoch, reason, best_model = stop_trick.should_stop_training(val['loss'], epoch, GE_model, DownStreamTask)
                if stop_flag:
                    print(f'best epoch:{best_epoch}, stop reason:{reason}')
                    break
        # if not args.ge_model == "GCNTest":
        GE_model.load_state_dict(best_model[0])
        DownStreamTask.load_state_dict(best_model[1])
        torch.save(best_model[0], "models/GE_model/{}/{}-meta{}.pth".format(args.dataset,args.ge_model+add_model, str(args.meta_emb)))
        torch.save(best_model[1], "models/DownStreamTask/{}/{}-meta{}.pth".format(args.dataset,args.ge_model+add_model, str(args.meta_emb)))
    else:
        GE_model.load_state_dict(torch.load("models/GE_model/{}/{}-meta{}.pth".format(args.dataset,args.ge_model+add_model, str(args.meta_emb))
                                            ))
        DownStreamTask.load_state_dict(torch.load("models/DownStreamTask/{}/{}-meta{}.pth".format(args.dataset,args.ge_model+add_model, str(args.meta_emb))
                                                  ))
    with torch.no_grad():
        test_update_cal = Update_cal()
        GE_model.eval(), DownStreamTask.eval()
        dataset.is_train = False
        for batch in range(batch_num):
            data = dataset[batch].to(device)
            node_feat = data.x
            pos_edge_index = data.edge_index
            neg_edge_index = data['neg_edge_index']
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            test_label = torch.cat([torch.ones(pos_edge_index.shape[1]), torch.zeros(neg_edge_index.shape[1])])
            if args.ge_model == "Node2Vec":
                node_emb = GE_model.embedding.weight
            elif args.ge_model == "DeepWalk":
                node_emb = GE_model.embedding.weight
            elif args.ge_model == "GCL":
                node_emb = GE_model(data.x, data.edge_index, data.batch)[0]
            else:
                node_emb = GE_model(node_feat, pos_edge_index)
            test_pred = DownStreamTask(node_emb, edge_index)

            test_update_cal.update(test_label, test_pred)


    metric_val = {}
    metric_val = test_update_cal.cal(['auc'])
    test_pred = test_pred.cpu()
    test_label = test_label.cpu()
    test_pred_binary = np.where(test_pred >= 0.5, 1, 0)
    metric_val['auc'] = roc_auc_score(test_label, test_pred, average='macro')
    metric_val['f1'] = f1_score(test_label, test_pred_binary, average='macro')
    metric_val['recall'] = recall_score(test_label, test_pred_binary, average='macro')
    metric_val['precision'] = precision_score(test_label, test_pred_binary, average='macro')
    metric_val['ap'] = average_precision_score(test_label, test_pred_binary, average='macro')
    # test_pred_multilabel = torch.exp(test_pred) / torch.sum(torch.exp(test_pred), axis=1, keepdims=True)

    # metric_val['ndcg'] = ndcg_score(test_label, test_pred_multilabel)
    # metric_val['ndcg'] = ndcg_score(test_label, test_pred)
    print(metric_val)

    with torch.no_grad():
        data = dataset.data.cpu()
        GE_model = GE_model.cpu()
        result = {}
        if args.ge_model == "Node2Vec":
            result['nemb'] = GE_model.embedding.weight
        elif args.ge_model == "DeepWalk":
            result['nemb'] = GE_model.embedding.weight
        elif args.ge_model == "GCL":
            result['nemb'] = GE_model(data.x, data.edge_index, data.batch)
        else:
            result['nemb'] = GE_model(data.x, data.edge_index)
        result['edge_index'] = data.edge_index
        result['node_class'] = data.y
    return result, metric_val, final_usage


def GC_inductive(dataset, GE_model, DownStreamTask, opt1, opt2, device, logger):
    graph_num = len(dataset)
    train_num = int(0.6 * graph_num)
    val_num = int(0.2 * graph_num)
    batch_size = 64

    dataset_shuf = dataset.shuffle()
    train_dataset = dataset_shuf[:train_num]
    val_dataset = dataset_shuf[train_num:train_num+val_num]
    test_dataset = dataset_shuf[train_num + val_num:]
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    stop_trick = Stop_trick(max_epoch=args.epochs, patience=args.patience)
    final_usage = 0
    for epoch in range(args.epochs):
        initial_memory = get_memory_usage()
        initial_gpu_memory = get_gpu_memory()
        train_update_cal = Update_cal(loss_fn)
        GE_model.train(), DownStreamTask.train()
        for data in train_dataloader:
            train_data = data.to(device)
            if args.ge_model == "Node2Vec":
                node_emb = GE_model.embedding.weight
            elif args.ge_model == "DeepWalk":
                node_emb = GE_model.embedding.weight
            elif args.ge_model == "GCL":
                node_emb = GE_model(train_data.x, train_data.edge_index, data.batch)[0]
            else:
                node_emb = GE_model(train_data.x, train_data.edge_index)
            train_pred = DownStreamTask(node_emb, train_data.batch)
            train_label = train_data.y
            train_update_cal.update(train_label, train_pred)

        train = train_update_cal.cal(['loss', 'acc'])

        backward(opt1, opt2, train['loss'])

        final_memory = get_memory_usage()
        final_gpu_memory = get_gpu_memory()
        gpu_usage = final_gpu_memory - initial_gpu_memory
        memory_usage = final_memory - initial_memory
        final_usage = max(gpu_usage+memory_usage, final_usage)

        with torch.no_grad():
            val_update_cal = Update_cal(loss_fn)
            GE_model.eval(), DownStreamTask.eval()
            for data in val_dataloader:
                val_data = data.to(device)
                if args.ge_model == "Node2Vec":
                    node_emb = GE_model.embedding.weight
                elif args.ge_model == "DeepWalk":
                    node_emb = GE_model.embedding.weight
                elif args.ge_model == "GCL":
                    node_emb = GE_model(val_data.x, val_data.edge_index, data.batch)[0]
                else:
                    node_emb = GE_model(val_data.x, val_data.edge_index)
                val_pred = DownStreamTask(node_emb, val_data.batch)
                val_label = val_data.y
                val_update_cal.update(val_label, val_pred)

            val = val_update_cal.cal(['acc', 'loss'])
            if epoch % 100 == 0:
                print(f'epoch:{epoch}',
                      f"acc_train:{train['acc']:.4}",
                      f"loss_train:{train['loss']:.4}",
                      f"acc_val:{val['acc']:.4}",
                      f"val_loss:{val['loss']:.4}",
                      )
            stop_flag, best_epoch, reason, best_model = stop_trick.should_stop_training(val['loss'], epoch, GE_model, DownStreamTask)
            if stop_flag:
                print(f'best epoch:{best_epoch}, stop reason:{reason}')
                break

    GE_model.load_state_dict(best_model[0])
    DownStreamTask.load_state_dict(best_model[1])

    with torch.no_grad():
        test_update_cal = Update_cal()
        GE_model.eval(), DownStreamTask.eval()
        for data in test_dataloader:
            test_data = data.to(device)
            if args.ge_model == "Node2Vec":
                node_emb = GE_model.embedding.weight
            elif args.ge_model == "DeepWalk":
                node_emb = GE_model.embedding.weight
            elif args.ge_model == "GCL":
                node_emb = GE_model(test_data.x, test_data.edge_index, data.batch)[0]
            else:
                node_emb = GE_model(test_data.x, test_data.edge_index)
            test_pred = DownStreamTask(node_emb, test_data.batch)
            test_label = test_data.y
            test_update_cal.update(test_label, test_pred)
        metric_val = test_update_cal.cal(['acc', 'f1c'])

        print(metric_val)

        node_emb = torch.zeros((dataset.data.num_nodes, node_emb.shape[1]))
        # graph_emb = torch.zeros((len(dataset), node_emb.shape[1]))
        i = 0
        for graph_id in range(graph_num):
            data = dataset[graph_id].to(device)
            if args.ge_model == "Node2Vec":
                batch_node_emb = GE_model.embedding.weight
            elif args.ge_model == "DeepWalk":
                batch_node_emb = GE_model.embedding.weight
            elif args.ge_model == "GCL":
                batch_node_emb = GE_model(data.x, data.edge_index, data.batch)[0]
            else:
                batch_node_emb = GE_model(data.x, data.edge_index)
            node_emb[i:i + batch_node_emb.shape[0]] = batch_node_emb
            i += batch_node_emb.shape[0]


        result = {}
        result['nemb'] = node_emb
        result['edge_index'] = dataset.data.edge_index
        result['graph_class'] = dataset.data.y

        return result, metric_val, final_usage


"""
def EP_inductive1(dataset, GE_model, DownStreamTask, opt1, opt2, device, logger):
    data = dataset.data.to(device)

    loss_fn = BPR_loss()
    stop_trick = Stop_trick()

    batch_size = 1024
    batch_num = data['num_user'] // batch_size + 1

    train_batch_num = int(batch_num * 0.7)
    val_batch_num = batch_num - train_batch_num

    num_user = data['num_user']
    num_item = data['num_item']

    for epoch in range(args.epochs + 1):
        GE_model.train(), DownStreamTask.train()
        train_user_list = torch.randperm(num_user).to(device)
        train_item_list = data['train_item_list']
        feat_user = data.x[train_user_list]
        feat_item = data.x[train_item_list]
        for batch in range(train_batch_num):
            right_bound = min((batch+1)*batch_size, num_item)
            batch_user = train_user_list[batch*batch_size: right_bound]
            batch_node = torch.cat([batch_user, train_item_list])
            node_feat = torch.cat([feat_user[batch_user], feat_item], dim=0)

            pos_edge_index = subgraph(batch_node, data.edge_index, relabel_nodes=True)[0]
            neg_edge_index = negative_sampling(pos_edge_index, len(node_feat))
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            train_label = torch.cat([torch.ones(pos_edge_index.shape[1]), torch.zeros(neg_edge_index.shape[1])])

            node_emb = GE_model(node_feat, pos_edge_index)
            train_pred = DownStreamTask(node_emb, edge_index)
            train_loss = loss_fn(train_pred, train_label.to(device))

            opt1.zero_grad(), opt2.zero_grad()
            train_loss.backward()
            opt1.step(), opt2.step()

        with torch.no_grad():
            GE_model.eval(), DownStreamTask.eval()
            for batch in range(train_batch_num, batch_num):
                right_bound = min((batch+1)*batch_size, num_item)
                batch_user = train_user_list[batch*batch_size: right_bound]
                batch_node = torch.cat([batch_user, train_item_list])
                node_feat = torch.cat([feat_user[batch_user], feat_item], dim=0)

                pos_edge_index = subgraph(batch_node, data.edge_index, relabel_nodes=True)[0]
                neg_edge_index = negative_sampling(pos_edge_index, len(node_feat))
                edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
                val_label = torch.cat([torch.ones(pos_edge_index.shape[1]), torch.zeros(neg_edge_index.shape[1])])

                node_emb = GE_model(node_feat, pos_edge_index)
                val_pred = DownStreamTask(node_emb, edge_index)
                val_loss = loss_fn(val_pred, val_label.to(device))
            if epoch % 30 == 0: print('val_loss', val_loss.item())

            stop_flag, best_epoch, reason, best_model = stop_trick.should_stop_training(val_loss.item(), epoch, GE_model, DownStreamTask)
            if stop_flag: break

    test_user_list = torch.randperm(num_user).to(device)
    test_item_list = data['test_item_list']
    feat_user = data.x[test_user_list]
    feat_item = data.x[test_item_list]
    with torch.no_grad():
        test_pred, test_label = [], []
        GE_model.eval(), DownStreamTask.eval()
        for batch in range(batch_num):
            right_bound = min((batch + 1) * batch_size, num_item)
            batch_user = test_user_list[batch * batch_size: right_bound]
            batch_node = torch.cat([batch_user, test_item_list])
            node_feat = torch.cat([feat_user[batch_user], feat_item], dim=0)

            pos_edge_index = subgraph(batch_node, data.edge_index, relabel_nodes=True)[0]
            neg_edge_index = negative_sampling(pos_edge_index, len(node_feat))
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            batch_test_label = torch.cat([torch.ones(pos_edge_index.shape[1]), torch.zeros(neg_edge_index.shape[1])])

            node_emb = GE_model(node_feat, pos_edge_index)
            batch_test_pred = DownStreamTask(node_emb, edge_index)

            test_pred.append(batch_test_pred)
            test_label.append(batch_test_label)

        test_pred = torch.cat(test_pred, dim=0).cpu()
        test_label = torch.cat(test_label, dim=0).cpu()
        test_score = (test_pred > 0.)

    metric_val = {}
    metric_val['f1'] = f1_score(test_label, test_score, average='macro')
    metric_val['recall'] = recall_score(test_label, test_score, average='macro')
    metric_val['precision'] = precision_score(test_label, test_score, average='macro')
    metric_val['auc'] = roc_auc_score(test_label, test_pred, average='macro')
    metric_val['ap'] = average_precision_score(test_label, test_pred, average='macro')
    print(metric_val)

    with torch.no_grad():
        data = data.cpu()
        GE_model = GE_model.cpu()
        result = {}
        result['nemb'] = GE_model(data.x, data.edge_index)
        result['edge_index'] = data.edge_index
        result['node_class'] = data.y
    return result
"""

"""
def GC_inductive1(dataset, GE_model, DownStreamTask, opt1, opt2, device, logger):
    graph_num = len(dataset)
    train_num = int(0.6 * graph_num)
    val_num = int((1 - 0.6) / 2 * graph_num)
    test_num = graph_num - (train_num + val_num)
    class_num = DownStreamTask.linear_out.out_features

    rand_prem = torch.randperm(graph_num)

    loss_fn = torch.nn.CrossEntropyLoss()
    stop_trick = Stop_trick(max_epoch=args.epochs, patience=args.patience)

    for epoch in range(args.epochs):
        train_update_cal = Update_cal(loss_fn)
        GE_model.train(), DownStreamTask.train()
        for i, graph_id in enumerate(rand_prem[:train_num]):
            train_data = dataset[graph_id].to(device)
            node_emb = GE_model(train_data.x, train_data.edge_index)
            train_pred = DownStreamTask(node_emb)
            train_label = train_data.y
            train_update_cal.update(train_label, train_pred)

        train = train_update_cal.cal(['loss', 'acc'])
        backward(opt1, opt2, train['loss'])

        with torch.no_grad():
            val_update_cal = Update_cal(loss_fn)
            GE_model.eval(), DownStreamTask.eval()
            for i, graph_id in enumerate(rand_prem[train_num:train_num + val_num]):
                val_data = dataset[graph_id].to(device)
                node_emb = GE_model(val_data.x, val_data.edge_index)
                val_pred = DownStreamTask(node_emb)
                val_label = val_data.y
                val_update_cal.update(val_label.detach(), val_pred.detach())

            val = val_update_cal.cal(['acc', 'loss'])
            if epoch % 100 == 0:
                print(f'epoch:{epoch}',
                      f"acc_train:{train['acc']:.4}",
                      f"loss_train:{train['loss']:.4}",
                      f"acc_val:{val['acc']:.4}",
                      f"val_loss:{val['loss']:.4}",
                      )
            stop_flag, best_epoch, reason, best_model = stop_trick.should_stop_training(val['loss'], epoch, GE_model, DownStreamTask)
            if stop_flag: break

    with torch.no_grad():
        test_update_cal = Update_cal()
        GE_model.eval(), DownStreamTask.eval()
        for i, graph_id in enumerate(rand_prem[train_num + val_num:]):
            test_data = dataset[graph_id].to(device)
            node_emb = GE_model(test_data.x, test_data.edge_index)
            test_pred = DownStreamTask(node_emb)
            test_label = test_data.y
            test_update_cal.update(test_label, test_pred)

        metric_val = {}
        metric_val = test_update_cal.cal(['acc'])
        # test_pred = test_pred.cpu()
        # test_label = test_label.cpu()
        # test_pred = torch.argmax(test_pred, dim=-1)
        # metric_val['f1'] = f1_score(test_label, test_pred, average='macro')
        # metric_val['recall'] = recall_score(test_label, test_pred, average='macro')
        # metric_val['precision'] = precision_score(test_label, test_pred, average='macro')

        print(metric_val)

        node_emb = torch.zeros((dataset.data.num_nodes, node_emb.shape[1]))
        graph_emb = torch.zeros((len(dataset), node_emb.shape[1]))
        i = 0
        for graph_id in range(graph_num):
            data = dataset[graph_id].to(device)
            batch_node_emb = GE_model(data.x, data.edge_index)
            node_emb[i:i + batch_node_emb.shape[0]] = batch_node_emb
            i += batch_node_emb.shape[0]

        result = {}
        result['nemb'] = node_emb
        result['edge_index'] = dataset.data.edge_index
        result['graph_class'] = dataset.data.y

        return result
"""