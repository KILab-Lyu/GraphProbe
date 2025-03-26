import torch
from sklearn.metrics import precision_score, recall_score, f1_score


class BPR_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 0.00001

    def forward(self, pred, label):
        sign = 1 - 2 * label  # pos 1; neg -1
        loss = torch.mean(sign * pred, dim=-1)  # neg - pos
        return loss


class Metric:
    def __init__(self, top_k=20):
        self.top_k = top_k
        self.ndcg = torch.tensor([], dtype=torch.float16)
        self.n_hit = 0
        self.n_recall = 0
        self.n_precision = 0

        self.pred = torch.tensor([], dtype=torch.int)
        self.label = torch.tensor([], dtype=torch.float)

    def update_rating_mat(self, pred_batch, label_batch, idx_batch):
        assert pred_batch.shape[0] == label_batch.shape[0] == idx_batch.shape[1]

        pred_mat = torch.zeros((torch.max(idx_batch[0]) + 1, torch.max(idx_batch[1]) + 1))
        label_mat = torch.zeros_like(pred_mat)
        pred_mat[idx_batch[0], idx_batch[1]] = pred_batch
        label_mat[idx_batch[0], idx_batch[1]] = label_batch

        pred_topk_values, pred_topk_col_indices = torch.topk(pred_mat, k=self.top_k, dim=-1, sorted=True)
        hits = label_mat.gather(1, pred_topk_col_indices)

        self.n_hit += torch.sum(hits)
        self.n_precision += len(pred_batch) * self.top_k
        self.n_recall += torch.sum(label_batch)

        # ndcg
        position = torch.arange(1, 1 + self.top_k)  # 计算位置关系，从2开始计
        weights = 1 / torch.log2(position + 1)  # 根据位置关系计算位置权重
        dcg = (hits * weights).sum(1)  # 计算DCG
        # 计算iDCG，由于相关性得分为0，1，且经过排序，所以计算前面为1对应weights之和即可。
        idcg = torch.Tensor([weights[:min(n, self.top_k)].sum() for n in label_mat.sum(1).int()])
        ndcg = (dcg.t() / idcg.t())[idcg != 0]
        self.ndcg = torch.cat([self.ndcg, ndcg], dim=0)

    def get_f1_score(self):
        return 2 / (1. / self.get_precision() + 1. / self.get_recall())

    def get_precision(self):
        return self.n_hit / (1.0 * self.n_precision)

    def get_recall(self):
        return self.n_hit / (1.0 * self.n_recall)

    def get_ndcg(self):
        return torch.mean(self.ndcg)

    def update(self, pred_batch, label_batch):
        pred_batch = (pred_batch > 0.5).int()
        self.pred = torch.cat([self.pred, pred_batch])
        self.label = torch.cat([self.label, label_batch])

    def get_metric_values(self, metric_list, task='LinkPredTask'):
        metric_val = {}
        if task == 'LinkPredTask':
            if 'f1' in metric_list:
                metric_val['f1'] = self.get_f1_score()
            if 'recall' in metric_list:
                metric_val['recall'] = self.get_recall()
            if 'precision' in metric_list:
                metric_val['precision'] = self.get_precision()
            if 'ndcg' in metric_list:
                metric_val['ndcg'] = self.get_ndcg()
        else:
            if 'f1' in metric_list:
                metric_val['f1'] = f1_score(self.label, self.pred, average='micro')
            if 'recall' in metric_list:
                metric_val['recall'] = recall_score(self.label, self.pred, average='micro')
            if 'precision' in metric_list:
                metric_val['precision'] = precision_score(self.label, self.pred, average='micro')

        return metric_val
