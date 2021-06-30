import logging
import os
import time
import torch
import sklearn.metrics as metrics
from sklearn.metrics import f1_score
import torch.nn as nn
import random
import torch.nn.functional as F

def ls_distance(logits, flag='source'):
    if flag == 'source':
        domain_loss = torch.mean((logits) ** 2)
    else:
        domain_loss = torch.mean((logits - 1) ** 2)
    return domain_loss


class entropy_loss(nn.Module):
    def __init__(self):
        super(entropy_loss, self).__init__()

    def forward(self, logits):
        y_pred = F.softmax(logits, dim=-1)
        size = logits.size(0)
        if size == 0:
            loss = 0.0
        else:
            loss = torch.sum(-y_pred * torch.log(y_pred + 1e-5), dim=1)
        return torch.mean(loss)


class elr_loss(nn.Module):
    def __init__(self, num_examp, num_classes=10, lambda_idx=5, beta=0.9):
        r"""Early Learning Regularization.
        Parameters
        * `num_examp` Total number of training examples.
        * `num_classes` Number of classes in the classification problem.
        * `lambda` Regularization strength; must be a positive float, controling the strength of the ELR.
        * `beta` Temporal ensembling momentum for target estimation.
        """

        super(elr_loss, self).__init__()
        self.num_classes = num_classes
        self.USE_CUDA = torch.cuda.is_available()
        self.beta = beta
        self.lambda_idx = lambda_idx
        self.target = torch.zeros(num_examp, self.num_classes).cuda() if self.USE_CUDA else torch.zeros(num_examp,
                                                                                                        self.num_classes)

    def forward(self, index, output, label, contrastive_loss, confi_weight):
        r"""Early Learning Regularization.
         Args
         * `index` Training sample index, used to track training examples in different iterations.
         * `output` Model's prediction, same as PyTorch provided functions.
         * `label` Labels, same as PyTorch provided loss functions.
         """

        y_pred = F.softmax(output, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        y_pred_ = y_pred.data.detach()
        self.target[index] = self.beta * self.target[index] + (1 - self.beta) * (
                    (y_pred_) / (y_pred_).sum(dim=1, keepdim=True))
        elr_reg = ((1 - (self.target[index] * y_pred).sum(dim=1)).log()).mean()
        final_loss = contrastive_loss + self.lambda_idx * elr_reg
        return final_loss


class infoNCE():
    def __init__(self, features=None, labels=None, class_num=10, feature_dim=512):
        super(infoNCE, self).__init__()
        self.features = features
        self.labels = labels
        self.class_num = class_num

    def get_posAndneg(self, features, labels, tgt_label=None, feature_q_idx=None, co_fea=None):
        self.features = features
        self.labels = labels

        # get the label of q
        q_label = tgt_label[feature_q_idx]

        # get the positive sample
        positive_sample_idx = []
        for i, label in enumerate(self.labels):
            if label == q_label:
                positive_sample_idx.append(i)

        if len(positive_sample_idx) != 0:
            feature_pos = self.features[random.choice(positive_sample_idx)].unsqueeze(0)
        else:
            feature_pos = co_fea.unsqueeze(0)


        # get the negative samples
        negative_sample_idx = []
        for idx in range(features.shape[0]):
            if self.labels[idx] != q_label:
                negative_sample_idx.append(idx)

        negative_pairs = torch.Tensor([]).cuda()
        for i in range(self.class_num - 1):
            negative_pairs = torch.cat((negative_pairs, self.features[random.choice(negative_sample_idx)].unsqueeze(0)))
        if negative_pairs.shape[0] == self.class_num - 1:
            features_neg = negative_pairs
        else:
            raise Exception('Negative samples error!')

        return torch.cat((feature_pos, features_neg))


class infoNCE_g():
    def __init__(self, features=None, labels=None, class_num=10, feature_dim=512):
        super(infoNCE_g, self).__init__()
        self.features = features
        self.labels = labels
        self.class_num = class_num
        self.fc_infoNCE = nn.Linear(feature_dim, 1).cuda()

    def get_posAndneg(self, features, labels, feature_q_idx=None):
        self.features = features
        self.labels = labels

        # get the label of q
        q_label = self.labels[feature_q_idx]

        # get the positive sample
        positive_sample_idx = []
        for i, label in enumerate(self.labels):
            if label == q_label and i != feature_q_idx:
                positive_sample_idx.append(i)

        if len(positive_sample_idx) != 0:
            feature_pos = self.features[random.choice(positive_sample_idx)].unsqueeze(0)
        else:
            feature_pos = self.features[feature_q_idx].unsqueeze(0)

        # get the negative samples
        negative_sample_idx = []
        for idx in range(features.shape[0]):
            if self.labels[idx] != q_label:
                negative_sample_idx.append(idx)

        negative_pairs = torch.tensor([]).cuda()
        for i in range(self.class_num - 1):
            negative_pairs = torch.cat((negative_pairs, self.features[random.choice(negative_sample_idx)].unsqueeze(0)))
        if negative_pairs.shape[0] == self.class_num - 1:
            features_neg = negative_pairs
        else:
            raise Exception('Negative samples error!')

        return torch.cat((feature_pos, features_neg))


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            if m.bias is not None:
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (- targets * log_probs).mean(0).sum()
        else:
            loss = (- targets * log_probs).sum(1)
        return loss


def log():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    log_dir_path = os.path.join(os.getcwd(), 'log')
    if not os.path.exists(log_dir_path):
        os.mkdir(log_dir_path)
    log_name = os.path.join(log_dir_path, time.strftime('%Y%m%d%H%M') + '.txt')
    handler = logging.FileHandler(log_name, mode='w')
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)

    return logger


def analyse(gt_list, p_list, logger, prob=True):
    if prob:
        AUROC = metrics.ranking.roc_auc_score(gt_list, p_list)
        logger.info('AUROC: %.4f' % AUROC)

        p_list[p_list>=0.5] = 1
        p_list[p_list<0.5] = 0

    t_open, f_narrow, f_open, t_narrow = metrics.confusion_matrix(gt_list, p_list).ravel()
    logger.info('true_open: %s ; false_narrow: %s ; false_open: %s ; true_narrow: %s' % (t_open, f_narrow, f_open, t_narrow))

    F1 = f1_score(gt_list, p_list)
    accuracy = (t_narrow+t_open) / (t_narrow+t_open+f_narrow+f_open)
    precision = t_narrow / (t_narrow+f_narrow)
    recall = t_narrow / (t_narrow+f_open)
    logger.info('accuracy: %.4f' % accuracy)
    logger.info('F1: %.4f' % F1)
    logger.info('precision: %.4f ; recall: %.4f' % (precision, recall))
    return AUROC, F1

