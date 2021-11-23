# coding=utf-8
"""
validation
"""

import torch
import numpy as np
import os
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.metrics import auc, confusion_matrix
import torch.utils.data
import argparse
from dataset import visDataset_target


def val_source(net, test_loader):
    net.eval()
    correct = 0
    total = 0

    gt_list = []
    p_list = []

    for i, (inputs, labels, _) in enumerate(test_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        gt_list.append(labels.cpu().numpy())
        with torch.no_grad():
            outputs, _ = net(inputs)
        # 取得分最高的那个类 (outputs.data的索引号)
        output_prob = F.softmax(outputs, dim=1).data
        p_list.append(output_prob[:, 1].detach().cpu().numpy())
        _, predicted = torch.max(outputs, 1)
        total += inputs.size(0)
        num = (predicted == labels).sum()
        correct = correct + num

    acc = 100. * correct.item() / total
    prob_list = np.concatenate(p_list)
    gt_list = np.concatenate(gt_list)

    return acc


def val_pclass(net, test_loader):
    net.eval()
    start_test = True
    with torch.no_grad():
        iter_test = iter(test_loader)
        for i in range(len(test_loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs, _ = net(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    acc = matrix.diagonal() / matrix.sum(axis=1) * 100
    aacc = acc.mean()
    aa = [str(np.round(i, 2)) for i in acc]
    acc = ' '.join(aa)
    return aacc, acc


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='2', help='gpu device_ids for cuda')
    parser.add_argument('--batchsize', default=64, type=int)
    parser.add_argument('--test_path', default='/mnt/cephfs/home/qiuzhen/244/code/OCT_DAL/model_ada/model_visda/20201219-1254max_acca2w-our_best.pkl', type=str,
                        help='path to the pre-trained source model')
    parser.add_argument('--data_path', default='/mnt/cephfs/home/linhongbin/UDA/dataset/VISDA-C/validation', type=str,
                        help='path to target data')
    parser.add_argument('--label_file', default='./data/visda_real_train.pkl', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    net = torch.load(args.test_path).cuda()
    net = net.module
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # grayscale mean/std
    ])

    val_dataset = visDataset_target(args.data_path, args.label_file, train=False, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False,
                                             num_workers=2)
    acc = val_pclass(net, val_loader)
    print(acc)
