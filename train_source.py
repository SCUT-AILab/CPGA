# coding=utf-8

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import os
from test import val_office
import torchvision.transforms as transforms
from dataset import visDataset
import argparse
from utils import log, entropy_loss, CrossEntropyLabelSmooth
import time
from net.resnet import resnet101
import torch.nn.utils.weight_norm as weightNorm

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', help='gpu device_ids for cuda')
    parser.add_argument('--batchsize', default=32, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--MultiStepLR', default=[8, 15, 45, 60], nargs='+', type=int,
                        help='reduce LR by 0.1 when the current epoch is in this list')
    parser.add_argument('--max_epoch', default=100, type=int)

    parser.add_argument('--data_root', default='/home/linhongbin/UDA/dataset/VISDA-C/train', type=str)
    parser.add_argument('--label_file', default='./data/visda_synthesis_9_1_split.pkl', type=str)

    args = parser.parse_args()
    return args


class Trainer(object):
    def __init__(self):
        self.loss_entropy = entropy_loss().cuda()

    def train_process(self, model, optimizer, x_val, y_val, loss):
        model.train()
        optimizer.zero_grad()
        output, _ = model.forward(x_val)

        hloss = loss.forward(output, y_val)

        hloss.backward()
        optimizer.step()
        _, predicted = torch.max(output, 1)
        correct = (predicted == y_val).sum()

        return hloss.item(), correct, output.size(0)

    def train(self):
        torch.multiprocessing.set_sharing_strategy('file_system')
        args = arg_parser()
        logger = log()
        time_stamp_launch = time.strftime('%Y%m%d') + '-' + time.strftime('%H%M')
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        n_gpus = len(args.gpu.split(','))

        # set parameters
        path = args.data_root
        label_file = args.label_file
        batch_size = args.batchsize
        epochs = args.max_epoch
        best_acc = 0

        dataset_name = path.split('/')[-2]

        logger.info(path.split('/')[-2] + '_' + time_stamp_launch + 'model : resnet101  lr: %s' % args.lr)
        logger.info('dataset is: ' + dataset_name)

        net = resnet101(pretrained=True)
        input_dim = net.fc.in_features
        net.fc = weightNorm(nn.Linear(input_dim, 12), name="weight")
        net = net.cuda()

        param_group = []
        for k, v in net.named_parameters():
            if k[:2] == 'fc':
                param_group += [{'params': v, 'lr': args.lr}]
            else:
                param_group += [{'params': v, 'lr': args.lr * 0.1}]

        loss = CrossEntropyLabelSmooth(num_classes=12).cuda()

        optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=5e-4)
        scheduler = MultiStepLR(optimizer, milestones=args.MultiStepLR, gamma=0.1)

        # training dataset
        transform_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # grayscale mean/std
        ])
        # train_dataset = AsoctDataset(path, label_file, args.imgs_per_volume, train=True, transform=transform_train)
        train_dataset = visDataset(path, label_file, train=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=2 * n_gpus if n_gpus <= 2 else 2)

        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # grayscale mean/std
        ])
        val_dataset = visDataset(path, label_file, train=False, transform=transform_test)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                                 num_workers=2 * n_gpus if n_gpus <= 2 else 2)

        for i in range(epochs):
            accnum = 0.0
            total = 0.0
            running_loss = []
            net.train()

            for j, (img_data, img_label, ind) in enumerate(train_loader):
                img_data = img_data.cuda()
                img_label = img_label.cuda()
                r_loss, correct_num, bs_num = self.train_process(net, optimizer, img_data, img_label,
                                                         loss)
                running_loss += [r_loss]
                total += bs_num
                accnum += correct_num

            scheduler.step()
            avg_loss = np.mean(running_loss)
            temp_acc = 100 * np.float(accnum) / np.float(total)
            logger.info("Epoch %d running_loss=%.3f" % (i + 1, avg_loss))
            logger.info("Accuracy of the prediction on the train dataset : %f %%" % (temp_acc))

            # 验证模型
            acc = val_office(net, val_loader, logger)
            if acc >= best_acc:
                logger.info('saving the best model!')
                torch.save(net, './model_source/' + time_stamp_launch + '-' + dataset_name +'9_1_resnet50_best.pkl')
                best_acc = acc
            else:
                torch.save(net, './model_source/' + time_stamp_launch + '-' + dataset_name +'9_1_resnet50_last.pkl')

            logger.info('best acc is : %.04f, acc is : %.04f' % (best_acc, acc))
            logger.info('================================================')

        logger.info("Finished  Training")


if __name__ == '__main__':
    oct_trainer = Trainer()
    oct_trainer.train()
