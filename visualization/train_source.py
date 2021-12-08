# coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import os
import torchvision.transforms as transforms
import argparse
import time
import torch.nn.utils.weight_norm as weightNorm
import torch.nn.functional as F

import pickle as pkl
from net.resnet import resnet18, resnet50, resnet101
from torch.utils.data import Dataset
from PIL import Image


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='6', help='gpu device_ids for cuda')
    parser.add_argument('--batchsize', default=32, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--MultiStepLR', default=[2, 4], nargs='+', type=int,
                        help='reduce LR by 0.1 when the current epoch is in this list')
    parser.add_argument('--max_epoch', default=6, type=int)

    parser.add_argument('--data_path', default='/mnt/cephfs/home/linhongbin/UDA/dataset/VISDA-C/train')
    parser.add_argument('--label_file', default='./data/visda_synthesis_9_1_split.pkl')


    args = parser.parse_args()
    return args


def val_model(net, test_loader):
    net.eval()
    correct = 0
    total = 0

    for i, (inputs, labels, _) in enumerate(test_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            outputs, _ = net(inputs)
        # 取得分最高的那个类 (outputs.data的索引号)
        output_prob = F.softmax(outputs, dim=1).data
        _, predicted = torch.max(outputs, 1)
        total += inputs.size(0)
        num = (predicted == labels).sum()
        correct = correct + num

    acc = 100. * correct.item() / total

    return acc


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


class visDataset(Dataset):
    """
    ASOCT-2 class
    """

    def __init__(self, root, label_file, train=True, transform=None):
        super(visDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train  # training set or test set
        self.label_file = label_file
        with open(self.label_file, 'rb') as f:
            train_dict = pkl.load(f)
        train_list = train_dict['train_list']
        val_list = train_dict['test_list']

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for i in range(len(train_list)):
                self.train_data.append(os.path.join(self.root, train_list[i][0]))
                self.train_labels.append(train_list[i][1])

        else:
            self.test_data = []
            self.test_labels = []
            for i in range(len(val_list)):
                self.test_data.append(os.path.join(self.root, val_list[i][0]))
                self.test_labels.append(val_list[i][1])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img_name, target = self.train_data[index], self.train_labels[index]
        else:
            img_name, target = self.test_data[index], self.test_labels[index]

        img = Image.open(img_name)

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


class Trainer(object):
    def __init__(self):
        self.MSE_loss = nn.MSELoss().cuda()

    def train_half(self, model, optimizer, x_val, y_val, loss):
        """裁剪两半训练"""
        model.train()
        optimizer.zero_grad()
        output, fe = model.forward(x_val)

        # for visual
        pred = model.vis_fc(fe)
        pred = model.class_fc(pred)

        hloss = loss.forward(output, y_val)
        hloss_vis = loss.forward(pred, y_val)

        total_loss = hloss + hloss_vis

        total_loss.backward()
        optimizer.step()
        _, predicted = torch.max(output, 1)
        correct = (predicted == y_val).sum()

        _, predicted_vis = torch.max(pred, 1)
        correct_vis = (predicted_vis == y_val).sum()

        return hloss.item(), correct, output.size(0), correct_vis

    def train(self):
        torch.multiprocessing.set_sharing_strategy('file_system')
        args = arg_parser()
        time_stamp_launch = time.strftime('%Y%m%d') + '-' + time.strftime('%H%M')
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        n_gpus = len(args.gpu.split(','))

        # set parameters
        path = args.data_path
        label_file = args.label_file

        batch_size = args.batchsize
        epochs = args.max_epoch
        best_acc = 0

        print(
            'visda_9_1_split_wn_label_smooth_synthesis_with_argument' + time_stamp_launch + 'model : resnet101  lr: %s' % args.lr)

        net = resnet101(pretrained=True)
        input_dim = net.fc.in_features
        net.fc = weightNorm(nn.Linear(input_dim, 12), name="weight")

        # for the visualization
        net.vis_fc = nn.Linear(2048, 3).cuda()
        net.class_fc = nn.Linear(3, 12).cuda()

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

        # setting1
        transform_train_2 = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # grayscale mean/std
        ])

        train_dataset = visDataset(path, label_file, train=True, transform=transform_train_2)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=2 * n_gpus if n_gpus <= 2 else 2)
        # setting1
        transform_test = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # grayscale mean/std
        ])

        val_dataset = visDataset(path, label_file, train=False, transform=transform_test)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                                 num_workers=2 * n_gpus if n_gpus <= 2 else 2)

        for i in range(epochs):
            accnum = 0.0
            accnum_vis = 0.0
            total = 0.0
            running_loss = []
            net.train()

            for j, (img_data, img_label, ind) in enumerate(train_loader):
                img_data = img_data.cuda()
                img_label = img_label.cuda()
                r_loss, correct_num, bs_num, correct_num_vis = self.train_half(net, optimizer, img_data, img_label,
                                                                               loss)
                running_loss += [r_loss]
                total += bs_num
                accnum += correct_num
                accnum_vis += correct_num_vis

            scheduler.step()

            # evaluation
            acc = val_model(net, val_loader)
            if acc >= best_acc:
                print('save the best model.')
                torch.save(net, './model_source/' + time_stamp_launch + '-visDA_9_1_resnet50_best.pkl')
                best_acc = acc
            else:
                torch.save(net, './model_source/' + time_stamp_launch + '-visDA_9_1_resnet50_last.pkl')


        print("Finished training the source model.")


if __name__ == '__main__':
    oct_trainer = Trainer()
    oct_trainer.train()
