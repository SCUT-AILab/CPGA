import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse

from torch.nn import DataParallel
from torch.autograd import Variable


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='4,5', help='gpu device_ids for cuda')
    parser.add_argument('--batchsize', default=32, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--ce_warm_epoch', default=100, type=int)
    parser.add_argument('--max_epoch', default=1000, type=int)
    parser.add_argument('--source_model', default='./model_source/20211129-1121-cl9_1_resnet50_best.pkl')


    args = parser.parse_args()
    return args


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


class generator_fea_deconv(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, input_size=224, class_num=10):
        super(generator_fea_deconv, self).__init__()
        self.input_dim = input_dim
        self.input_size = input_size
        self.class_num = class_num
        self.batch_size = 64

        # label embedding
        self.label_emb = nn.Embedding(self.class_num, self.input_dim)

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            # nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.Linear(1024, 128 * (self.input_size // 16) * (self.input_size // 16)),
            nn.BatchNorm1d(128 * (self.input_size // 16) * (self.input_size // 16)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            # nn.Tanh(),
        )
        initialize_weights(self)

    def forward(self, input, label):
        x = torch.mul(self.label_emb(label), input)
        # x = torch.cat([input, label], 1)
        x = self.fc(x)
        x = x.view(-1, 512, (self.input_size // 32), (self.input_size // 32))
        x = self.deconv(x)
        x = x.view(x.size(0), -1)

        return x


class Trainer(object):
    def __init__(self, args):
        self.MSE_loss = nn.MSELoss().cuda()
        self.loss = nn.CrossEntropyLoss().cuda()
        self.args = args

    def cosine_similarity(self, feature, pairs):
        feature = F.normalize(feature)  # F.normalize只能处理两维的数据，L2归一化
        pairs = F.normalize(pairs)
        similarity = feature.mm(pairs.t())  # 计算余弦相似度
        return similarity  # 返回余弦相似度

    def exp_lr_scheduler(self, optimizer, step, lr_decay_step=2000,
                         step_decay_weight=0.95):

        init_lr = self.args.lr
        # Decay learning rate by a factor of step_decay_weight every lr_decay_step
        current_lr = init_lr * (step_decay_weight ** (step / lr_decay_step))

        if step % lr_decay_step == 0:
            print('learning rate is set to %f' % current_lr)

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        return optimizer

    def cal_accuracy(self, features, labels):
        cnt = 0
        predict = torch.softmax(features, dim=1)
        predict = torch.argmax(predict, dim=1)  # predict label
        for i, pred in enumerate(predict):
            if pred.cpu().numpy() == labels[i].cpu().numpy():
                cnt += 1
        return cnt / features.shape[0]

    def train(self):
        torch.multiprocessing.set_sharing_strategy('file_system')
        print('original train process')
        model_root = './model_visda'
        if not os.path.exists(model_root):
            os.mkdir(model_root)
        cuda = True
        cudnn.benchmark = True
        batch_size = self.args.batchsize
        batch_size_g = batch_size * 2
        num_cls = 12

        # before the warm-up stage, only train with CE loss
        ce_warm_epoch = self.args.ce_warm_epoch
        total_epoch = self.args.max_epoch

        weight_decay = 1e-6
        momentum = 0.9
        manual_seed = random.randint(1, 10000)
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        #####################
        #  load model       #
        #####################
        generator = generator_fea_deconv(class_num=num_cls)
        my_net = torch.load(self.args.source_model)

        generator = DataParallel(generator, device_ids=[0, 1])
        my_net = DataParallel(my_net, device_ids=[0, 1])
        my_net.eval()


        for p in generator.parameters():
            p.requires_grad = True

        optimizer_g = optim.SGD(generator.parameters(), lr=self.args.lr, momentum=momentum, weight_decay=weight_decay)
        loss_similarity = torch.nn.CrossEntropyLoss()

        if cuda:
            my_net = my_net.cuda()
            generator = generator.cuda()
            loss_similarity = loss_similarity.cuda()

        #############################
        # training network          #
        #############################
        for epoch in range(total_epoch):
            my_net.train()

            if epoch < ce_warm_epoch:
                generator.train()
                z = Variable(torch.rand(batch_size_g, 100)).cuda()

                # Get labels ranging from 0 to n_classes for n rows
                labels = Variable(torch.randint(0, num_cls, (batch_size_g,))).cuda()
                z = z.contiguous()
                labels = labels.contiguous()
                images = generator(z, labels)
                vis_feas = my_net.module.vis_fc(images)
                output_teacher_batch = my_net.module.class_fc(vis_feas)
                # calculate the accuracy
                accuracy_pos = self.cal_accuracy(output_teacher_batch, labels)

                # One hot loss
                loss_one_hot = loss_similarity(output_teacher_batch, labels)
                # loss_one_hot = self.adms_loss(output_teacher_batch, labels)

                if epoch >= ce_warm_epoch:
                    # contrastive loss
                    total_contrastive_loss = torch.tensor(0.).cuda()
                    contrastive_label = torch.tensor([0]).cuda()

                    # MarginNCE
                    margin = 0.5
                    gamma = 0.1
                    nll = nn.NLLLoss()
                    for idx in range(images.size(0)):
                        pairs4q = self.gen_c.get_posAndneg(features=images, labels=labels, feature_q_idx=idx)

                        # 余弦相似度 [-1 1]
                        result = self.cosine_similarity(images[idx].unsqueeze(0), pairs4q)

                        # MarginNCE
                        # softmax
                        numerator = torch.exp((result[0][0] - margin) / gamma)
                        denominator = numerator + torch.sum(torch.exp((result / gamma)[0][1:]))
                        # log
                        result = torch.log(numerator / denominator).unsqueeze(0).unsqueeze(0)
                        # nll_loss
                        contrastive_loss = nll(result, contrastive_label)

                        # contrastive_loss = self.criterion(result, contrastive_label)
                        total_contrastive_loss = total_contrastive_loss + contrastive_loss
                    total_contrastive_loss = total_contrastive_loss / images.size(0)
                else:
                    total_contrastive_loss = torch.tensor(0.).cuda()

                # loss of Generator
                optimizer_g.zero_grad()
                loss_G = loss_one_hot + total_contrastive_loss
                print('loss is {:.3f}, acc is {:.3f}'.format(loss_G.item(), accuracy_pos))

                loss_G.backward()
                optimizer_g = self.exp_lr_scheduler(optimizer=optimizer_g, step=epoch)
                optimizer_g.step()

            if epoch == total_epoch - 1:
                torch.save(generator, './model_source/generator_{}_{}_visda_visual.pkl'.format(ce_warm_epoch, total_epoch))

if __name__ == '__main__':
    args = arg_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    oct_trainer = Trainer(args)
    oct_trainer.train()

