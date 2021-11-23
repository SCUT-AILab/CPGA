import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import DataParallel
from torchvision import transforms
from model_compat import Classifier, Discriminator_fea, generator_fea_deconv, contrastor, LinearAverage
from dataset import visDataset_target
from test import val_pclass
from utils import log, elr_loss, entropy_loss, ls_distance, infoNCE, infoNCE_g
import time
from scipy.spatial.distance import cdist
from tensorboardX import SummaryWriter
from auto_augment import AutoAugment

######################
# params             #
######################

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.loss = nn.CrossEntropyLoss().cuda()
        self.loss_entropy = entropy_loss().cuda()
        self.infonce = infoNCE(class_num=12)
        self.gen_c = infoNCE_g(class_num=12)
        self.writer = SummaryWriter()
        self.alpha = 1
        self.logger = log()
        self.lr = args.lr
        self.same_ind = np.array([])
        self.confi_pre = np.array([])

    def cosine_similarity(self, feature, pairs):
        feature = F.normalize(feature)  # F.normalize只能处理两维的数据，L2归一化
        pairs = F.normalize(pairs)
        similarity = feature.mm(pairs.t())  # 计算余弦相似度
        return similarity  # 返回余弦相似度

    def exp_lr_scheduler(self, optimizer, step, lr_decay_step=2000,
                         step_decay_weight=0.95):

        init_lr = self.lr
        # Decay learning rate by a factor of step_decay_weight every lr_decay_step
        current_lr = init_lr * (step_decay_weight ** (step / lr_decay_step))

        if step % lr_decay_step == 0:
            print('learning rate is set to %f' % current_lr)

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        return optimizer

    def obtain_label(self, loader, my_net):
        my_net.eval()
        start_test = True
        with torch.no_grad():
            iter_test = iter(loader)
            for _ in range(len(loader)):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                t_indx = data[2]
                inputs = inputs.cuda()
                outputs, feas = my_net(inputs)
                if start_test:
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    all_indx = t_indx.float()
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
                    all_indx = torch.cat((all_indx, t_indx.float()), 0)
        all_output = nn.Softmax(dim=1)(all_output)
        max_prob, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

        # 找出模型输出的高概率样本
        model_ind = torch.squeeze((max_prob > self.alpha).nonzero())
        model_ind = model_ind.numpy()
        model_pre = predict.numpy().astype('int')

        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.float().cpu().numpy()

        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])

        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

        for round in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])  # (nclass, f_dim)
            dd = cdist(all_fea, initc, 'cosine')
            pred_label = dd.argmin(axis=1)
            acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

        # obtain the indexes of same prediction between model prediction and clustering
        same_ind = np.where(pred_label.astype('int') == model_pre)[0]
        union = np.intersect1d(model_ind, same_ind)

        return union, pred_label.astype('int')[union]

    def obtain_residue_label(self, loader, my_net, confi_pre, confi_ind):
        my_net.eval()
        start_test = True
        with torch.no_grad():
            iter_test = iter(loader)
            for _ in range(len(loader)):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                t_indx = data[2]
                inputs = inputs.cuda()
                outputs, feas = my_net(inputs)
                if start_test:
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    all_indx = t_indx.float()
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
                    all_indx = torch.cat((all_indx, t_indx.float()), 0)
        all_output = nn.Softmax(dim=1)(all_output)
        max_prob, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.float().cpu().numpy()

        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        initc_output = aff.transpose().dot(all_fea)
        initc_output = initc_output / (1e-8 + aff.sum(axis=0)[:, None])

        # initialize the cluster centroids
        class_tup = []
        for i in confi_pre:
            if i not in class_tup:
                class_tup.append(i)

        aff_confi = np.eye(K)[confi_pre]
        initc = aff_confi.transpose().dot(all_fea[confi_ind])
        for i in range(self.args.num_class):
            if i not in class_tup:
                initc[i] = initc_output[i]

        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

        for round in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])  # (nclass, f_dim)
            dd = cdist(all_fea, initc, 'cosine')
            pred_label = dd.argmin(axis=1)
            acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

        self.logger.info('Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100))

        # calculate the confidence weights
        max_prob, _ = torch.max(F.softmax(torch.from_numpy(1 - dd) / 0.07, dim=1), dim=1)

        pred_label = pred_label.astype('int')
        pred_label[confi_ind] = confi_pre
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

        return pred_label, acc, all_indx.numpy(), max_prob.detach()

    def adaptation_step(self, tgt_img, tgt_pre_label, sor_img, labels, t_indx, model, discriminator, fea_contrastor, optimizer, epoch, sam_confidence):
        model.train()
        discriminator.train()
        fea_contrastor.train()
        optimizer.zero_grad()
        outputs, feas = model(tgt_img)
        reflect_fea = fea_contrastor(feas)
        all_fea = feas.float().cpu()
        all_ref_fea = reflect_fea.float().cpu()

        all_sam_indx, all_in, _ = np.intersect1d(t_indx, t_indx, return_indices=True)

        # calculate neighborhood clustering loss
        feat_t = F.normalize(all_fea.cuda())
        feat_mat = self.lemniscate(feat_t, t_indx)
        feat_mat[:, t_indx] = -1 / 0.05
        ### Calculate mini-batch x mini-batch similarity
        feat_mat2 = torch.matmul(feat_t,
                                 feat_t.t()) / 0.05
        mask = torch.eye(feat_mat2.size(0),
                         feat_mat2.size(0)).type(torch.bool).cuda()
        feat_mat2.masked_fill_(mask.byte(), -1 / 0.05)
        loss_nc = 0.05 * self.loss_entropy(torch.cat([feat_mat,
                                                      feat_mat2], 1).cuda()).cuda()

        adv_loss = Variable(torch.tensor(0.).cuda())
        source_dann = Variable(torch.tensor(0.).cuda())
        if (epoch - self.generator_epoch) < self.warm_epoch:
            adv_loss = ls_distance(discriminator(all_fea.cuda(), self.p), 'target').cuda()
            source_dann = ls_distance(discriminator(sor_img, self.p), 'source').cuda()

        # calculate weighted contrastive loss
        sor_img_con = fea_contrastor(sor_img)

        total_contrastive_loss = Variable(torch.tensor(0.).cuda())
        contrastive_label = torch.tensor([0]).cuda()

        # MarginNCE
        gamma = 0.07
        nll = nn.NLLLoss()
        if len(all_in) > 0:
            for idx in range(len(all_in)):
                pairs4q = self.infonce.get_posAndneg(features=sor_img_con, labels=labels, tgt_label=tgt_pre_label,
                                                     feature_q_idx=t_indx[all_in[idx]],
                                                     co_fea=all_ref_fea[all_in[idx]].cuda())

                # calculate cosine similarity [-1 1]
                result = self.cosine_similarity(all_ref_fea[all_in[idx]].unsqueeze(0).cuda(), pairs4q)

                # MarginNCE
                # softmax
                numerator = torch.exp((result[0][0]) / gamma)
                denominator = numerator + torch.sum(torch.exp((result / gamma)[0][1:]))
                # log
                result = torch.log(numerator / denominator).unsqueeze(0).unsqueeze(0)
                # nll_loss
                contrastive_loss = nll(result, contrastive_label) * sam_confidence[t_indx[all_in[idx]]]
                total_contrastive_loss = total_contrastive_loss + contrastive_loss
            total_contrastive_loss = total_contrastive_loss / len(all_in)

        # obtain prototype of each class
        la_tup = []
        all_class_prototypes = torch.Tensor([]).cuda()
        for i, lab_id in enumerate(labels):
            if lab_id not in la_tup:
                la_tup.append(lab_id)
                all_class_prototypes = torch.cat(
                    (all_class_prototypes, sor_img_con[i].unsqueeze(0)))

        elr_loss = Variable(torch.tensor(0.).cuda())
        if len(all_in) > 0:
            # calculate non-parametric prediction
            similarity_output = self.cosine_similarity(all_ref_fea[all_in].cuda(), all_class_prototypes) / gamma
            elr_loss = self.elr_loss(index=t_indx[all_in], output=similarity_output.cuda(),
                                     label=torch.from_numpy(tgt_pre_label[t_indx[all_in]]).cuda(),
                                     contrastive_loss=total_contrastive_loss, confi_weight=sam_confidence[t_indx[all_in]])

        if (epoch - self.generator_epoch) < self.warm_epoch:
            loss = adv_loss + source_dann
        else:
            loss = elr_loss + loss_nc

        if loss != 0:
            loss.backward()
            optimizer.step()

        self.lemniscate.update_weight(feat_t, t_indx)
        return loss.item(), total_contrastive_loss.item()

    def obtain_pseudo_label_and_confidence_weight(self, test_loader, source_net):
        self.same_ind, self.confi_pre = self.obtain_label(test_loader, source_net)
        pseudo_label, pseudo_label_acc, all_indx, confidence_weight = self.obtain_residue_label(test_loader, source_net,
                                                                                                self.confi_pre,
                                                                                                self.same_ind)
        return pseudo_label, pseudo_label_acc, all_indx, confidence_weight

    def train_prototype_generator(self, epoch, batch_size_g, num_cls, optimizer_g, generator, source_classifier, loss_gen_ce):
        z = Variable(torch.rand(batch_size_g, 100)).cuda()

        # Get labels ranging from 0 to n_classes for n rows
        labels = Variable(torch.randint(0, num_cls, (batch_size_g,))).cuda()
        z = z.contiguous()
        labels = labels.contiguous()
        images = generator(z, labels)
        output_teacher_batch = source_classifier(images)

        # One hot loss
        loss_one_hot = loss_gen_ce(output_teacher_batch, labels)

        if epoch >= 30:
            # contrastive loss
            total_contrastive_loss = torch.tensor(0.).cuda()
            contrastive_label = torch.tensor([0]).cuda()

            # MarginNCE
            margin = 0.5
            gamma = 1
            nll = nn.NLLLoss()
            for idx in range(images.size(0)):
                pairs4q = self.gen_c.get_posAndneg(features=images, labels=labels, feature_q_idx=idx)

                # 余弦相似度 [-1 1]
                result = self.cosine_similarity(images[idx].unsqueeze(0), pairs4q)

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
        loss_G.backward()
        optimizer_g = self.exp_lr_scheduler(optimizer=optimizer_g, step=epoch)
        optimizer_g.step()

    def train(self):
        torch.multiprocessing.set_sharing_strategy('file_system')

        path = self.args.data_path
        label_file = self.args.label_path
        self.logger.info('original train process')
        time_stamp_launch = time.strftime('%Y%m%d') + '-' + time.strftime('%H%M')
        self.logger.info(path.split('/')[-2] + time_stamp_launch)
        best_acc = 0
        model_root = './model_' + path.split('/')[-2]
        if not os.path.exists(model_root):
            os.mkdir(model_root)
        cuda = True
        cudnn.benchmark = True
        batch_size = self.args.batchsize
        batch_size_g = batch_size * 2
        image_size = (224, 224)
        num_cls = self.args.num_class

        self.generator_epoch = self.args.generator_epoch
        self.warm_epoch = 10
        n_epoch = self.args.max_epoch
        weight_decay = 1e-6
        momentum = 0.9

        manual_seed = random.randint(1, 10000)
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)

        #######################
        # load data           #
        #######################
        target_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.435, 0.418, 0.396), (0.284, 0.308, 0.335)),  # grayscale mean/std
        ])

        dataset_train = visDataset_target(path, label_file, train=True, transform=target_train)

        dataloader_train = torch.utils.data.DataLoader(
            dataset=dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=3
        )
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.435, 0.418, 0.396), (0.284, 0.308, 0.335)),  # grayscale mean/std
        ])

        test_dataset = visDataset_target(path, label_file, train=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                  num_workers=3)

        #####################
        #  load model       #
        #####################
        self.lemniscate = LinearAverage(2048, test_dataset.__len__(), 0.05, 0.00).cuda()
        self.elr_loss = elr_loss(num_examp=test_dataset.__len__(), num_classes=12).cuda()

        generator = generator_fea_deconv(class_num=num_cls)

        discriminator = Discriminator_fea()
        source_net = torch.load(self.args.source_model_path)
        source_classifier = Classifier(num_classes=num_cls)
        fea_contrastor = contrastor()

        # load pre-trained source classifier
        fc_dict = source_classifier.state_dict()
        pre_dict = source_net.state_dict()
        pre_dict = {k: v for k, v in pre_dict.items() if k in fc_dict}
        fc_dict.update(pre_dict)
        source_classifier.load_state_dict(fc_dict)

        generator = DataParallel(generator, device_ids=[0, 1])
        discriminator = DataParallel(discriminator, device_ids=[0, 1])
        fea_contrastor = DataParallel(fea_contrastor, device_ids=[0, 1])
        source_net = DataParallel(source_net, device_ids=[0, 1])
        source_classifier = DataParallel(source_classifier, device_ids=[0, 1])
        source_classifier.eval()

        for p in generator.parameters():
            p.requires_grad = True
        for p in source_net.parameters():
            p.requires_grad = True

        # freezing the source classifier
        for name, value in source_net.named_parameters():
            if name[:9] == 'module.fc':
                value.requires_grad = False

        # setup optimizer
        params = filter(lambda p: p.requires_grad, source_net.parameters())
        discriminator_group = []
        for k, v in discriminator.named_parameters():
            discriminator_group += [{'params': v, 'lr': self.lr * 3}]

        model_params = []
        for v in params:
            model_params += [{'params': v, 'lr': self.lr}]

        contrastor_para = []
        for k, v in fea_contrastor.named_parameters():
            contrastor_para += [{'params': v, 'lr': self.lr * 5}]

        #####################
        # setup optimizer   #
        #####################

        # only train the extractor
        optimizer = optim.SGD(model_params + discriminator_group + contrastor_para, momentum=momentum, weight_decay=weight_decay)
        optimizer_g = optim.SGD(generator.parameters(), lr=self.lr, momentum=momentum, weight_decay=weight_decay)

        loss_gen_ce = torch.nn.CrossEntropyLoss()

        if cuda:
            source_net = source_net.cuda()
            generator = generator.cuda()
            discriminator = discriminator.cuda()
            fea_contrastor = fea_contrastor.cuda()
            loss_gen_ce = loss_gen_ce.cuda()
            source_classifier = source_classifier.cuda()

        #############################
        # training network          #
        #############################

        len_dataloader = len(dataloader_train)
        self.logger.info('the step of one epoch: ' + str(len_dataloader))

        current_step = 0
        for epoch in range(n_epoch):
            source_net.train()
            discriminator.train()
            fea_contrastor.train()

            data_train_iter = iter(dataloader_train)

            if epoch < self.generator_epoch:
                generator.train()
                self.train_prototype_generator(epoch, batch_size_g, num_cls, optimizer_g, generator, source_classifier, loss_gen_ce)

            if epoch >= self.generator_epoch:
                if epoch == self.generator_epoch:
                    torch.save(generator, model_root + '/generator_' +path.split('/')[-2] + '.pkl')

                # prototype generation
                generator.eval()
                z = Variable(torch.rand(self.args.num_class*2, 100)).cuda()

                # Get labels ranging from 0 to n_classes for n rows
                label_t = torch.linspace(0, num_cls-1, steps=num_cls).long()
                for ti in range(self.args.num_class*2//num_cls-1):
                    label_t = torch.cat([label_t, torch.linspace(0, num_cls-1, steps=num_cls).long()])
                labels = Variable(label_t).cuda()
                z = z.contiguous()
                labels = labels.contiguous()
                images = generator(z, labels)

                self.alpha = 0.9 - (epoch - self.generator_epoch) / (n_epoch - self.generator_epoch) * 0.2

                # obtain the target pseudo label and confidence weight
                pseudo_label, pseudo_label_acc, all_indx, confidence_weight = self.obtain_pseudo_label_and_confidence_weight(
                    test_loader, source_net)

                i = 0
                while i < len_dataloader:
                    ###################################
                    #        prototype adaptation         #
                    ###################################
                    p = float(i + (epoch - self.generator_epoch) * len_dataloader) / (n_epoch - self.generator_epoch) / len_dataloader
                    self.p = 2. / (1. + np.exp(-10 * p)) - 1
                    data_target_train = data_train_iter.next()
                    s_img, s_label, s_indx = data_target_train

                    batch_size_s = len(s_label)

                    input_img_s = torch.FloatTensor(batch_size_s, 3, image_size[0], image_size[1])
                    class_label_s = torch.LongTensor(batch_size_s)

                    if cuda:
                        s_img = s_img.cuda()
                        s_label = s_label.cuda()
                        input_img_s = input_img_s.cuda()
                        class_label_s = class_label_s.cuda()

                    input_img_s.resize_as_(s_img).copy_(s_img)
                    class_label_s.resize_as_(s_label).copy_(s_label)
                    target_inputv_img = Variable(input_img_s)
                    target_classv_label = Variable(class_label_s)

                    # learning rate decay
                    optimizer = self.exp_lr_scheduler(optimizer=optimizer,
                                                      step=current_step)

                    loss, contrastive_loss = self.adaptation_step(target_inputv_img, pseudo_label,
                                                                                      images.detach(), labels,
                                                                                      s_indx.numpy(), source_net,
                                                                                      discriminator, fea_contrastor, optimizer, epoch, confidence_weight.float())

                    # visualization on tensorboard
                    self.writer.add_scalar('contrastive_loss', contrastive_loss,
                                           global_step=current_step)
                    self.writer.add_scalar('overall_loss', loss, global_step=current_step)
                    self.writer.add_scalar('pseudo_label_acc', pseudo_label_acc,
                                           global_step=current_step)

                    i += 1
                    current_step += 1

                self.logger.info('epoch: %d' % epoch)
                self.logger.info(
                    'contrastive_loss: %f'
                    % (contrastive_loss))
                self.logger.info('loss: %f' % loss)
                accu, ac_list = val_pclass(source_net, test_loader)
                self.writer.add_scalar('test_acc', accu,
                                       global_step=current_step)
                self.logger.info(ac_list)
                if accu >= best_acc:
                    self.logger.info('saving the best model!')
                    torch.save(source_net, model_root + '/' + time_stamp_launch + '_best_model_' + path.split('/')[-2] + '.pkl')
                    best_acc = accu

                self.logger.info('acc is : %.04f, best acc is : %.04f' % (accu, best_acc))
                self.logger.info('================================================')

        self.logger.info('training done! ! !')

