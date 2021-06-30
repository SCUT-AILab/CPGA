import torch.nn as nn
from functions import ReverseLayerF
import torch
from utils import initialize_weights
import torch.nn.utils.weight_norm as weightNorm
import torch.nn.functional as F

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(Classifier, self).__init__()
        self.fc = weightNorm(nn.Linear(2048, num_classes), name="weight")
        self.fc.apply(init_weights)
        # self.fc = nn.Linear(2048, num_classes)

    def forward(self, input_data):

        class_label = self.fc(input_data)
        return class_label


class contrastor(nn.Module):
    def __init__(self, output_size=256):
        super(contrastor, self).__init__()
        self.shared_encoder_pred_domain = nn.Sequential()
        self.shared_encoder_pred_domain.add_module('d_se6', nn.Linear(in_features=2048, out_features=2048))
        self.shared_encoder_pred_domain.add_module('relu_se6', nn.ReLU())

        # classify two domain
        # self.shared_encoder_pred_domain.add_module('d_se7', nn.Linear(in_features=1024, out_features=512))
        # self.shared_encoder_pred_domain.add_module('relu_se7', nn.LeakyReLU())
        self.shared_encoder_pred_domain.add_module('d_se8', nn.Linear(in_features=2048, out_features=output_size))

    def forward(self, input_data):
        reflect_vec = self.shared_encoder_pred_domain(input_data)

        return reflect_vec


class LinearAverage(nn.Module):
    def __init__(self, inputSize, outputSize, T=0.05, momentum=0.0):
        super(LinearAverage, self).__init__()
        self.nLem = outputSize
        self.momentum = momentum
        self.register_buffer('params', torch.tensor([T, momentum]));
        self.register_buffer('memory', torch.zeros(outputSize, inputSize))
        self.flag = 0
        self.T = T
        self.memory = self.memory.cuda()

    def forward(self, x, y):
        out = torch.mm(x, self.memory.t()) / self.T
        return out

    def update_weight(self, features, index):
        index = torch.from_numpy(index).cuda()
        if not self.flag:
            weight_pos = self.memory.index_select(0, index.data.view(-1)).resize_as_(features)
            weight_pos.mul_(0.0)
            weight_pos.add_(torch.mul(features.data, 1.0))

            w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_weight = weight_pos.div(w_norm)
            self.memory.index_copy_(0, index, updated_weight)
            self.flag = 1
        else:
            weight_pos = self.memory.index_select(0, index.data.view(-1)).resize_as_(features)
            weight_pos.mul_(self.momentum)
            weight_pos.add_(torch.mul(features.data, 1 - self.momentum))

            w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_weight = weight_pos.div(w_norm)
            self.memory.index_copy_(0, index, updated_weight)
        self.memory = F.normalize(self.memory)#.cuda()

    def set_weight(self, features, index):
        self.memory.index_copy_(0, index, features)


class Discriminator_fea(nn.Module):
    def __init__(self, code_size=100, n_class=2, domain_class=3):
        super(Discriminator_fea, self).__init__()
        self.shared_encoder_pred_domain = nn.Sequential()
        self.shared_encoder_pred_domain.add_module('d_se6', nn.Linear(in_features=2048, out_features=1024))
        self.shared_encoder_pred_domain.add_module('relu_se7', nn.LeakyReLU(0.01))

        # classify two domain
        self.shared_encoder_pred_domain.add_module('d_se8', nn.Linear(in_features=1024, out_features=1))

    def forward(self, input_data, p=0):
        reversed_re_flatten = ReverseLayerF.apply(input_data, p)
        domain_label = self.shared_encoder_pred_domain(reversed_re_flatten)

        return domain_label


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
