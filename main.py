import argparse
import os
from train_visda import Trainer
import torch


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0,1', help='gpu device_ids for cuda')
    parser.add_argument('--batchsize', default=64, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--num_class', default=12, type=int)
    parser.add_argument('--source_model_path', default='/mnt/cephfs/home/qiuzhen/244/code/OCT_DAL/model_ada/model_source/20201025-1042-synthesis_resnet101_best.pkl', type=str,
                        help='path to the pre-trained source model')
    parser.add_argument('--max_epoch', default=1400, type=int)
    parser.add_argument('--generator_epoch', default=1000, type=int)
    parser.add_argument('--data_path', default='/mnt/cephfs/home/linhongbin/UDA/dataset/VISDA-C/validation', type=str,
                        help='path to target data')
    parser.add_argument('--label_path', default='./data/visda_real_train.pkl', type=str)

    args = parser.parse_args()
    return args


torch.multiprocessing.set_sharing_strategy('file_system')
args = arg_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
oct_trainer = Trainer(args)
oct_trainer.train()
