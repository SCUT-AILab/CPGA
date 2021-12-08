import scipy.io as scio
from train_generator import *

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='4,5,6,7', help='gpu device_ids for cuda')
    parser.add_argument('--batchsize', default=32, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--source_model', default='./model_source/20211129-1121-cl9_1_resnet50_best.pkl')
    parser.add_argument('--ce_generator', default='./model_source/generator_no_contrastive_visda_visual.pkl')
    parser.add_argument('--contras_generator', default='./model_source/generator_visda_visual.pkl')


    args = parser.parse_args()
    return args


def save_points(embeds, labels, path):
    train_dict = {}
    train_dict['embeds'] = embeds
    train_dict['labels'] = labels
    scio.savemat(path, train_dict)


if __name__ == '__main__':
    args = arg_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    batch_nums = 50
    batch_size = 4
    num_cls = 12
    pos_nums = batch_nums * batch_size

    source_model = torch.load(args.source_model)
    source_model = source_model.cuda()

    contras_generator = torch.load(args.contras_generator)
    contras_generator = contras_generator.eval()

    ce_generator = torch.load(args.ce_generator)
    ce_generator = ce_generator.eval()

    # for the original prototypes
    z = Variable(torch.rand(num_cls * pos_nums, 100)).cuda()  # class_nums
    # Get labels ranging from 0 to n_classes for n rows
    label_t = torch.linspace(0, num_cls - 1, steps=num_cls).long()
    for ti in range(pos_nums * num_cls // num_cls - 1):
        label_t = torch.cat([label_t, torch.linspace(0, num_cls - 1, steps=num_cls).long()])
    labels = Variable(torch.cat([torch.randint(0, num_cls, (num_cls * pos_nums % num_cls,)), label_t])).cuda()
    z = z.contiguous()
    labels = labels.contiguous()

    # for the visual prototypes
    contras_prototypes = contras_generator(z, labels)
    contras_output = source_model.vis_fc(contras_prototypes)

    contras_output = F.normalize(contras_output)
    save_points(contras_output.detach().cpu().numpy(), labels.detach().cpu().numpy(),
                './visual_prototypes_visda.mat')

    # for the ce prototypes
    ce_prototypes = ce_generator(z, labels)
    ce_output = source_model.vis_fc(ce_prototypes)

    ce_output = F.normalize(ce_output)
    save_points(ce_output.detach().cpu().numpy(), labels.detach().cpu().numpy(),
                './ce_prototypes_visda.mat')
    print('Save embeddings!')
