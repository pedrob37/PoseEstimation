import argparse
import os
from utils.utils import *
import torch
import models


class BaseOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--data_path', type=str, default='/storage/PoseEstimation-related/OrganisedData', help='Data path')
        parser.add_argument('--job_name', type=str, default='test', help='Test')
        parser.add_argument('--base_dir', type=str, default='/nfs/home/pedro', help='Base directory path')
        parser.add_argument('--gpu_number', type=str, default='0', help='GPU number')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--patch_size', nargs='+', default=40, type=int, help='Size of the patches extracted from the image')
        parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')
        parser.add_argument('--resample', default=False, help='Decide or not to rescale the images to a new resolution')
        parser.add_argument('--new_resolution', default=(1, 1, 1), help='New resolution (if you want to resample the data again during training')

        parser.add_argument('--min_pixel', default=0.1, help='Percentage of minimum non-zero pixels in the cropped label')
        parser.add_argument('--drop_ratio', default=0, help='Probability to drop a cropped area if the label is empty. All empty patches will be dropped for 0 and accept all cropped patches if set to 1')

        parser.add_argument('--ngf', type=int, default=32, help='# of gen filters in first conv layer')
        parser.add_argument('--ndf', type=int, default=32, help='# of discrim filters in first conv layer')
        parser.add_argument('--netD', type=str, default='n_layers', help='selects models to use for netD')
        parser.add_argument('--n_layers_D', type=int, default=5, help='only used if netD==n_layers')
        parser.add_argument('--n_D', type=int, default=1, help='Number of multi-scale discriminators')
        parser.add_argument('--netG', type=str, default='resnet_6blocks', help='selects models to use for netG. Look on Networks3D to see the all list')

        parser.add_argument('--gpu_ids', default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--models', type=str, default='cycle_gan', help='chooses which models to use. cycle_gan')

        # parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--workers', default=10, type=int, help='number of data loading workers')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')

        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--scheduler_gamma', type=float, default=0.995, help='LR scheduler gamma')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')

        # Additional variables
        parser.add_argument('--num_epochs_backbone', default=50, type=int, help='Number of backbone epochs')
        parser.add_argument('--validation_interval', default=10, type=int, help='Validation interval')
        parser.add_argument('--num_joints', default=13, type=int, help='Number of joints to regress')
        parser.add_argument('--num_pafs', default=11, type=int, help='Number of PAFs to regress')
        parser.add_argument('--num_stages', default=7, type=int, help='Number of iterative stage steps')
        parser.add_argument('--num_stages_paf', default=7, type=int, help='Number of PAF stages')
        parser.add_argument('--num_vector_fields', default=3, type=int, help='Number of vector fields')
        parser.add_argument('--num_samples', default=1, type=int, help='Number of samples')
        parser.add_argument('--model', type=str, default='vgg', help='Base model to select')
        parser.add_argument('--load_model', type=str, default='none', help='Preload model or not')
        parser.add_argument('--criterion_heatmap', type=str, default='l1', help='Heatmap loss criterion')
        parser.add_argument('--criterion_paf', type=str, default='l1', help='PAF loss criterion')
        parser.add_argument('--branch_selection', type=str, default='main', help='Branch selection')
        parser.add_argument('--pretrain_path', type=str, default='/storage/PoseEstimation-related/MedicalNet/pretrain',
                            help='Path to pretrained models')
        parser.add_argument('--model_version', type=str, default='old', help='What version of the model to use')
        parser.add_argument('--phase', type=str, default='train', help='Train or inference')
        parser.add_argument('--augmentation_level', type=str, default='none', help='none, light')
        parser.add_argument('--resnet_size', default=18, type=int, help='ResNet size (18, 34, or 50)')
        parser.add_argument('--pretrained_backbone', type=self.str2bool, nargs='?', const=True, default=True,
                            help='Using coordconv or not')
        parser.add_argument('--densenet_approach', type=self.str2bool, nargs='?', const=True, default=True,
                            help='Using densenet approach to 3x3 conv blocks or not')

        self.initialized = True

        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify models-related parser options
        # model_name = opt.model
        # model_option_setter = models.get_option_setter(model_name)
        # parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with the new defaults

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        # expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        # mkdirs(expr_dir)
        # file_name = os.path.join(expr_dir, 'opt.txt')
        # with open(file_name, 'wt') as opt_file:
        #     opt_file.write(message)
        #     opt_file.write('\n')

    def parse(self):

        opt = self.gather_options()

        # process opt.suffix
        # if opt.suffix:
        #     suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
        #     opt.name = opt.name + suffix

        self.print_options(opt)

        opt.gpu_ids = torch.device("cuda", 0)
        torch.cuda.set_device(opt.gpu_ids)

        self.opt = opt
        return self.opt

    # Function for proper handling of bools in argparse
    @staticmethod
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


