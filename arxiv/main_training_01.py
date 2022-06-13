import torch
from torch import optim
import os
#from fire import Fire
import glob
from torch.utils.data import DataLoader
import monai
from monai.transforms import Compose, LoadNiftid, AddChanneld, CropForegroundd, RandCropByPosNegLabeld, Orientationd, ToTensord, NormalizeIntensityd
from monai.data import list_data_collate#, sliding_window_inference

from trainTT import trainTT
from models.td_model_provider import create_model, create_optimizer

monai.config.print_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

class Const_strcData:
    pass

pwd = os.getcwd()
#pwd = '/nfs/home/ctangwiriyasakul/DGXwork/projects/2021/202109_Sep/week03/20210921_3DposeEST3D_MCA'

train_root_images = '/media/chayanin/Storage/chin/data2020/SABRE/SABRE_tr'
train_root_heatmaps = '/media/chayanin/Storage/chin/data2020/SABRE/SABRE_tr'
train_root_pafs = '/media/chayanin/Storage/chin/data2020/SABRE/SABRE_tr'

#train_root_images = '/nfs/home/ctangwiriyasakul/DGXwork/datasets/SABRE/SABRE_tr'
#train_root_heatmaps = '/nfs/home/ctangwiriyasakul/DGXwork/datasets/SABRE/SABRE_tr'
#train_root_pafs = '/nfs/home/ctangwiriyasakul/DGXwork/datasets/SABRE/SABRE_tr'

images_suffix = 'SABRE_*'
heatmaps_suffix = 'SABRE_*'
pafs_suffix = 'SABRE_*'

n_subj = 10 #10
batch = 2
patch_sz = 16
n_samp = 24 #18
n_of_epBB = 1001
val_interval = 20
#tmp_catch = '/Users/ct20/Documents/MyPyCharm/2020/2020_version_06/catch/20210907_catch_training'
tmp_catch = '/home/chayanin/PycharmProjects/2020_v04_vrienv002/catch/MCA_20210921_catch_training'
#tmp_catch = '/nfs/home/ctangwiriyasakul/DGXwork/datasets/catch/MCA_20210921_catch_training'
para = Const_strcData()
para.n_subj = n_subj
para.n_batch = batch
para.patch_sz = patch_sz
para.n_samp = n_samp
para.n_of_epBB = n_of_epBB
para.val_interval = val_interval

# Loading data
train_images = sorted(glob.glob(os.path.join(train_root_images, images_suffix, 'post3D', 'SABRE_*_im.nii.gz')))
train_labels = sorted(glob.glob(os.path.join(train_root_heatmaps, heatmaps_suffix, 'post3D', 'SABRE_*_allpointmap.nii.gz')))
train_heatmap28 = sorted(glob.glob(os.path.join(train_root_heatmaps, heatmaps_suffix, 'post3D', 'SABRE_*_heatmap_28.nii.gz')))
train_heatmap29 = sorted(glob.glob(os.path.join(train_root_heatmaps, heatmaps_suffix, 'post3D', 'SABRE_*_heatmap_29.nii.gz')))
train_heatmap30 = sorted(glob.glob(os.path.join(train_root_heatmaps, heatmaps_suffix, 'post3D', 'SABRE_*_heatmap_30.nii.gz')))
train_heatmap31 = sorted(glob.glob(os.path.join(train_root_heatmaps, heatmaps_suffix, 'post3D', 'SABRE_*_heatmap_31.nii.gz')))
train_heatmap32 = sorted(glob.glob(os.path.join(train_root_heatmaps, heatmaps_suffix, 'post3D', 'SABRE_*_heatmap_32.nii.gz')))
train_heatmap33 = sorted(glob.glob(os.path.join(train_root_heatmaps, heatmaps_suffix, 'post3D', 'SABRE_*_heatmap_33.nii.gz')))
train_heatmap34 = sorted(glob.glob(os.path.join(train_root_heatmaps, heatmaps_suffix, 'post3D', 'SABRE_*_heatmap_34.nii.gz')))
train_heatmap35 = sorted(glob.glob(os.path.join(train_root_heatmaps, heatmaps_suffix, 'post3D', 'SABRE_*_heatmap_35.nii.gz')))
train_heatmap36 = sorted(glob.glob(os.path.join(train_root_heatmaps, heatmaps_suffix, 'post3D', 'SABRE_*_heatmap_36.nii.gz')))
train_heatmap37 = sorted(glob.glob(os.path.join(train_root_heatmaps, heatmaps_suffix, 'post3D', 'SABRE_*_heatmap_37.nii.gz')))
train_heatmap38 = sorted(glob.glob(os.path.join(train_root_heatmaps, heatmaps_suffix, 'post3D', 'SABRE_*_heatmap_38.nii.gz')))
train_heatmap39 = sorted(glob.glob(os.path.join(train_root_heatmaps, heatmaps_suffix, 'post3D', 'SABRE_*_heatmap_39.nii.gz')))
train_heatmap40 = sorted(glob.glob(os.path.join(train_root_heatmaps, heatmaps_suffix, 'post3D', 'SABRE_*_heatmap_40.nii.gz')))
train_heatmap41 = sorted(glob.glob(os.path.join(train_root_heatmaps, heatmaps_suffix, 'post3D', 'SABRE_*_heatmap_41.nii.gz')))
train_paf_0_37 = sorted(glob.glob(os.path.join(train_root_pafs, pafs_suffix, 'post3D', 'SABRE_*_paf_0_37.nii.gz')))
train_paf_0_38 = sorted(glob.glob(os.path.join(train_root_pafs, pafs_suffix, 'post3D', 'SABRE_*_paf_0_38.nii.gz')))
train_paf_0_39 = sorted(glob.glob(os.path.join(train_root_pafs, pafs_suffix, 'post3D', 'SABRE_*_paf_0_39.nii.gz')))
train_paf_0_40 = sorted(glob.glob(os.path.join(train_root_pafs, pafs_suffix, 'post3D', 'SABRE_*_paf_0_40.nii.gz')))
train_paf_0_41 = sorted(glob.glob(os.path.join(train_root_pafs, pafs_suffix, 'post3D', 'SABRE_*_paf_0_41.nii.gz')))
train_paf_0_42 = sorted(glob.glob(os.path.join(train_root_pafs, pafs_suffix, 'post3D', 'SABRE_*_paf_0_42.nii.gz')))
train_paf_0_43 = sorted(glob.glob(os.path.join(train_root_pafs, pafs_suffix, 'post3D', 'SABRE_*_paf_0_43.nii.gz')))
train_paf_0_44 = sorted(glob.glob(os.path.join(train_root_pafs, pafs_suffix, 'post3D', 'SABRE_*_paf_0_44.nii.gz')))
train_paf_0_45 = sorted(glob.glob(os.path.join(train_root_pafs, pafs_suffix, 'post3D', 'SABRE_*_paf_0_45.nii.gz')))
train_paf_0_46 = sorted(glob.glob(os.path.join(train_root_pafs, pafs_suffix, 'post3D', 'SABRE_*_paf_0_46.nii.gz')))
train_paf_0_47 = sorted(glob.glob(os.path.join(train_root_pafs, pafs_suffix, 'post3D', 'SABRE_*_paf_0_47.nii.gz')))
train_paf_1_37 = sorted(glob.glob(os.path.join(train_root_pafs, pafs_suffix, 'post3D', 'SABRE_*_paf_1_37.nii.gz')))
train_paf_1_38 = sorted(glob.glob(os.path.join(train_root_pafs, pafs_suffix, 'post3D', 'SABRE_*_paf_1_38.nii.gz')))
train_paf_1_39 = sorted(glob.glob(os.path.join(train_root_pafs, pafs_suffix, 'post3D', 'SABRE_*_paf_1_39.nii.gz')))
train_paf_1_40 = sorted(glob.glob(os.path.join(train_root_pafs, pafs_suffix, 'post3D', 'SABRE_*_paf_1_40.nii.gz')))
train_paf_1_41 = sorted(glob.glob(os.path.join(train_root_pafs, pafs_suffix, 'post3D', 'SABRE_*_paf_1_41.nii.gz')))
train_paf_1_42 = sorted(glob.glob(os.path.join(train_root_pafs, pafs_suffix, 'post3D', 'SABRE_*_paf_1_42.nii.gz')))
train_paf_1_43 = sorted(glob.glob(os.path.join(train_root_pafs, pafs_suffix, 'post3D', 'SABRE_*_paf_1_43.nii.gz')))
train_paf_1_44 = sorted(glob.glob(os.path.join(train_root_pafs, pafs_suffix, 'post3D', 'SABRE_*_paf_1_44.nii.gz')))
train_paf_1_45 = sorted(glob.glob(os.path.join(train_root_pafs, pafs_suffix, 'post3D', 'SABRE_*_paf_1_45.nii.gz')))
train_paf_1_46 = sorted(glob.glob(os.path.join(train_root_pafs, pafs_suffix, 'post3D', 'SABRE_*_paf_1_46.nii.gz')))
train_paf_1_47 = sorted(glob.glob(os.path.join(train_root_pafs, pafs_suffix, 'post3D', 'SABRE_*_paf_1_47.nii.gz')))
train_paf_2_37 = sorted(glob.glob(os.path.join(train_root_pafs, pafs_suffix, 'post3D', 'SABRE_*_paf_2_37.nii.gz')))
train_paf_2_38 = sorted(glob.glob(os.path.join(train_root_pafs, pafs_suffix, 'post3D', 'SABRE_*_paf_2_38.nii.gz')))
train_paf_2_39 = sorted(glob.glob(os.path.join(train_root_pafs, pafs_suffix, 'post3D', 'SABRE_*_paf_2_39.nii.gz')))
train_paf_2_40 = sorted(glob.glob(os.path.join(train_root_pafs, pafs_suffix, 'post3D', 'SABRE_*_paf_2_40.nii.gz')))
train_paf_2_41 = sorted(glob.glob(os.path.join(train_root_pafs, pafs_suffix, 'post3D', 'SABRE_*_paf_2_41.nii.gz')))
train_paf_2_42 = sorted(glob.glob(os.path.join(train_root_pafs, pafs_suffix, 'post3D', 'SABRE_*_paf_2_42.nii.gz')))
train_paf_2_43 = sorted(glob.glob(os.path.join(train_root_pafs, pafs_suffix, 'post3D', 'SABRE_*_paf_2_43.nii.gz')))
train_paf_2_44 = sorted(glob.glob(os.path.join(train_root_pafs, pafs_suffix, 'post3D', 'SABRE_*_paf_2_44.nii.gz')))
train_paf_2_45 = sorted(glob.glob(os.path.join(train_root_pafs, pafs_suffix, 'post3D', 'SABRE_*_paf_2_45.nii.gz')))
train_paf_2_46 = sorted(glob.glob(os.path.join(train_root_pafs, pafs_suffix, 'post3D', 'SABRE_*_paf_2_46.nii.gz')))
train_paf_2_47 = sorted(glob.glob(os.path.join(train_root_pafs, pafs_suffix, 'post3D', 'SABRE_*_paf_2_47.nii.gz')))

train_files = [{'image': image_name, 'label': label_name,\
                'heatmap28': heatmap_name28, 'heatmap29': heatmap_name29, 'heatmap30': heatmap_name30, \
                'heatmap31': heatmap_name31, 'heatmap32': heatmap_name32, 'heatmap33': heatmap_name33, 'heatmap34': heatmap_name34, 'heatmap35': heatmap_name35, \
                'heatmap36': heatmap_name36, 'heatmap37': heatmap_name37, 'heatmap38': heatmap_name38, 'heatmap39': heatmap_name39, 'heatmap40': heatmap_name40, 'heatmap41': heatmap_name41, \
                'paf_0_37': paf_name_0_37, 'paf_0_38': paf_name_0_38, 'paf_0_39': paf_name_0_39, 'paf_0_40': paf_name_0_40, \
                'paf_0_41': paf_name_0_41, 'paf_0_42': paf_name_0_42, 'paf_0_43': paf_name_0_43, 'paf_0_44': paf_name_0_44, 'paf_0_45': paf_name_0_45, \
                'paf_0_46': paf_name_0_46, 'paf_0_47': paf_name_0_47, \
                'paf_1_37': paf_name_1_37, 'paf_1_38': paf_name_1_38, 'paf_1_39': paf_name_1_39, 'paf_1_40': paf_name_1_40, \
                'paf_1_41': paf_name_1_41, 'paf_1_42': paf_name_1_42, 'paf_1_43': paf_name_1_43, 'paf_1_44': paf_name_1_44, 'paf_1_45': paf_name_1_45, \
                'paf_1_46': paf_name_1_46, 'paf_1_47': paf_name_1_47, \
                'paf_2_37': paf_name_2_37, 'paf_2_38': paf_name_2_38, 'paf_2_39': paf_name_2_39, 'paf_2_40': paf_name_2_40, \
                'paf_2_41': paf_name_2_41, 'paf_2_42': paf_name_2_42, 'paf_2_43': paf_name_2_43, 'paf_2_44': paf_name_2_44, 'paf_2_45': paf_name_2_45, 'paf_2_46': paf_name_2_46, 'paf_2_47': paf_name_2_47
                } \
                for image_name, label_name, \
                    heatmap_name28, heatmap_name29, heatmap_name30, \
                    heatmap_name31, heatmap_name32, heatmap_name33, heatmap_name34, heatmap_name35, heatmap_name36, heatmap_name37, heatmap_name38, heatmap_name39, heatmap_name40, heatmap_name41, \
                    paf_name_0_37, paf_name_0_38, paf_name_0_39, paf_name_0_40, \
                    paf_name_0_41, paf_name_0_42, paf_name_0_43, paf_name_0_44, paf_name_0_45, paf_name_0_46, paf_name_0_47, \
                    paf_name_1_37, paf_name_1_38, paf_name_1_39, paf_name_1_40, \
                    paf_name_1_41, paf_name_1_42, paf_name_1_43, paf_name_1_44, paf_name_1_45, paf_name_1_46, paf_name_1_47, \
                    paf_name_2_37, paf_name_2_38, paf_name_2_39, paf_name_2_40, \
                    paf_name_2_41, paf_name_2_42, paf_name_2_43, paf_name_2_44, paf_name_2_45, paf_name_2_46, paf_name_2_47 \
               in zip(train_images, train_labels, \
                           train_heatmap28, train_heatmap29, train_heatmap30, \
                           train_heatmap31, train_heatmap32, train_heatmap33, train_heatmap34, train_heatmap35, train_heatmap36, train_heatmap37, train_heatmap38, train_heatmap39, train_heatmap40, train_heatmap41, \
                           train_paf_0_37, train_paf_0_38, train_paf_0_39, train_paf_0_40, \
                           train_paf_0_41, train_paf_0_42, train_paf_0_43, train_paf_0_44, train_paf_0_45, train_paf_0_46, train_paf_0_47, \
                           train_paf_1_37, train_paf_1_38, train_paf_1_39, train_paf_1_40, \
                           train_paf_1_41, train_paf_1_42, train_paf_1_43, train_paf_1_44, train_paf_1_45, train_paf_1_46, train_paf_1_47, \
                           train_paf_2_37, train_paf_2_38, train_paf_2_39, train_paf_2_40, \
                           train_paf_2_41, train_paf_2_42, train_paf_2_43, train_paf_2_44, train_paf_2_45, train_paf_2_46, train_paf_2_47)]
train_files = train_files[:para.n_subj]

train_transforms = Compose([
    LoadNiftid(keys=['image', 'label',\
                     'heatmap28', 'heatmap29', 'heatmap30', \
                     'heatmap31', 'heatmap32', 'heatmap33', 'heatmap34', 'heatmap35', 'heatmap36', 'heatmap37', 'heatmap38', 'heatmap39', 'heatmap40', \
                     'heatmap41', \
                     'paf_0_37', 'paf_0_38', 'paf_0_39', 'paf_0_40', \
                     'paf_0_41', 'paf_0_42', 'paf_0_43', 'paf_0_44', 'paf_0_45', 'paf_0_46', 'paf_0_47', \
                     'paf_1_37', 'paf_1_38', 'paf_1_39', 'paf_1_40', \
                     'paf_1_41', 'paf_1_42', 'paf_1_43', 'paf_1_44', 'paf_1_45', 'paf_1_46', 'paf_1_47', \
                     'paf_2_37', 'paf_2_38', 'paf_2_39', 'paf_2_40', \
                     'paf_2_41', 'paf_2_42', 'paf_2_43', 'paf_2_44', 'paf_2_45', 'paf_2_46', 'paf_2_47']),
    AddChanneld(keys=['image', 'label',\
                     'heatmap28', 'heatmap29', 'heatmap30', \
                     'heatmap31', 'heatmap32', 'heatmap33', 'heatmap34', 'heatmap35', 'heatmap36', 'heatmap37', 'heatmap38', 'heatmap39', 'heatmap40', \
                     'heatmap41', \
                     'paf_0_37', 'paf_0_38', 'paf_0_39', 'paf_0_40', \
                     'paf_0_41', 'paf_0_42', 'paf_0_43', 'paf_0_44', 'paf_0_45', 'paf_0_46', 'paf_0_47', \
                     'paf_1_37', 'paf_1_38', 'paf_1_39', 'paf_1_40', \
                     'paf_1_41', 'paf_1_42', 'paf_1_43', 'paf_1_44', 'paf_1_45', 'paf_1_46', 'paf_1_47', \
                     'paf_2_37', 'paf_2_38', 'paf_2_39', 'paf_2_40', \
                     'paf_2_41', 'paf_2_42', 'paf_2_43', 'paf_2_44', 'paf_2_45', 'paf_2_46', 'paf_2_47']),
    Orientationd(keys=['image', 'label', \
                     'heatmap28', 'heatmap29', 'heatmap30', \
                     'heatmap31', 'heatmap32', 'heatmap33', 'heatmap34', 'heatmap35', 'heatmap36', 'heatmap37', 'heatmap38', 'heatmap39', 'heatmap40', \
                     'heatmap41', \
                     'paf_0_37', 'paf_0_38', 'paf_0_39', 'paf_0_40', \
                     'paf_0_41', 'paf_0_42', 'paf_0_43', 'paf_0_44', 'paf_0_45', 'paf_0_46', 'paf_0_47', \
                     'paf_1_37', 'paf_1_38', 'paf_1_39', 'paf_1_40', \
                     'paf_1_41', 'paf_1_42', 'paf_1_43', 'paf_1_44', 'paf_1_45', 'paf_1_46', 'paf_1_47', \
                     'paf_2_37', 'paf_2_38', 'paf_2_39', 'paf_2_40', \
                     'paf_2_41', 'paf_2_42', 'paf_2_43', 'paf_2_44', 'paf_2_45', 'paf_2_46', 'paf_2_47'], axcodes='RAS'),
    NormalizeIntensityd(keys=['image'], channel_wise=True),
    # RandGaussianNoised(keys=['image'], prob=0.75, mean=0.0, std=1.75),
    # RandRotate90d(keys=['image', 'heatmap', 'paf'], prob=0.5, spatial_axes=[0, 2]),
    CropForegroundd(keys=['image', 'label', \
                    'heatmap28', 'heatmap29', 'heatmap30', \
                     'heatmap31', 'heatmap32', 'heatmap33', 'heatmap34', 'heatmap35', 'heatmap36', 'heatmap37', 'heatmap38', 'heatmap39', 'heatmap40', \
                     'heatmap41', \
                     'paf_0_37', 'paf_0_38', 'paf_0_39', 'paf_0_40', \
                     'paf_0_41', 'paf_0_42', 'paf_0_43', 'paf_0_44', 'paf_0_45', 'paf_0_46', 'paf_0_47', \
                     'paf_1_37', 'paf_1_38', 'paf_1_39', 'paf_1_40', \
                     'paf_1_41', 'paf_1_42', 'paf_1_43', 'paf_1_44', 'paf_1_45', 'paf_1_46', 'paf_1_47', \
                     'paf_2_37', 'paf_2_38', 'paf_2_39', 'paf_2_40', \
                     'paf_2_41', 'paf_2_42', 'paf_2_43', 'paf_2_44', 'paf_2_45', 'paf_2_46', 'paf_2_47'], source_key='image'),
    RandCropByPosNegLabeld(keys=['image', 'label', \
                     'heatmap28', 'heatmap29', 'heatmap30', \
                     'heatmap31', 'heatmap32', 'heatmap33', 'heatmap34', 'heatmap35', 'heatmap36', 'heatmap37', 'heatmap38', 'heatmap39', 'heatmap40', \
                     'heatmap41', \
                     'paf_0_37', 'paf_0_38', 'paf_0_39', 'paf_0_40', \
                     'paf_0_41', 'paf_0_42', 'paf_0_43', 'paf_0_44', 'paf_0_45', 'paf_0_46', 'paf_0_47', \
                     'paf_1_37', 'paf_1_38', 'paf_1_39', 'paf_1_40', \
                     'paf_1_41', 'paf_1_42', 'paf_1_43', 'paf_1_44', 'paf_1_45', 'paf_1_46', 'paf_1_47', \
                     'paf_2_37', 'paf_2_38', 'paf_2_39', 'paf_2_40', \
                     'paf_2_41', 'paf_2_42', 'paf_2_43', 'paf_2_44', 'paf_2_45', 'paf_2_46', 'paf_2_47'], label_key='label', \
                     spatial_size=[para.patch_sz, para.patch_sz, para.patch_sz], pos=1, neg=1, num_samples=para.n_samp, image_threshold=1),
    ToTensord(keys=['image', 'label', \
                     'heatmap28', 'heatmap29', 'heatmap30', \
                     'heatmap31', 'heatmap32', 'heatmap33', 'heatmap34', 'heatmap35', 'heatmap36', 'heatmap37', 'heatmap38', 'heatmap39', 'heatmap40', \
                     'heatmap41', \
                     'paf_0_37', 'paf_0_38', 'paf_0_39', 'paf_0_40', \
                     'paf_0_41', 'paf_0_42', 'paf_0_43', 'paf_0_44', 'paf_0_45', 'paf_0_46', 'paf_0_47', \
                     'paf_1_37', 'paf_1_38', 'paf_1_39', 'paf_1_40', \
                     'paf_1_41', 'paf_1_42', 'paf_1_43', 'paf_1_44', 'paf_1_45', 'paf_1_46', 'paf_1_47', \
                     'paf_2_37', 'paf_2_38', 'paf_2_39', 'paf_2_40', \
                     'paf_2_41', 'paf_2_42', 'paf_2_43', 'paf_2_44', 'paf_2_45', 'paf_2_46', 'paf_2_47'])
])

## Define CacheDataset and DataLoader for training and validation
os.mkdir(tmp_catch)
train_ds = monai.data.PersistentDataset(data=train_files, transform=train_transforms, cache_dir=tmp_catch)
# train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)#, cache_dir=tmp_catch)
train_loader = DataLoader(train_ds, batch_size=para.n_batch, shuffle=True, num_workers=0, collate_fn=list_data_collate)

class Const_strcData:
    pass

opt = Const_strcData()

save_fold = pwd
opt.model = 'vgg'
#opt.models = 'fc'
opt.loadModel = 'none'
opt.criterionHm = 'l1'
opt.criterionPaf = 'l1'

model, criterion_hm, criterion_paf = create_model(opt)
model = model.to(device)
#criterion_hm = criterion_hm.cuda()
#criterion_paf = criterion_paf.cuda()

# Create optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)
#optimizer = create_optimizer(opt, models)

# # Other params
# n_epochs = opt.nEpoch
# to_train = opt.train
# drop_lr = opt.dropLR
# val_interval = opt.valInterval
# learn_rate = opt.LR
# visualize_out = opt.vizOut

TT = trainTT(pwd, save_fold, train_loader, para, model, optimizer, criterion_hm, criterion_paf, val_interval) #, models, optimizer_backbone)
TT.train()
print("BB-training completes")
