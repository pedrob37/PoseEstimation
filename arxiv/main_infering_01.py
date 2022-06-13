import torch
from torch import optim
import os
#from fire import Fire
import glob
from torch.utils.data import DataLoader
import monai
from monai.transforms import Compose, LoadNiftid, AddChanneld, CropForegroundd, RandCropByPosNegLabeld, Orientationd, ToTensord, NormalizeIntensityd
from monai.data import list_data_collate #, sliding_window_inference

from inferII import inferII
from models.td_model_provider import create_model, create_optimizer
from modelOBJ import modelOBJ

monai.config.print_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

class Const_strcData:
    pass

pwd = os.getcwd()
val_root_images = '/media/chayanin/Storage/chin/data2020/SABRE/SABRE_te'
val_root_heatmaps = '/media/chayanin/Storage/chin/data2020/SABRE/SABRE_te'
val_root_pafs = '/media/chayanin/Storage/chin/data2020/SABRE/SABRE_te'
#train_root_images = '/nfs/home/ctangwiriyasakul/DGXwork/datasets/SABRE/SABRE_te'
#train_root_heatmaps = '/nfs/home/ctangwiriyasakul/DGXwork/datasets/SABRE/SABRE_te'
#train_root_pafs = '/nfs/home/ctangwiriyasakul/DGXwork/datasets/SABRE/SABRE_te'

images_suffix = 'SABRE_*'
heatmaps_suffix = 'SABRE_*'
pafs_suffix = 'SABRE_*'

n_subj = 1#4
batch = 1
#patch_sz = 16
#n_samp = 5
#n_of_epBB = 501
#tmp_catch = '/Users/ct20/Documents/MyPyCharm/2020/2020_version_06/catch/20210921_catch_training'
tmp_catch = '/home/chayanin/PycharmProjects/2020_v04_vrienv002/catch/20210921_catch_inferring'
#tmp_catch = '/nfs/home/ctangwiriyasakul/DGXwork/datasets/catch/20210921_catch_inferring'
para = Const_strcData()
para.n_subj = n_subj
para.n_batch = batch
#para.patch_sz = patch_sz
#para.n_samp = n_samp
#para.n_of_epBB = n_of_epBB

# Loading data
val_images = sorted(glob.glob(os.path.join(val_root_images, images_suffix, 'post3D', 'SABRE_*_im.nii.gz')))
val_labels = sorted(glob.glob(os.path.join(val_root_heatmaps, heatmaps_suffix, 'post3D', 'SABRE_*_allpointmap.nii.gz')))


val_files = [{'image': image_name, 'label': label_name} \
                for image_name, label_name, \
                    in zip(val_images, val_labels)]
val_files = val_files[:para.n_subj]

train_transforms = Compose([
    LoadNiftid(keys=['image', 'label']),
    AddChanneld(keys=['image', 'label']),
    Orientationd(keys=['image', 'label'], axcodes='RAS'),
    NormalizeIntensityd(keys=['image'], channel_wise=True),
    ToTensord(keys=['image', 'label'])
])

## Define CacheDataset and DataLoader for training and validation
os.mkdir(tmp_catch)
val_ds = monai.data.PersistentDataset(data=val_files, transform=train_transforms, cache_dir=tmp_catch)
# train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)#, cache_dir=tmp_catch)
val_loader = DataLoader(val_ds, batch_size=para.n_batch, shuffle=True, num_workers=0, collate_fn=list_data_collate)

class Const_strcData:
    pass

opt = Const_strcData()

#previous_model = '/home/chayanin/PycharmProjects/2020_v04_vrienv002/2021/202109_Sep/week02/20210916_3DposeEST/PostEst_model_401.pt'
previous_model = pwd + '/PostEst_model_901.pt'

save_fold = pwd
opt.model = 'vgg'
opt.loadModel = 'none'
opt.criterionHm = 'l1'
opt.criterionPaf = 'l1'
opt.loadModel = previous_model

model, criterion_hm, criterion_paf = create_model(opt)
model = model.to(device)
model.eval()

# opt_pre = Const_strcData()
# opt_pre.models = 'vgg'
# opt_pre.loadModel = 'none'
# opt_pre.criterionHm = 'l1'
# opt_pre.criterionPaf = 'l1'
# opt_pre.loadModel = pwd + '/PostEst_model_101.pt'
# model_pre, criterion_hm_pre, criterion_paf_pre = create_model(opt_pre)
# model_pre = model_pre.to(device)
# model_pre.eval()

roi_size = 50
no_paf = 11#47

model_use = modelOBJ(model)

II = inferII(pwd, save_fold, val_loader, para, model_use, roi_size, no_paf, val_files) # models, optimizer_backbone)
II.infer()
print("II-infering completes")