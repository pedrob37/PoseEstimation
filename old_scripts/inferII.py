import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch import nn
from monai.inferers import sliding_window_inference
from monai.utils import BlendMode
from monai.transforms import LoadNifti
#from create_infer_obj import create_infer_obj
import nibabel as nib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

class Const_strcData:
    pass
# pwd, save_fold, train_loader, para, models, optimizer, criterion_hm, criterion_paf

class inferII():
    def __init__(self, pwd, save_fold, val_loader, para, model_use, roi_size, val_files): #, para, models, optimizer_backbone):
        self.pwd = pwd
        self.save_fold = save_fold
        self.model_use = model_use
        self.val_loader = val_loader
        self.para = para
        self.roi_size = roi_size
        self.n_paf = para.n_paf
        self.val_files = val_files

    def infer(self):
        sw_batch_size = 1  # 4
        #overlap = float(0.55)
        overlap = float(0.05)# 0.55
        mode = BlendMode.GAUSSIAN

        #for patch_s in self.val_loader:
        for i, val_data in enumerate(self.val_loader):
            batch_image = val_data['image']
            batch_image = batch_image.to(device)
            input_cuda = batch_image.float().cuda()

            loader11 = LoadNifti(dtype=np.float32)
            img11, metadata11 = loader11(self.val_files[i]['image'])
            affine11 = metadata11.get('affine')
            del img11, metadata11
            del batch_image

            final_images = sliding_window_inference(input_cuda, self.roi_size, sw_batch_size, self.model_use.inference, mode=mode, overlap=overlap)
            print('final_images.shape=', final_images.shape)
            final_images0 = final_images[0,0, ...]  # [0,...].data.cpu().numpy()
            del final_images
            tmp11 = final_images0.data.cpu().numpy()
            tmp11 = np.float32(tmp11)
            # affine_co = np.eye(4)
            save_img = nib.Nifti1Image(tmp11, affine=affine11)
            #NIIname = 'xxxImage_sel_feat_xx' + '_.nii.gz'
            #NIIname = 'xxxHeatmap_xx' + '_.nii.gz'
            NIIname = 'xxxFinal_xx' + '_.nii.gz'
            nib.save(save_img, NIIname)
            del tmp11, save_img
            del input_cuda
            del final_images0
        print('chin')