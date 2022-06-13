import nibabel as nib
import numpy as np

xx = batch_image
for i in range(1):
    tmp11 = xx[0, i,...].data.cpu().numpy()
    tmp11 = np.float32(tmp11)
    affine_co = np.eye(4)
    save_img = nib.Nifti1Image(tmp11, affine=affine_co)
    NIIname = 'TT_batch_image' + str(i) + '_' + '.nii.gz'
    #NIIname = 'TMPinputs00_default' + '_' + '.nii.gz'
    nib.save(save_img, NIIname)
    del tmp11

del xx
xx = batch_heatmap
for i in range(13):
    tmp11 = xx[0, i,...]
    tmp11 = np.float32(tmp11)
    affine_co = np.eye(4)
    save_img = nib.Nifti1Image(tmp11, affine=affine_co)
    NIIname = 'TT_batch_heatmap_' + str(i) + '_' + '.nii.gz'
    #NIIname = 'TMPinputs00_default' + '_' + '.nii.gz'
    nib.save(save_img, NIIname)
    del tmp11

del xx
xx = batch_paf
for i in range(1):
    for j in range(3):
        tmp11 = xx[0, i, j, ...]
        tmp11 = np.float32(tmp11)
        affine_co = np.eye(4)
        save_img = nib.Nifti1Image(tmp11, affine=affine_co)
        NIIname = 'TT_batch_paf_' + str(j) + '_' + str(i) + '_' + '.nii.gz'
        #NIIname = 'TMPinputs00_default' + '_' + '.nii.gz'
        nib.save(save_img, NIIname)
        del tmp11

##############################

# del xx
# xx = batch_heatmap
# for i in range(13):
#     tmp11 = xx[0, i,...].data.cpu().numpy()
#     tmp11 = np.float32(tmp11)
#     affine_co = np.eye(4)
#     save_img = nib.Nifti1Image(tmp11, affine=affine_co)
#     NIIname = 'TT_batch_heatmap_out' + str(i) + '_' + '.nii.gz'
#     #NIIname = 'TMPinputs00_default' + '_' + '.nii.gz'
#     nib.save(save_img, NIIname)
#     del tmp11

xx = batch_image
for i in range(1):
    tmp11 = xx[0, i,...].data.cpu().numpy()
    tmp11 = np.float32(tmp11)
    affine_co = np.eye(4)
    save_img = nib.Nifti1Image(tmp11, affine=affine_co)
    NIIname = 'TT_batch_image' + str(i) + '_' + '.nii.gz'
    #NIIname = 'TMPinputs00_default' + '_' + '.nii.gz'
    nib.save(save_img, NIIname)
    del tmp11


del xx
xx = feat
for i in range(20):
    tmp11 = xx[0, i, ...].data.cpu().numpy()
    tmp11 = np.float32(tmp11)
    affine_co = np.eye(4)
    save_img = nib.Nifti1Image(tmp11, affine=affine_co)
    NIIname = 'TT_feat_outputs' + '_i=' + str(i) + '_' + '.nii.gz'
    nib.save(save_img, NIIname)
    del tmp11
del feat

for j in range(7):
    del xx
    xx = heatmap_outputs[j]
    for i in range(13):
        tmp11 = xx[0, i,...].data.cpu().numpy()
        tmp11 = np.float32(tmp11)
        affine_co = np.eye(4)
        save_img = nib.Nifti1Image(tmp11, affine=affine_co)
        NIIname = 'TT_heatmap_outputs' + str(j) + '_i=' + str(i) + '_' + '.nii.gz'
        #NIIname = 'TMPinputs00_default' + '_' + '.nii.gz'
        nib.save(save_img, NIIname)
        del tmp11

del xx
xx = heatmap_t_cuda
for i in range(13):
    tmp11 = xx[0, i,...].data.cpu().numpy()
    tmp11 = np.float32(tmp11)
    affine_co = np.eye(4)
    save_img = nib.Nifti1Image(tmp11, affine=affine_co)
    NIIname = 'TT_heatmap_t_cuda' + '_i=' + str(i) + '_' + '.nii.gz'
    #NIIname = 'TMPinputs00_default' + '_' + '.nii.gz'
    nib.save(save_img, NIIname)
    del tmp11