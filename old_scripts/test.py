import nibabel as nib
import numpy as np
#xx = score_map
#xx = map_ellipse
#xx = chosen_gt
xx= batch_image#[0,0,...]#[0,...].data.cpu().numpy()

for i in range(8):
    tmp11 = np.squeeze(xx[i,...].data.cpu().numpy())
    tmp11 = np.float32(tmp11)
    affine_co = np.eye(4)
    save_img = nib.Nifti1Image(tmp11, affine=affine_co)
    NIIname = 'batch_image_' + str(i) + '_' + '.nii.gz'
    #NIIname = 'heatmap_' + str(i) + '_' + '.nii.gz'
    #NIIname = 'TMPinputs00_default' + '_' + '.nii.gz'
    nib.save(save_img, NIIname)
    del tmp11

del xx
xx= batch_label
for i in range(8):
    tmp11 = np.squeeze(xx[i,...].data.cpu().numpy())
    tmp11 = np.float32(tmp11)
    affine_co = np.eye(4)
    save_img = nib.Nifti1Image(tmp11, affine=affine_co)
    NIIname = 'batch_label_' + str(i) + '_' + '.nii.gz'
    #NIIname = 'heatmap_' + str(i) + '_' + '.nii.gz'
    #NIIname = 'TMPinputs00_default' + '_' + '.nii.gz'
    nib.save(save_img, NIIname)
    del tmp11

del xx
xx= batch_heatmap#[0,0,...]#[0,...].data.cpu().numpy()
for i in range(8):
    for j in range(13):
        tmp11 = np.squeeze(xx[i, j, ...].data.cpu().numpy())
        tmp11 = np.float32(tmp11)
        affine_co = np.eye(4)
        save_img = nib.Nifti1Image(tmp11, affine=affine_co)
        NIIname = 'heatmap_' + str(i) + '_' + str(j) + '.nii.gz'
        nib.save(save_img, NIIname)
        del tmp11

del xx
xx = batch_paf
sel_f = 0
for i in range(8):
    for j in range(11):
        tmp11 = np.squeeze(xx[i, j, sel_f, ...].data.cpu().numpy())
        tmp11 = np.float32(tmp11)
        affine_co = np.eye(4)
        save_img = nib.Nifti1Image(tmp11, affine=affine_co)
        NIIname = 'batch_paf_' + str(i) + '_' + str(j) + 'sel_f=' + str(sel_f) + '.nii.gz'
        nib.save(save_img, NIIname)
        del tmp11


del xx
xx= batch_heatmap#[0,0,...]#[0,...].data.cpu().numpy()
for i in range(8):
    for j in range(13):
        tmp11 = np.squeeze(xx[i, j, ...].data.cpu().numpy())
        print('i,j, sum', str(i), str(j), tmp11.sum())
        del tmp11


#############
import nibabel as nib
import numpy as np

labell = np.squeeze(label[0, ...])
affine_co = np.eye(4)
save_img = nib.Nifti1Image(labell, affine=affine_co)
NIIname = 'labell2.nii.gz'
nib.save(save_img, NIIname)

imagee = np.squeeze(image[0, ...])
affine_co = np.eye(4)
save_img = nib.Nifti1Image(imagee, affine=affine_co)
NIIname = 'imagee2.nii.gz'
nib.save(save_img, NIIname)

labell2 = np.squeeze(label[0, ...])
affine_co = np.eye(4)
save_img = nib.Nifti1Image(labell2, affine=affine_co)
NIIname = 'labell2.nii.gz'