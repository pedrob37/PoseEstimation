import os
import nibabel as nib
from utils.utils import save_img
import re
import numpy as np


def read_file(filename):
    img = nib.load(filename)
    data = img.get_fdata()
    aff = img.affine
    return data, aff


regex = re.compile(r'\d+')

images_dir = '/storage/PoseEstimation-related/PoseEstimationGen2/Images'
cropped_images_dir = '/storage/PoseEstimation-related/PoseEstimationGen2/Cropped_Images'

PAFs_dir = '/storage/PoseEstimation-related/PoseEstimationGen2/PAFs'
mega_merged_PAFs_dir = '/storage/PoseEstimation-related/PoseEstimationGen2/Mega_Merged_PAFs'

images_list = os.listdir(images_dir)

# x_size, y_size, z_size = 193, 229, 193
# crop_x, crop_y, crop_z = 80, 80, 72
# x_lower_lim, y_lower_lim, z_lower_lim = ((x_size-1) - crop_x) // 2, ((y_size-1) - crop_y) // 2, ((z_size-1) - crop_z) // 2

for image in images_list:
    my_id = regex.findall(image)[0]
    total_paf = np.zeros((193, 229, 193, 3))
    for paf_num in range(1, 48):
        my_paf, aff = read_file(os.path.join(PAFs_dir, f"SABRE_{int(my_id)}_PAF_full_{paf_num}.nii.gz"))
        total_paf += my_paf
        # vol, aff = read_file(os.path.join(images_dir, image))
    save_img(total_paf, aff, os.path.join(mega_merged_PAFs_dir, f"SABRE_{int(my_id)}_PAF.nii.gz"), True)
