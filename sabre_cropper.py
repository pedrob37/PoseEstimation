import os
import nibabel as nib
from utils.utils import save_img


def read_file(filename):
    img = nib.load(filename)
    data = img.get_fdata()
    aff = img.affine
    return data, aff


images_dir = '/storage/PoseEstimation-related/PoseEstimationGen2/Images'
cropped_images_dir = '/storage/PoseEstimation-related/PoseEstimationGen2/Cropped_Images'

images_list = os.listdir(images_dir)

x_size, y_size, z_size = 193, 229, 193
crop_x, crop_y, crop_z = 80, 80, 72
x_lower_lim, y_lower_lim, z_lower_lim = ((x_size-1) - crop_x) // 2, ((y_size-1) - crop_y) // 2, ((z_size-1) - crop_z) // 2

for image in images_list:
    vol, aff = read_file(os.path.join(images_dir, image))
    save_img(vol[x_lower_lim:x_lower_lim+crop_x,
             y_lower_lim:y_lower_lim+crop_y,
             0:crop_z], aff, os.path.join(cropped_images_dir, image), True)
