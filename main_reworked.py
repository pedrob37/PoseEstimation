from torch import optim
from options.train_options import BaseOptions
import glob
from torch.utils.data import DataLoader
import monai
from monai.transforms import Compose, LoadImaged, AddChanneld, CropForegroundd, RandCropByPosNegLabeld, Orientationd, \
    ToTensord, NormalizeIntensityd, Spacingd, RandAffined, RandGaussianNoised, AsChannelFirstd, SpatialCropd
from monai.data import list_data_collate
from utils.utils import *
import random
from torch.utils.tensorboard import SummaryWriter

from trainTT import trainTT, testTT
from models.td_model_provider import create_model

if __name__ == '__main__':
    print(monai.__version__)
    monai.config.print_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----  Loading the init options -----
    opt = BaseOptions().parse()

    # Patch size handling
    if len(opt.patch_size) == 1:
        opt.patch_size = opt.patch_size * 3
    elif len(opt.patch_size) == 3:
        # 3D, so this is fine
        pass
    else:
        raise OSError("Patch size should either be one, or three integers")

    ## Relevant job directories
    base_dir = opt.base_dir
    CACHE_DIR = f"{base_dir}/Outputs-Pose-Estimation/Cache/{opt.job_name}"
    FIG_DIR = f"{base_dir}/Outputs-Pose-Estimation/Figures/{opt.job_name}"
    LOG_DIR = f'{base_dir}/Outputs-Pose-Estimation/Logs/{opt.job_name}'
    SLIM_LOG_DIR = f'{base_dir}/Outputs-Pose-Estimation/Slim_Logs/{opt.job_name}'
    MODELS_DIR = f"{base_dir}/Outputs-Pose-Estimation/Models/{opt.job_name}"

    # Create directories
    create_path(CACHE_DIR)
    create_path(FIG_DIR)
    create_path(LOG_DIR)
    create_path(SLIM_LOG_DIR)
    create_path(MODELS_DIR)

    # Data directories
    debug = False
    if not debug:
        images_dir = os.path.join(opt.data_path, 'Images')
        labels_dir = os.path.join(opt.data_path, 'Labels')
        heatmaps_dir = os.path.join(opt.data_path, 'Heatmaps')
        PAFs_dir = os.path.join(opt.data_path, 'PAFs')

        # Won't necessarily handle ALL branches: Can be more selective
        if opt.branch_selection == "full":
            selected_heatmaps = list(range(0, 41))
            selected_PAFs = list(range(0, 48))
        elif opt.branch_selection == "main":
            selected_heatmaps = list(range(28, 41))
            selected_PAFs = list(range(37, 48))

        # Loading data: Always take all images and corresponding labels
        full_images = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
        full_labels = sorted(glob.glob(os.path.join(labels_dir, "*.nii.gz")))

        # For heatmaps and PAFs loop through only relevant IDs
        full_heatmaps = {}
        full_PAFs = {}

        import re

        regex = re.compile(r'\d+')

        for relevant_heatmap_ID in selected_heatmaps:
            full_heatmaps[f"heatmaps_{relevant_heatmap_ID}"] = sorted(glob.glob(os.path.join(heatmaps_dir,
                                                                                             f"*heatmap_{relevant_heatmap_ID}.nii.gz")))

        for relevant_PAF_ID in selected_PAFs:
            full_PAFs[f"PAFs_{relevant_PAF_ID}"] = sorted(glob.glob(os.path.join(PAFs_dir,
                                                                                 f"*PAF_full_{relevant_PAF_ID}.nii.gz")))

        # Splits: Random NON-GLOBAL shuffle:
        # https://stackoverflow.com/questions/19306976/python-shuffling-with-a-parameter-to-get-the-same-result
        random.Random(1).shuffle(full_images)
        random.Random(1).shuffle(full_labels)

        train_images, val_images, inf_images = create_folds(full_images, train_split=0.8, val_split=0.1)
        train_labels, val_labels, inf_labels = create_folds(full_labels, train_split=0.8, val_split=0.1)

        # For heatmaps and PAFs loop through only relevant IDs
        from collections import OrderedDict

        train_heatmaps, val_heatmaps, inf_heatmaps = OrderedDict(), OrderedDict(), OrderedDict()  # {}, {}, {}
        train_PAFs, val_PAFs, inf_PAFs = OrderedDict(), OrderedDict(), OrderedDict()  # {}, {}, {}

        # Train
        for subject in train_images:
            # Isolate subject ID
            base_sub = regex.findall(os.path.basename(subject))[0]
            sub_relevant_heatmaps = []
            sub_relevant_PAFs = []

            # Include only relevant heatmaps/ PAFs
            for relevant_heatmap_ID in selected_heatmaps:
                sub_relevant_heatmaps.extend(glob.glob(os.path.join(heatmaps_dir,
                                                                    f"*{base_sub}*heatmap_{relevant_heatmap_ID}.nii.gz")))
            for relevant_PAF_ID in selected_PAFs:
                sub_relevant_PAFs.extend(glob.glob(os.path.join(PAFs_dir,
                                                                f"*{base_sub}*PAF_full_{relevant_PAF_ID}.nii.gz")))
            # Assign to relevant keys
            train_heatmaps[f"SABRE_{base_sub}"] = sub_relevant_heatmaps
            train_PAFs[f"SABRE_{base_sub}"] = sub_relevant_PAFs

        # Validation
        for subject in val_images:
            # Isolate subject ID
            base_sub = regex.findall(os.path.basename(subject))[0]
            sub_relevant_heatmaps = []
            sub_relevant_PAFs = []

            # Include only relevant heatmaps/ PAFs
            for relevant_heatmap_ID in selected_heatmaps:
                sub_relevant_heatmaps.extend(glob.glob(os.path.join(heatmaps_dir,
                                                                    f"*{base_sub}*heatmap_{relevant_heatmap_ID}.nii.gz")))
            for relevant_PAF_ID in selected_PAFs:
                sub_relevant_PAFs.extend(glob.glob(os.path.join(PAFs_dir,
                                                                f"*{base_sub}*PAF_full_{relevant_PAF_ID}.nii.gz")))
            # Assign to relevant keys
            val_heatmaps[f"SABRE_{base_sub}"] = sub_relevant_heatmaps
            val_PAFs[f"SABRE_{base_sub}"] = sub_relevant_PAFs

        # Inference
        for subject in inf_images:
            # Isolate subject ID
            base_sub = regex.findall(os.path.basename(subject))[0]
            sub_relevant_heatmaps = []
            sub_relevant_PAFs = []
            # Include only relevant heatmaps/ PAFs
            for relevant_heatmap_ID in selected_heatmaps:
                sub_relevant_heatmaps.extend(glob.glob(os.path.join(heatmaps_dir,
                                                                    f"*{base_sub}*heatmap_{relevant_heatmap_ID}.nii.gz")))
            for relevant_PAF_ID in selected_PAFs:
                sub_relevant_PAFs.extend(glob.glob(os.path.join(PAFs_dir,
                                                                f"*{base_sub}*PAF_full_{relevant_PAF_ID}.nii.gz")))
            # Assign to relevant keys
            inf_heatmaps[f"SABRE_{base_sub}"] = sub_relevant_heatmaps
            inf_PAFs[f"SABRE_{base_sub}"] = sub_relevant_PAFs

        # Create data dicts
        train_data_dict = [{'image': image_name, 'label': label_name, 'heatmap': heatmap_name, 'PAF': PAF_name} for
                           image_name, label_name, heatmap_name, PAF_name
                           in zip(train_images, train_labels, train_heatmaps.values(), train_PAFs.values())]

        val_data_dict = [{'image': image_name, 'label': label_name, 'heatmap': heatmap_name, 'PAF': PAF_name} for
                         image_name, label_name, heatmap_name, PAF_name
                         in zip(val_images, val_labels, val_heatmaps.values(), val_PAFs.values())]

        inf_data_dict = [{'image': image_name, 'label': label_name, 'heatmap': heatmap_name, 'PAF': PAF_name} for
                         image_name, label_name, heatmap_name, PAF_name
                         in zip(inf_images, inf_labels, inf_heatmaps.values(), inf_PAFs.values())]

        # Assign heatmaps and PAFs to relevant dictionaries: Train, Val, Inf
        for train_sub in train_data_dict:
            list_to_dict_reorganiser(train_sub)

        for val_sub in val_data_dict:
            list_to_dict_reorganiser(val_sub)

        for inf_sub in inf_data_dict:
            list_to_dict_reorganiser(inf_sub)

        print(f"Length of inference images, labels, heatmaps, PAFs: {len(inf_images)}, "
              f"{len(inf_labels)}, "
              f"{len(inf_heatmaps)}, "
              f"{len(inf_PAFs)}")

        # Print sizes
        logging_interval = int(len(train_images) / 2)
        print(f'The length of the training is {len(train_images)}')
        print(f'The length of the validation is {len(val_images)}')
        print(f'The length of the inference is {len(inf_images)}')

        # Shuffle!
        do_shuffling = True

        # MONAI transforms
        relevant_keys = ['image', 'label'] + [f"heatmap_{hm}" for hm in selected_heatmaps] + [f"PAF_{paf}" for paf in
                                                                                              selected_PAFs]
        PAF_less_keys = ['image', 'label'] + [f"heatmap_{hm}" for hm in selected_heatmaps]
        PAF_keys = [f"PAF_{paf}" for paf in selected_PAFs]
        num_heatmaps = len([f"heatmap_{hm}" for hm in selected_heatmaps])
        num_PAFs = len([f"PAF_{paf}" for paf in selected_PAFs])
        nearest_list = ["nearest"] * (num_heatmaps + num_PAFs)
        zeros_list = ["zeros"] * (num_heatmaps + num_PAFs)
        # Transform lists
        train_transform_list = [LoadImaged(keys=relevant_keys),
                                AddChanneld(keys=PAF_less_keys),
                                AsChannelFirstd(keys=PAF_keys),
                                # Temp
                                # CoordConvd(keys=['image'], spatial_channels=(1, 2, 3)),
                                # Don't normalise? Dealing with binaries
                                # NormalizeIntensityd(keys=['image', 'label'], channel_wise=True),
                                ]

        # Cropped size
        if opt.weighted_sampling:
            cropped_roi_size = opt.patch_size
            train_transform_list.append(SpatialCropd(keys=['image', 'label'],  # , 'coords'],
                                                     roi_size=cropped_roi_size,
                                                     roi_center=(96, 114, 0)))  # Leaning right 56, 74, 0
            # CoordConvd(keys=['image'], spatial_channels=(1, 2, 3))]
            if opt.augmentation_level == "none":
                # Don't add any augmentations
                pass
            elif opt.augmentation_level == 'light':
                train_transform_list.extend([RandGaussianNoised(keys=['image'], prob=0.5, mean=0.0, std=0.25),
                                             RandAffined(keys=relevant_keys,
                                                         # spatial_size=(201, 201, 71),
                                                         scale_range=(0.1, 0.1, 0.1),
                                                         rotate_range=(0.25, 0.25, 0.25),
                                                         translate_range=(20, 20, 20),
                                                         mode=["nearest", "nearest"] + nearest_list,
                                                         as_tensor_output=False, prob=0.5,
                                                         padding_mode=['zeros', 'zeros'] + zeros_list)])
        else:
            if opt.augmentation_level == "none":
                # Don't add any augmentations
                pass
            elif opt.augmentation_level == 'light':
                train_transform_list.extend([RandGaussianNoised(keys=['image'], prob=0.5, mean=0.0, std=0.25),
                                             RandAffined(keys=relevant_keys,
                                                         spatial_size=(201, 201, 71),
                                                         scale_range=(0.1, 0.1, 0.1),
                                                         rotate_range=(0.25, 0.25, 0.25),
                                                         translate_range=(20, 20, 20),
                                                         mode=["bilinear", "nearest"] + nearest_list,
                                                         as_tensor_output=False, prob=0.5,
                                                         padding_mode=['zeros', 'zeros'] + zeros_list)])
        # Extend with missing transforms
        if not opt.weighted_sampling:
            train_transform_list.extend([RandCropByPosNegLabeld(keys=relevant_keys,
                                                                label_key='label', image_key='image',
                                                                spatial_size=opt.patch_size, pos=100, neg=0,
                                                                num_samples=opt.num_samples),
                                         Spacingd(keys=relevant_keys, pixdim=(1, 1, 1, 1)),
                                         ToTensord(keys=relevant_keys)])
        elif opt.weighted_sampling:
            train_transform_list.extend([Spacingd(keys=relevant_keys, pixdim=(1, 1, 1, 1)),
                                         ToTensord(keys=relevant_keys)])

        # Compose
        train_transforms = Compose(train_transform_list)

        val_transforms = Compose([
            LoadImaged(keys=relevant_keys),
            AddChanneld(keys=PAF_less_keys),
            AsChannelFirstd(keys=PAF_keys),
            # Orientationd(keys=relevant_keys, axcodes='RAS'),
            NormalizeIntensityd(keys=['image'], channel_wise=True),
            # RandGaussianNoised(keys=['image'], prob=0.75, mean=0.0, std=1.75),
            # RandRotate90d(keys=['image', 'heatmap', 'paf'], prob=0.5, spatial_axes=[0, 2]),
            # CropForegroundd(keys=relevant_keys,
            #                 source_key='image'),
            # RandCropByPosNegLabeld(keys=relevant_keys,
            #                        label_key='label', image_key='image',
            #                        spatial_size=[opt.patch_size, opt.patch_size, opt.patch_size], pos=100, neg=0,
            #                        num_samples=opt.num_samples),
            Spacingd(keys=relevant_keys, pixdim=(1, 1, 1, 1)),
            ToTensord(keys=relevant_keys)
        ])

        if opt.phase == "inference":
            inf_transforms = Compose([
                LoadImaged(keys=relevant_keys),
                AddChanneld(keys=relevant_keys),
                # Orientationd(keys=relevant_keys, axcodes='RAS'),
                NormalizeIntensityd(keys=['image'], channel_wise=True),
                # RandGaussianNoised(keys=['image'], prob=0.75, mean=0.0, std=1.75),
                # RandRotate90d(keys=['image', 'heatmap', 'paf'], prob=0.5, spatial_axes=[0, 2]),
                # CropForegroundd(keys=relevant_keys,
                #                 source_key='image'),
                # RandCropByPosNegLabeld(keys=relevant_keys,
                #                        label_key='label', image_key='image',
                #                        spatial_size=[opt.patch_size, opt.patch_size, opt.patch_size], pos=100, neg=0,
                #                        num_samples=opt.num_samples),
                Spacingd(keys=relevant_keys, pixdim=(1, 1, 1, 1)),
                ToTensord(keys=relevant_keys)
            ])

            inf_ds = monai.data.Dataset(data=inf_data_dict,
                                        transform=inf_transforms,
                                       )

            inf_loader = DataLoader(inf_ds,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=opt.workers,
                                    collate_fn=list_data_collate
                                    )

        ## Define CacheDataset and DataLoader for training and validation
        if opt.debug:
            train_ds = monai.data.Dataset(data=train_data_dict,
                                          transform=train_transforms,
                                                    )

            val_ds = monai.data.Dataset(data=val_data_dict,
                                        transform=val_transforms,
                                                  )
        else:
            train_ds = monai.data.PersistentDataset(data=train_data_dict,
                                                    transform=train_transforms,
                                                    cache_dir=CACHE_DIR
                                                    )

            val_ds = monai.data.PersistentDataset(data=val_data_dict,
                                                  transform=val_transforms,
                                                  cache_dir=CACHE_DIR
                                                  )

        train_loader = DataLoader(train_ds,
                                  batch_size=opt.batch_size,
                                  shuffle=do_shuffling,
                                  num_workers=opt.workers,
                                  collate_fn=list_data_collate
                                  )

        # Validation
        val_loader = DataLoader(val_ds,
                                batch_size=opt.batch_size,
                                shuffle=do_shuffling,
                                num_workers=opt.workers,
                                collate_fn=list_data_collate
                                )

    # Model loading: If parameter is not None, then it has to, currently, be a full path to a model!
    model, criterion_heatmap, criterion_paf = create_model(opt, models_dir=MODELS_DIR, model_device=device)
    model = model.to(device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, opt.scheduler_gamma)

    # Create logger
    writer = SummaryWriter(os.path.join(LOG_DIR, 'runBackBone'))

    if opt.phase == "train":
        TT = trainTT(MODELS_DIR, FIG_DIR, writer, train_loader, val_loader, opt, model, optimizer, scheduler,
                     criterion_heatmap, criterion_paf,
                     selected_heatmaps, selected_PAFs,
                     opt.debug)
        TT.train()
        print("BB-training completed")
    else:
        TT = testTT(MODELS_DIR, FIG_DIR, writer, inf_loader, opt, model,
                    selected_heatmaps, selected_PAFs,
                    opt.debug)
        TT.test()
        print("BB-inference completed")

