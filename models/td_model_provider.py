from models.td_vgg import VGG
from models.td_fc import FC
from models.td_paf_model import PAFModel, PAFModel2019
from models.custom_resnet import resnet18, resnet34, resnet50
import torch.nn as nn
import torch
import os


def generate_model(opt, partial_load=False):
    if opt.resnet_size == 18:
        model = resnet18(n_input_channels=1)
    elif opt.resnet_size == 34:
        model = resnet34(n_input_channels=1)
    elif opt.resnet_size == 50:
        model = resnet50(n_input_channels=1)

    net_dict = model.state_dict()

    print(f'There are {len(net_dict)} keys in the model dict!')

    # Load pretrained model
    if opt.pretrained_backbone:
        print(f'Loading pretrained model from {opt.pretrain_path}, size {opt.resnet_size}')
        pretrain = torch.load(os.path.join(opt.pretrain_path, f"resnet_{opt.resnet_size}_23dataset.pth"))

        if partial_load:
            conv_count = 0
            layer_iter = 1
            while conv_count < 11:
                from itertools import islice
                from collections import OrderedDict
                sliced = islice(pretrain['state_dict'].items(), layer_iter)
                sliced_dict = OrderedDict(sliced)
                conv_count = 0
                for item in sliced_dict.keys():
                    if "conv" in item:
                        conv_count += 1
                # print(conv_count)
                # print(list(sliced_dict.keys())[-1])
                layer_iter += 1

            # Get rid of final conv
            conv_count = 0
            # Subtract two because while loop increments even on completion
            sliced = islice(pretrain['state_dict'].items(), layer_iter - 2)
            sliced_dict = OrderedDict(sliced)
            for item in sliced_dict.keys():
                if "conv" in item:
                    conv_count += 1
            print(f"There are {conv_count}")
            # print(list(sliced_dict.keys())[-6:])

            print(f"Loading up {layer_iter} parameters from original network, with {conv_count} convolutions!")
            pretrain_dict_og = {k: v for k, v in sliced_dict.items()}
        else:
            pretrain_dict_og = {k: v for k, v in pretrain['state_dict'].items()}
        pretrain_dict_og = {k[7:]: v for k, v in pretrain_dict_og.items()}
        pretrain_dict = {k: v for k, v in pretrain_dict_og.items() if k in net_dict.keys()}
        missing_dict = {k: v for k, v in net_dict.items() if k not in pretrain_dict_og.keys()}

        # If missing too many keys then raise an error
        print(f'There are {len(pretrain_dict)} keys in the pretrain dict: {len(pretrain_dict)/len(net_dict)*100:.0f}%')
        if len(pretrain_dict) < 20:
            print(missing_dict.keys())
            raise NotImplementedError

        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)

        if partial_load:
            return model, sliced_dict
        else:
            return model, pretrain_dict

    return model, model.parameters()


verbose_check = False
if verbose_check:
    class ConstData:
        pass


    opt = ConstData

    opt.resnet_size = 50
    opt.pretrained_backbone = True
    opt.pretrain_path = "/home/pedro/PoseEstimation/MedicalNet/pretrain"
    test_model, test_pretrained_model = generate_model(opt, partial_load=False)

    # Check if loading properly
    print((test_model.state_dict()["layer1.0.conv1.weight"].cuda() == test_pretrained_model["layer1.0.conv1.weight"].cuda()).sum()/torch.numel(test_model.state_dict()["layer1.0.conv1.weight"]))

    # Try a forward pass
    output = test_model(torch.zeros((1, 1, 80, 80, 80)))
    print(output.shape)


def parse_criterion(criterion):
    if criterion == 'l1':
        return nn.L1Loss()
    elif criterion == 'mse':
        return nn.MSELoss()
    else:
        raise ValueError('Criterion ' + criterion + ' not supported')


def create_model(opt, models_dir=None, model_device=None):
    if opt.model == 'vgg':
        backend = VGG()
        backend_feats = 128
        print("Using VGG!")
    elif opt.model == 'fc':
        backend = FC()
        backend_feats = 128
        print("Using FC!")
    elif opt.model == 'resnet':
        # Partial load is False because networks have been defined to be smaller in advance
        backend, _ = generate_model(opt, partial_load=False)
        backend_feats = 128
        print(f"Using ResNet-{opt.resnet_size}!")
    else:
        raise ValueError('Model ' + opt.model + ' not available.')
    if opt.model_version == "old":
        model = PAFModel(backend=backend,
                         backend_outp_feats=backend_feats,
                         n_joints=opt.num_joints,
                         n_paf=opt.num_pafs*opt.num_vector_fields,
                         n_stages=opt.num_stages
                         )
    elif opt.model_version == "new":
        model = PAFModel2019(backend=backend,
                             backend_outp_feats=backend_feats,
                             n_joints=opt.num_joints,
                             n_paf=opt.num_pafs*opt.num_vector_fields,  # Times ND for N spatial dimensions
                             n_stages_total=opt.num_stages*2,  # Multiply by two because PAF and HM stages are separate
                             n_stages_paf=opt.num_stages_paf,
                             num_conv_blocks=5,
                             densenet_approach=opt.densenet_approach
                             )

    # Find relevant model
    import glob
    model_files = glob.glob(os.path.join(models_dir, '*.pth'))
    if len(model_files) > 0:
        for some_model_file in model_files:
            print(some_model_file)
        sorted_model_files = sorted(model_files, key=os.path.getmtime)
        # Allows inference to be run on nth latest file!
        latest_model_file = sorted_model_files[-1]
        checkpoint = torch.load(latest_model_file, map_location=model_device)

        model.load_state_dict(checkpoint)
        print('Loaded models from ' + models_dir)

    criterion_hm = parse_criterion(opt.criterion_heatmap)
    criterion_paf = parse_criterion(opt.criterion_paf)
    return model, criterion_hm, criterion_paf


def create_optimizer(opt, model):
    return torch.optim.Adam(model.parameters(), opt.LR)
