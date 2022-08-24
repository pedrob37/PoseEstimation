import torch.nn as nn
from models.td_helper import init, make_standard_block, make_2019_block
import torch


class HMModel(nn.Module):
    def __init__(self, backend, backend_outp_feats, n_joints, n_stages=7):
        super(HMModel, self).__init__()
        assert (n_stages > 0)
        self.backend = backend
        stages = [Stage(backend_outp_feats, n_joints, True)]
        for _ in range(n_stages - 1):
            stages.append(Stage(backend_outp_feats, n_joints, False))
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        img_feats = self.backend(x)
        cur_feats = img_feats
        heatmap_outs = []
        for i, stage in enumerate(self.stages):
            heatmap_out = stage(cur_feats)
            heatmap_outs.append(heatmap_out)

            cur_feats = torch.cat([img_feats, heatmap_out], 1)

        return heatmap_outs


class PAFModel2019(nn.Module):
    def __init__(self, backend, backend_outp_feats, n_joints, n_paf, n_stages_total, n_stages_paf,
                 num_conv_blocks=5, densenet_approach=True):
        super(PAFModel2019, self).__init__()
        assert (n_stages_total > 0)
        assert (n_stages_paf > 0)
        assert (n_stages_total > n_stages_paf)
        self.n_stages_paf = n_stages_paf
        self.n_stages_heatmap = n_stages_total - n_stages_paf
        self.n_stages_total = n_stages_total
        self.backend = backend
        # First PAF stage
        stages = [Stage2019(backend_outp_feats, n_joints, n_paf, "first_paf", num_conv_blocks, densenet_approach)]
        # Loop through remaining PAF stages
        for _ in range(n_stages_paf-1):
            stages.append(Stage2019(backend_outp_feats, n_joints, n_paf, "paf", num_conv_blocks, densenet_approach))
        # First Heatmap stage
        stages.append(Stage2019(backend_outp_feats, n_joints, n_paf, "first_heatmap", num_conv_blocks, densenet_approach))
        for _ in range(self.n_stages_heatmap-1):
            stages.append(Stage2019(backend_outp_feats, n_joints, n_paf, "heatmap", num_conv_blocks, densenet_approach))
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        img_feats = self.backend(x)
        cur_feats = img_feats
        heatmap_outs = []
        paf_outs = []
        for step, stage in enumerate(self.stages):
            if step < self.n_stages_paf:
                paf_out = stage(cur_feats)
                paf_outs.append(paf_out)

                # Concatenate features
                cur_feats = torch.cat([img_feats, paf_out], 1)

            elif step < self.n_stages_total:
                heatmap_out = stage(cur_feats)
                heatmap_outs.append(heatmap_out)

                # Concatenate features
                cur_feats = torch.cat([img_feats, heatmap_out], 1)

        return heatmap_outs, paf_outs


class Stage(nn.Module):
    def __init__(self, backend_outp_feats, n_joints, stage1):
        super(Stage, self).__init__()
        inp_feats = backend_outp_feats
        if stage1:
            #print('stage1.inp_feats', inp_feats)
            self.block1 = make_paf_block_stage1(inp_feats, n_joints)
            # self.block2 = make_paf_block_stage1(inp_feats, n_paf)
            #self.block2 = make_paf_block_stage1(inp_feats, n_paf*n_vector_field)
        else:
            inp_feats = backend_outp_feats + n_joints
            #print('other stage.inp_feats', inp_feats)
            self.block1 = make_paf_block_stage2(inp_feats, n_joints)
            # self.block2 = make_paf_block_stage2(inp_feats, n_paf)
            #self.block2 = make_paf_block_stage2(inp_feats, n_paf*n_vector_field)
        #self.m_chin = nn.Conv3d(n_paf, n_vector_field, kernel_size=3) # to manually somehow change from 1 Ch to 2 Ch (vector field),
        init(self.block1)
        # init(self.block2)
        #self.n_vector_field = n_vector_field
        #print('self.n_vector_field', self.n_vector_field)
        #init(self.m_chin)

    def forward(self, x):
        #m_chin = nn.Conv3d(1, 2, kernel_size=3) # to manually somehow change from 1 Ch to 2 Ch (vector field),
        y1 = self.block1(x)
        #print('y1.shape', y1.shape)
        # y2 = self.block2(x)
        #print('y2.shape', y2.shape)
        #y2_chin = torch.reshape(y2, [int(y2.shape[0]), int(self.n_vector_field), int(y2.shape[1]/self.n_vector_field), int(y2.shape[2]), int(y2.shape[3]), int(y2.shape[4])])
        #y2_chin = self.m_chin(y2)
        #print('y2_chin.shape', y2_chin.shape)
        return y1
        #return y1, y2_chin


class Stage2019(nn.Module):
    def __init__(self, backend_outp_feats, n_joints, n_paf, current_stage, num_conv_blocks, densenet_approach):
        super(Stage2019, self).__init__()
        inp_feats = backend_outp_feats
        self.num_conv_blocks = num_conv_blocks
        if current_stage == "first_paf":
            self.conv_list = make_paf_block_2019(inp_feats, n_paf, num_conv_blocks, densenet_approach)
        elif current_stage == "first_heatmap":
            inp_feats = backend_outp_feats + n_paf
            self.conv_list = make_paf_block_2019(inp_feats, n_joints, num_conv_blocks, densenet_approach)
        elif current_stage == "paf":
            inp_feats = backend_outp_feats + n_paf
            self.conv_list = make_paf_block_2019(inp_feats, n_paf, num_conv_blocks, densenet_approach)
        elif current_stage == "heatmap":
            inp_feats = backend_outp_feats + n_joints
            self.conv_list = make_paf_block_2019(inp_feats, n_joints, num_conv_blocks, densenet_approach)
        else:
            raise NotImplementedError
        init(self.conv_list)

    def forward(self, x):
        for i in range(self.num_conv_blocks):
            y1 = self.conv_list[i*3](x)
            y2 = self.conv_list[(i*3)+1](y1)
            y3 = self.conv_list[(i*3)+2](y2)
            x = torch.cat([y1, y2, y3], 1)
        # Final layers
        x = self.conv_list[-2](x)
        x = self.conv_list[-1](x)
        return x


def make_paf_block_stage1(inp_feats, output_feats):
    layers = [make_standard_block(inp_feats, 128, 3),
              make_standard_block(128, 128, 3),
              make_standard_block(128, 128, 3),
              make_standard_block(128, 512, 1, 1, 0)]
    layers += [nn.Conv3d(512, output_feats, 1, 1, 0)]
    return nn.Sequential(*layers)


def make_paf_block_stage2(inp_feats, output_feats):
    layers = [make_standard_block(inp_feats, 128, 7, 1, 3),
              make_standard_block(128, 128, 7, 1, 3),
              make_standard_block(128, 128, 7, 1, 3),
              make_standard_block(128, 128, 7, 1, 3),
              make_standard_block(128, 128, 7, 1, 3),
              make_standard_block(128, 128, 1, 1, 0)]
    layers += [nn.Conv3d(128, output_feats, 1, 1, 0)]
    return nn.Sequential(*layers)


def make_paf_block_2019(inp_feats, output_feats, num_conv_blocks, densenet_approach):
    if not densenet_approach:
        # ModuleList usage and loop creation: https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html
        # convs = nn.ModuleList([make_standard_block(inp_feats, 128, 3, 1, 1)])
        # First set of convs take in 128 channels
        convs = nn.ModuleList([make_standard_block(inp_feats, 128, 3, 1, 1),
                               make_standard_block(128, 128, 3, 1, 1),
                               make_standard_block(128, 128, 3, 1, 1)])
        # Subsequent trios take in 128 x 3 = 384 channels
        trio_blocks = [make_standard_block(128 * 3, 128, 3, 1, 1),
                       make_standard_block(128, 128, 3, 1, 1),
                       make_standard_block(128, 128, 3, 1, 1)] * (num_conv_blocks-1)
        convs += nn.ModuleList(trio_blocks)
        # Define final layers
        final_layers = [make_standard_block(128 * 3, 128, 1, 1, 0)]  # 3 == Num consecutive convs in block
        # Why 512 here originally?
        final_layers += [nn.Conv3d(128, output_feats, 1, 1, 0)]
        convs += final_layers
    else:
        # ModuleList usage and loop creation: https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html
        # First set of convs take in 128 channels
        convs = nn.ModuleList([make_standard_block(inp_feats, 64, 3, 1, 1),
                               make_standard_block(64, 32, 3, 1, 1),
                               make_standard_block(32, 32, 3, 1, 1)])
        # Subsequent trios take in 128 channels, outputs from each conv are concatenated: 64 + 32 + 32 = 128 channels
        trio_blocks = [make_standard_block(128, 64, 3, 1, 1),
                       make_standard_block(64, 32, 3, 1, 1),
                       make_standard_block(32, 32, 3, 1, 1)] * (num_conv_blocks-1)
        convs += nn.ModuleList(trio_blocks)
        # Define final layers
        final_layers = [make_standard_block(128, 128, 1, 1, 0)]
        # Why 512 here originally?
        final_layers += [nn.Conv3d(128, output_feats, 1, 1, 0)]
        convs += final_layers
    return convs
