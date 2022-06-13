import torch
import numpy as np
from torch import nn

#from .td_vgg import VGG
#from .td_fc import FC
#from .td_paf_model import PAFModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Const_strcData:
    pass

#class modelOBJ(nn.Module):
class modelOBJ():
    # def __init__(self,
    #              backend,
    #              backend_feats,
    #              pretrained=None):
    #     super(modelOBJ, self).__init__()
    #     self.backend = backend
    #     self.backend_feats = backend_feats
    #     self.init_weights(pretrained=pretrained)
    #
    # def init_weights(self, pretrained=None):
    #     if pretrained is not None:
    #         self.load_state_dict(torch.load(pretrained['backbone'], map_location=torch.device(device)))
    #
    # def inference_rpn(self, img):
    #     models = PAFModel(self.backend, self.backend_feats, n_joints=41, n_paf=47 * 2, n_stages=7, n_vector_field=2)

    def __init__(self,
                 model):
        super(modelOBJ, self).__init__()
        self.model = model
        #self.init_weights(pretrained=pretrained)

    # def init_weights(self, pretrained=None):
    #     if pretrained is not None:
    #         self.load_state_dict(torch.load(pretrained, map_location=torch.device(device)))

    def inference_backend(self, img, sel_feat=1): # not ready
        feat = self.model.backend(img)
        out_feat = feat[0, sel_feat, ...]
        out_feat = out_feat.cpu().detach()
        out_feat = np.expand_dims(out_feat, axis=0)
        out_feat = np.expand_dims(out_feat, axis=0) # need to expand dimension to have the following form [1,1,X,Y,Z]
        return torch.tensor(out_feat)

    def inference_heatmap(self, img, sel_feat=0):
        print("Inference Heatmap")
        print('sel_feat=', sel_feat)
        heatmap_infer, paf_infer = self.model(img)
        del paf_infer  # just to clear memory
        heatmap_pick_outputs = heatmap_infer[-1] # pick the result from the last stage.
        del heatmap_infer #2021.10.21
        heatmap_infer = heatmap_pick_outputs[0, sel_feat, ...] # the first 0 mean we are taking the first subject (actually for infering we only feed one subject at a time)
        heatmap_infer = heatmap_infer.cpu().detach()
        #heatmap_infer = heatmap_infer.detach() # [x,y,z]
        heatmap_infer = np.expand_dims(heatmap_infer, axis=0)
        heatmap_infer = np.expand_dims(heatmap_infer, axis=0) # need to expand dimension to have the following form [1,1,X,Y,Z]
        #return torch.tensor(out_paf_infer)
        return torch.tensor(heatmap_infer)

    def inference(self, img, sel_feat=1):
        print("Inference")
        heatmap_infer, paf_infer = self.model(img)
        del heatmap_infer  # just to clear memory
        pick_paf_outputs = paf_infer[-1] # pick the result from the last stage.
        del paf_infer
        pick_paf_outputs = torch.reshape(pick_paf_outputs, [int(pick_paf_outputs.shape[0]), int(3), \
                                                            int(pick_paf_outputs.shape[1] / 3),
                                                            int(pick_paf_outputs.shape[2]),
                                                            int(pick_paf_outputs.shape[3]), \
                                                            int(pick_paf_outputs.shape[4])])
        pick_paf_outputs = pick_paf_outputs.transpose(2, 1)
        print('pick_paf_outputs.shape', pick_paf_outputs.shape)
        pick_paf_outputs = torch.squeeze(pick_paf_outputs).cpu().detach().numpy()
        # paf_infer : [47, 2, image.shape[-3], image.shape[-2], image.shape[-1]]
        selected_pick_paf_infer = pick_paf_outputs[:, sel_feat, ...]
        out_paf_infer = np.zeros((img.shape[-3], img.shape[-2], img.shape[-1]))
        for par_idx in range(selected_pick_paf_infer.shape[0]):  # should be 47 # looping over each PAF
            tmp = selected_pick_paf_infer[par_idx, ...]
            print('par_idx=', par_idx,'tmp.max', tmp.max(), 'tmp.min', tmp.min())
            out_paf_infer = out_paf_infer + tmp
            del tmp

        out_paf_infer = np.expand_dims(out_paf_infer, axis=0)
        out_paf_infer = np.expand_dims(out_paf_infer, axis=0) # need to expand dimension to have the following form [1,1,X,Y,Z]

        return torch.tensor(out_paf_infer)

