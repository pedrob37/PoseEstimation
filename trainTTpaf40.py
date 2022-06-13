import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

class Const_strcData:
    pass
# pwd, save_fold, train_loader, para, models, optimizer, criterion_hm, criterion_paf
class trainTTpaf40():
    def __init__(self, pwd, save_fold, train_loader, para, model, optimizer, criterion_hm, criterion_paf, val_interval): #, para, models, optimizer_backbone):
        self.pwd = pwd
        self.save_fold = save_fold
        self.train_loader = train_loader
        self.para = para
        self.model = model
        self.optimizer = optimizer
        self.criterion_hm = criterion_hm
        self.criterion_paf = criterion_paf
        self.patch_size = para.patch_sz

        self.HM_torch_loss = nn.L1Loss()
        # HM_torch_loss = nn.MSELoss()
        self.PAF_torch_loss = nn.L1Loss()

        self.val_interval = val_interval

        self.n_joints = para.n_joints
        self.n_paf = para.n_paf
        self.n_stages = para.n_stages
        self.n_vector_field = para.n_vector_field

        # PAF_torch_loss = nn.MSELoss()
        # self.suffix = suffix

        #self.models = models
        #self.optimizer_backbone = optimizer_backbone

    def train(self):
        writerBB = SummaryWriter(self.save_fold + '/runBB')
        for epoch in range(self.para.n_of_epBB):
            print('epoch=', epoch)

            self.model.train()
            if epoch % self.val_interval == 1:
                #name = self.pwd + self.suffix + '/RPN_ODMix_%d.pt' % epoch
                name = self.save_fold + '/PostEst_model_%d.pt' % epoch
                torch.save(self.model.state_dict(), name)
                print('we saved the models')

            for patch_s in self.train_loader:
                batch_image = patch_s['image']
                batch_label = patch_s['label']
                batch_heatmap = np.zeros((batch_image.shape[0], self.n_joints, batch_image.shape[-3], batch_image.shape[-2], batch_image.shape[-1]))
                batch_paf = np.zeros((batch_image.shape[0], self.n_paf, self.n_vector_field, batch_image.shape[-3], batch_image.shape[-2], batch_image.shape[-1]))
                for ii in range(batch_heatmap.shape[0]):
                    batch_heatmap[ii, 0, ...] = patch_s['heatmap30'][ii, 0, ...]
                    batch_heatmap[ii, 1, ...] = patch_s['heatmap38'][ii, 0, ...]
                    #batch_heatmap[ii, 13, ...] = patch_s['heatmap41'][ii, 0, ...]
                    batch_paf[ii, 0, 0, ...] = patch_s['paf_0_40'][ii, 0, ...]
                    batch_paf[ii, 0, 1, ...] = patch_s['paf_1_40'][ii, 0, ...]
                    batch_paf[ii, 0, 2, ...] = patch_s['paf_2_40'][ii, 0, ...]

                batch_image = batch_image.to(device)
                batch_heatmap = torch.tensor(batch_heatmap).to(device)
                batch_paf = torch.tensor(batch_paf).to(device)

                # xx = batch_heatmap  # [0,0,...]#[0,...].data.cpu().numpy()
                # for i in range(batch_heatmap.shape[0]):
                #     for j in range(batch_heatmap.shape[1]):
                #         tmp11 = np.squeeze(xx[i, j, ...].data.cpu().numpy())
                #         print('i,j, sum', str(i), str(j), tmp11.sum())
                #         del tmp11
                # del xx
                #
                # del batch_label

                input_cuda = batch_image.float().cuda()
                heatmap_t_cuda = batch_heatmap.float().cuda()
                paf_t_cuda = batch_paf.float().cuda()

                heatmap_outputs, paf_outputs = self.model(input_cuda) # patch_size=batch_image.shape[-1])
                #feat = self.models.backend(input_cuda)
                #del feat

                # HM_torch_loss = nn.L1Loss()
                # HM_torch_loss = nn.MSELoss()
                # PAF_torch_loss = nn.L1Loss()
                # PAF_torch_loss = nn.MSELoss()

                loss_HM = 0
                loss_PAF = 0
                for ii in range(len(heatmap_outputs)):
                    heatmap_out = heatmap_outputs[ii]
                    paf_out = paf_outputs[ii]
                    paf_out = torch.reshape(paf_out, [int(paf_out.shape[0]), int(3), int(paf_out.shape[1] / 3), int(paf_out.shape[2]), int(paf_out.shape[3]), int(paf_out.shape[4])])
                    paf_out = paf_out.transpose(2, 1)
                    loss_HM += self.HM_torch_loss(heatmap_t_cuda, heatmap_out)
                    loss_PAF += self.PAF_torch_loss(paf_t_cuda, paf_out)
                    del heatmap_out, paf_out

                loss = loss_HM + loss_PAF
                print('BB', loss)
                writerBB.add_scalar("Loss/Backbone", loss, epoch)
                # total_loss.requires_grad = True
                self.model.zero_grad()
                # gc.collect()
                loss.backward()
                self.optimizer.step()


                ## Validation
                if (epoch + 1) % self.val_interval == 0:
                    print('sort of')

                ## end of validaton

                del loss_HM, loss_PAF
                #del HM_torch_loss, PAF_torch_loss
                del heatmap_outputs, paf_outputs
                del batch_image, batch_heatmap, batch_paf
                del input_cuda, heatmap_t_cuda, paf_t_cuda

                print('chin')


