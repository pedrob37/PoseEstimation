import torch
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class trainTT:
    def __init__(self, models_dir, writer, train_loader, val_loader, opt, model, optimizer, criterion_hm,
                 criterion_paf, selected_heatmaps, selected_pafs):
        self.models_dir = models_dir
        self.writer = writer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.opt = opt
        self.model = model
        self.optimizer = optimizer

        self.HM_torch_loss = criterion_hm
        self.PAF_torch_loss = criterion_paf

        self.selected_heatmaps = selected_heatmaps
        self.selected_PAFs = selected_pafs

    def train(self):
        writerBB = self.writer
        for epoch in range(self.opt.num_epochs_backbone):
            print(f'Epoch: {epoch}')

            self.model.train()
            if epoch % self.opt.validation_interval == 1:
                name = os.path.join(self.models_dir, f'PoseEst_model_{epoch}.pt')
                torch.save(self.model.state_dict(), name)
                print(f'Saving Model, epoch {epoch}...')

            for iteration, patch_s in enumerate(self.train_loader):
                print(f"Fractional memory usage: {torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated():.3f}")
                batch_image = patch_s['image'].to(device)
                # batch_label = patch_s['label']

                # Concatenate all heatmaps and all PAFs into one structure
                batch_heatmap = torch.concat(tuple(patch_s[f"heatmap_{x}"] for x in self.selected_heatmaps), dim=1).to(
                    device)
                batch_paf = torch.concat(tuple(patch_s[f"PAF_{x}"] for x in self.selected_PAFs), dim=1).to(device)

                # Check
                # xx = batch_heatmap
                # for i in range(batch_heatmap.shape[0]):
                #     for j in range(batch_heatmap.shape[1]):
                #         tmp11 = np.squeeze(xx[i, j, ...].data.cpu().numpy())
                #         print('i,j, sum', str(i), str(j), tmp11.sum())
                #         del tmp11
                # del xx

                # Outputs
                heatmap_outputs, paf_outputs = self.model(batch_image)

                # Losses
                loss_HM = 0
                loss_PAF = 0

                # Looping through batches? Seems odd
                for ii in range(len(heatmap_outputs)):
                    heatmap_out = heatmap_outputs[ii]
                    paf_out = paf_outputs[ii]
                    paf_out = torch.reshape(paf_out, [paf_out.shape[0],
                                                      int(paf_out.shape[1] / 3),
                                                      paf_out.shape[2],
                                                      paf_out.shape[3],
                                                      paf_out.shape[4],
                                                      3
                                                      ])

                    # Isn't first variable size BS x ..., and second is 1 x ...?
                    # print(batch_heatmap.shape, heatmap_out.shape)
                    # print(batch_paf.shape, paf_out.shape)
                    loss_HM += self.HM_torch_loss(batch_heatmap, heatmap_out)
                    loss_PAF += self.PAF_torch_loss(batch_paf, paf_out)
                    del heatmap_out, paf_out

                loss = loss_HM + loss_PAF
                print(f'BackBone: {loss.item():.3f}')

                # Epoch is very small, so just log once per epoch
                if iteration == np.random.randint(0, len(self.train_loader)):
                    writerBB.add_scalar("Loss/Backbone_Overall", loss.item(), epoch)
                    writerBB.add_scalar("Loss/Backbone_HeatMap", loss_HM.item(), epoch)
                    writerBB.add_scalar("Loss/Backbone_PAF", loss_PAF.item(), epoch)

                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Variable deletion for next loop
                del loss_HM, loss_PAF
                del heatmap_outputs, paf_outputs
                del batch_image, batch_heatmap, batch_paf

            ## Validation
            if (epoch + 1) % self.opt.validation_interval == 0:
                with torch.no_grad():
                    agg_heatmap_loss = []
                    agg_paf_loss = []
                    agg_loss = []
                    for val_sample in self.val_loader:
                        val_batch_image = val_sample['image'].to(device)
                        # batch_label = patch_s['label']

                        # Concatenate all heatmaps and all PAFs into one structure
                        val_batch_heatmap = torch.concat(
                            tuple(val_sample[f"heatmap_{x}"] for x in self.selected_heatmaps), dim=1).to(device)
                        val_batch_paf = torch.concat(
                            tuple(val_sample[f"PAF_{x}"] for x in self.selected_PAFs), dim=1).to(device)

                        # Check
                        # xx = batch_heatmap
                        # for i in range(batch_heatmap.shape[0]):
                        #     for j in range(batch_heatmap.shape[1]):
                        #         tmp11 = np.squeeze(xx[i, j, ...].data.cpu().numpy())
                        #         print('i,j, sum', str(i), str(j), tmp11.sum())
                        #         del tmp11
                        # del xx

                        # Outputs
                        val_heatmap_outputs, val_paf_outputs = self.model(val_batch_image)

                        # Losses
                        val_loss_HM = 0
                        val_loss_PAF = 0

                        # Looping through batches? Seems odd
                        for ii in range(len(val_heatmap_outputs)):
                            val_heatmap_out = val_heatmap_outputs[ii]
                            val_paf_out = val_paf_outputs[ii]
                            val_paf_out = torch.reshape(val_paf_out, [val_paf_out.shape[0],
                                                                      int(val_paf_out.shape[1] / 3),
                                                                      3,
                                                                      val_paf_out.shape[2],
                                                                      val_paf_out.shape[3],
                                                                      val_paf_out.shape[4]])

                            # Isn't first variable size BS x ..., and second is 1 x ...?
                            val_loss_HM += self.HM_torch_loss(val_batch_heatmap, val_heatmap_out)
                            val_loss_PAF += self.PAF_torch_loss(val_batch_paf, val_paf_out)
                            del val_heatmap_out, val_paf_out

                        val_loss = val_loss_HM + val_loss_PAF

                        # Aggregate
                        agg_loss.append(val_loss.item())
                        agg_heatmap_loss.append(val_loss_HM.item())
                        agg_paf_loss.append(val_loss_PAF.item())

                        # Variable deletion for next loop
                        del val_loss_HM, val_loss_PAF
                        del val_heatmap_outputs, val_paf_outputs
                        del val_batch_image, val_batch_heatmap, val_batch_paf

                    # Write
                    print(f'Val BackBone: {np.mean(agg_loss)}')
                    writerBB.add_scalar("Loss/Val_Backbone_Overall", np.mean(agg_loss), epoch)
                    writerBB.add_scalar("Loss/Val_Backbone_HeatMap", np.mean(agg_heatmap_loss), epoch)
                    writerBB.add_scalar("Loss/Val_Backbone_PAF", np.mean(agg_paf_loss), epoch)
