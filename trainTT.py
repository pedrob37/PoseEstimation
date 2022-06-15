import torch
import numpy as np
import os
from utils.utils import val_saver

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class trainTT:
    def __init__(self, models_dir, figures_dir, writer, train_loader, val_loader, opt, model, optimizer, scheduler,
                 criterion_hm, criterion_paf, selected_heatmaps, selected_pafs):
        self.models_dir = models_dir
        self.figures_dir = figures_dir
        self.writer = writer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.opt = opt
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.HM_torch_loss = criterion_hm
        self.PAF_torch_loss = criterion_paf

        self.selected_heatmaps = selected_heatmaps
        self.selected_PAFs = selected_pafs

    def train(self):
        writerBB = self.writer
        for epoch in range(self.opt.num_epochs_backbone):
            print(f'Epoch: {epoch}')

            # Set model to train mode
            self.model.train()
            for iteration, patch_s in enumerate(self.train_loader):
                print(f"Fractional memory usage: {torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated():.3f}")
                batch_image = patch_s['image'].to(device)

                # Concatenate all heatmaps and all PAFs into one structure
                batch_heatmap = torch.concat(tuple(patch_s[f"heatmap_{x}"] for x in self.selected_heatmaps), dim=1).to(
                    device)
                batch_paf = torch.stack(tuple(patch_s[f"PAF_{x}"] for x in self.selected_PAFs), dim=5).to(device)

                # Outputs
                heatmap_outputs, paf_outputs = self.model(batch_image)

                # Losses
                loss_HM = 0
                loss_PAF = 0

                # Looping through stage samples: Loss is aggregation of these
                for ii in range(len(heatmap_outputs)):
                    heatmap_out = heatmap_outputs[ii]
                    paf_out = paf_outputs[ii]
                    paf_out = torch.reshape(paf_out, [paf_out.shape[0],
                                                      3,
                                                      paf_out.shape[2],
                                                      paf_out.shape[3],
                                                      paf_out.shape[4],
                                                      int(paf_out.shape[1] / 3)
                                                      ])

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
                del loss_HM, loss_PAF, loss
                del heatmap_outputs, paf_outputs
                del batch_image, batch_heatmap, batch_paf, patch_s

            ## Validation
            if (epoch + 1) % self.opt.validation_interval == 0:
                # Set model to eval mode
                self.model.eval()
                # Save Model
                name = os.path.join(self.models_dir, f'PoseEst_model_{epoch}.pt')
                torch.save(self.model.state_dict(), name)
                print(f'Saving Model, epoch {epoch}...')
                with torch.no_grad():
                    agg_heatmap_loss = []
                    agg_paf_loss = []
                    agg_loss = []
                    for val_sample in self.val_loader:
                        val_batch_image = val_sample['image'].to(device)

                        # Concatenate all heatmaps and all PAFs into one structure
                        val_batch_heatmap = torch.concat(
                            tuple(val_sample[f"heatmap_{x}"] for x in self.selected_heatmaps), dim=1).to(device)
                        val_batch_paf = torch.concat(
                            tuple(val_sample[f"PAF_{x}"] for x in self.selected_PAFs), dim=1).to(device)

                        # Affine: For output-saving purposes
                        val_affine = val_sample['image_meta_dict']['affine'][0, ...]

                        # Outputs
                        val_heatmap_outputs, val_paf_outputs = self.model(val_batch_image)

                        # Losses
                        val_loss_HM = 0
                        val_loss_PAF = 0

                        # Looping through stage samples: Loss is aggregation of these
                        for ii in range(len(val_heatmap_outputs)):
                            val_heatmap_out = val_heatmap_outputs[ii]
                            val_paf_out = val_paf_outputs[ii]
                            val_paf_out = torch.reshape(val_paf_out, [val_paf_out.shape[0],
                                                                      3,
                                                                      val_paf_out.shape[2],
                                                                      val_paf_out.shape[3],
                                                                      val_paf_out.shape[4],
                                                                      int(val_paf_out.shape[1] / 3)
                                                                      ])

                            val_loss_HM += self.HM_torch_loss(val_batch_heatmap, val_heatmap_out)
                            val_loss_PAF += self.PAF_torch_loss(val_batch_paf, val_paf_out)

                            # Save outputs once every epoch
                            if not agg_loss:  # i.e.: Empty list, so must be first validation step
                                # Save output heatmap and paf
                                val_saver(val_heatmap_out.max(axis=1)[0].squeeze().cpu().detach().numpy(),
                                          val_affine, self.figures_dir, "Val_heatmap", epoch, ii)
                                val_saver(val_paf_out.sum(axis=-1).squeeze().permute(3, 1, 2, 0).cpu().detach().numpy(),
                                          val_affine, self.figures_dir, "Val_paf", epoch, ii)
                                if ii == 0:
                                    # Save OGs
                                    val_saver(val_batch_heatmap.max(axis=1)[0].squeeze().cpu().detach().numpy(),
                                              val_affine, self.figures_dir, "GT_heatmap", epoch, None)
                                    val_saver(val_batch_paf.sum(axis=-1).squeeze().cpu().permute(3, 1, 2, 0).detach().numpy(),
                                              val_affine, self.figures_dir, "GT_paf", epoch, None)
                            del val_heatmap_out, val_paf_out

                        # Calculate validation loss
                        val_loss = val_loss_HM + val_loss_PAF

                        # Aggregate
                        agg_loss.append(val_loss.item())
                        agg_heatmap_loss.append(val_loss_HM.item())
                        agg_paf_loss.append(val_loss_PAF.item())

                        # Variable deletion for next loop
                        del val_loss_HM, val_loss_PAF, val_loss
                        del val_heatmap_outputs, val_paf_outputs
                        del val_batch_image, val_batch_heatmap, val_batch_paf

                    # Write
                    print(f'Val BackBone: {np.mean(agg_loss):.3f}')
                    writerBB.add_scalar("Loss/Val_Backbone_Overall", np.mean(agg_loss), epoch)
                    writerBB.add_scalar("Loss/Val_Backbone_HeatMap", np.mean(agg_heatmap_loss), epoch)
                    writerBB.add_scalar("Loss/Val_Backbone_PAF", np.mean(agg_paf_loss), epoch)
                    del agg_loss, agg_heatmap_loss, agg_paf_loss
            self.scheduler.step()
