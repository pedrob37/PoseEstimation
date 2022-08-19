import torch
import numpy as np
import os
from utils.utils import val_saver

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class trainTT:
    def __init__(self, models_dir, figures_dir, writer, train_loader, val_loader, opt, model, optimizer, scheduler,
                 criterion_hm, criterion_paf, selected_heatmaps, selected_pafs,
                 current_epoch=None, current_iteration=None, debug=False):
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

        self.current_epoch = current_epoch
        self.current_iteration = current_iteration

        self.debug = debug

    def train(self):
        writerBB = self.writer
        for epoch in range(self.current_epoch, self.opt.num_epochs_backbone):
            print(f'Epoch {epoch}:')

            # Set model to train mode
            self.model.train()
            for iteration, patch_s in enumerate(self.train_loader):
                # print(f"Fractional memory usage: {torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated():.3f}")
                batch_image = patch_s['image'].to(device)

                # Concatenate all heatmaps and all PAFs into one structure
                batch_heatmap = torch.concat(tuple(patch_s[f"heatmap_{x}"] for x in self.selected_heatmaps), dim=1).to(
                    device)
                # Dimension five because channels are reserved for x, y, z
                batch_paf = torch.stack(tuple(patch_s[f"PAF_{x}"] for x in self.selected_PAFs), dim=5).to(device)

                # Affine: For output-saving purposes
                affine = patch_s['image_meta_dict']['affine'][0, ...]

                # Debug saving
                if self.debug:
                    print("Debug saving!")
                    batch_label = patch_s['label'].to(device)
                    batch_coords = patch_s['coords'].to(device)
                    val_saver(batch_image.squeeze().cpu().detach().numpy(), affine, self.figures_dir, "Image_test",
                              epoch, iteration)
                    val_saver(batch_label.squeeze().cpu().detach().numpy(), affine, self.figures_dir, "Label_test",
                              epoch, iteration)
                    val_saver(batch_coords.squeeze().permute(1, 2, 3, 0).cpu().detach().numpy(), affine,
                              self.figures_dir, "Coords_test",
                              epoch, iteration)

                # Outputs
                heatmap_outputs, paf_outputs = self.model(batch_image)

                # Losses
                loss_HM = 0
                loss_PAF = 0

                # Looping through stage samples: Loss is aggregation of these
                for ii in range(len(heatmap_outputs)):
                    heatmap_out = heatmap_outputs[ii]
                    paf_out = torch.zeros([paf_outputs[ii].shape[0],
                                           3,
                                           paf_outputs[ii].shape[2],
                                           paf_outputs[ii].shape[3],
                                           paf_outputs[ii].shape[4],
                                           int(paf_outputs[ii].shape[1] / 3)
                                           ]).to(device)
                    # paf_out = torch.reshape(paf_out, [paf_out.shape[0],
                    #                                   3,
                    #                                   paf_out.shape[2],
                    #                                   paf_out.shape[3],
                    #                                   paf_out.shape[4],
                    #                                   int(paf_out.shape[1] / 3)
                    #                                   ])
                    paf_out[:, 0, ...] = paf_outputs[ii][:, ::3, ...].permute(0, 2, 3, 4, 1)
                    paf_out[:, 1, ...] = paf_outputs[ii][:, 1::3, ...].permute(0, 2, 3, 4, 1)
                    paf_out[:, 2, ...] = paf_outputs[ii][:, 2::3, ...].permute(0, 2, 3, 4, 1)

                    loss_HM += self.HM_torch_loss(batch_heatmap, heatmap_out)
                    loss_PAF += self.PAF_torch_loss(batch_paf, paf_out)

                    # Save outputs once every epoch
                    if (iteration % 100 == 0) and (ii == (len(heatmap_outputs) - 1)):
                        # Save output heatmap and paf
                        val_saver(heatmap_out.max(axis=1)[0].squeeze().cpu().detach().numpy(),
                                  affine, self.figures_dir, "Out_heatmap", epoch, iteration)
                        val_saver(heatmap_out.squeeze().cpu().permute(1, 2, 3, 0).detach().numpy(),
                                  affine, self.figures_dir, "Out_heatmap_separate", epoch, iteration)
                        val_saver(paf_out.sum(axis=-1).squeeze().permute(1, 2, 3, 0).cpu().detach().numpy(),
                                  affine, self.figures_dir, "Out_paf", epoch, iteration)

                        # Save OGs
                        val_saver(batch_heatmap.max(axis=1)[0].squeeze().cpu().detach().numpy(),
                                  affine, self.figures_dir, "Train_GT_heatmap", epoch, iteration)
                        val_saver(batch_heatmap.squeeze().cpu().permute(1, 2, 3, 0).detach().numpy(),
                                  affine, self.figures_dir, "Train_GT_heatmap_separate", epoch, iteration)
                        val_saver(batch_paf.sum(axis=-1).squeeze().cpu().permute(1, 2, 3, 0).detach().numpy(),
                                  affine, self.figures_dir, "Train_GT_paf", epoch, iteration)
                        val_saver(batch_image.squeeze().cpu().detach().numpy(),
                                  affine, self.figures_dir, "Train_GT_Image", epoch, iteration)

                    del heatmap_out, paf_out

                loss = loss_HM + loss_PAF
                print(f'Iter {iteration}: BackBone: {loss.item():.3f}, '
                      f'Heatmap: {loss_HM.item():.3f}, '
                      f'PAF: {loss_PAF.item():.3f}\n')

                # If epoch is very small, so just log once per epoch
                if len(self.train_loader) < 100:
                    if iteration == np.random.randint(0, len(self.train_loader)):
                        writerBB.add_scalar("Loss/Backbone_Overall", loss.item(), epoch)
                        writerBB.add_scalar("Loss/Backbone_HeatMap", loss_HM.item(), epoch)
                        writerBB.add_scalar("Loss/Backbone_PAF", loss_PAF.item(), epoch)
                else:
                    if iteration % 100 == 0:
                        writerBB.add_scalar("Loss/Backbone_Overall", loss.item(), epoch*(len(self.train_loader)) + iteration)
                        writerBB.add_scalar("Loss/Backbone_HeatMap", loss_HM.item(), epoch*(len(self.train_loader)) + iteration)
                        writerBB.add_scalar("Loss/Backbone_PAF", loss_PAF.item(), epoch*(len(self.train_loader)) + iteration)

                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Variable deletion for next loop
                del loss_HM, loss_PAF, loss
                del heatmap_outputs, paf_outputs
                del batch_image, batch_heatmap, batch_paf, patch_s

            ## Validation: Start at zero
            if epoch % self.opt.validation_interval == 0:
                # Set model to eval mode
                self.model.eval()
                # Save Model
                name = os.path.join(self.models_dir, f'PoseEst_model_{epoch}_iteration_{iteration}.pt')

                # Various relevant model-related variables to save
                current_state_dict = {
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch + 1,
                    'running_iter': iteration,
                    # 'total_steps': total_steps,
                    'patch_size': self.opt.patch_size,
                }

                torch.save(current_state_dict, name)
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
                        # Dimension five because channels are reserved for x, y, z
                        val_batch_paf = torch.stack(tuple(val_sample[f"PAF_{x}"] for x in self.selected_PAFs),
                                                    dim=5).to(device)

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
                            val_paf_out = torch.zeros([val_paf_outputs[ii].shape[0],
                                                   3,
                                                   val_paf_outputs[ii].shape[2],
                                                   val_paf_outputs[ii].shape[3],
                                                   val_paf_outputs[ii].shape[4],
                                                   int(val_paf_outputs[ii].shape[1] / 3)
                                                   ]).to(device)
                            # paf_out = torch.reshape(paf_out, [paf_out.shape[0],
                            #                                   3,
                            #                                   paf_out.shape[2],
                            #                                   paf_out.shape[3],
                            #                                   paf_out.shape[4],
                            #                                   int(paf_out.shape[1] / 3)
                            #                                   ])
                            val_paf_out[:, 0, ...] = val_paf_outputs[ii][:, ::3, ...].permute(0, 2, 3, 4, 1)
                            val_paf_out[:, 1, ...] = val_paf_outputs[ii][:, 1::3, ...].permute(0, 2, 3, 4, 1)
                            val_paf_out[:, 2, ...] = val_paf_outputs[ii][:, 2::3, ...].permute(0, 2, 3, 4, 1)

                            val_loss_HM += self.HM_torch_loss(val_batch_heatmap, val_heatmap_out)
                            val_loss_PAF += self.PAF_torch_loss(val_batch_paf, val_paf_out)

                            # Save outputs once every epoch
                            if not agg_loss:  # i.e.: Empty list, so must be first validation step
                                # Save output heatmap and paf
                                val_saver(val_heatmap_out.max(axis=1)[0].squeeze().cpu().detach().numpy(),
                                          val_affine, self.figures_dir, "Out_Val_heatmap", epoch, iteration)
                                val_saver(val_heatmap_out.squeeze().cpu().permute(1, 2, 3, 0).detach().numpy(),
                                          val_affine, self.figures_dir, "Out_Val_heatmap_separate", epoch, iteration)


                                val_saver(val_paf_out.sum(axis=-1).squeeze().permute(3, 1, 2, 0).cpu().detach().numpy(),
                                          val_affine, self.figures_dir, "Val_paf", epoch, iteration)
                                if ii == 0:
                                    # Save OGs
                                    val_saver(val_batch_heatmap.max(axis=1)[0].squeeze().cpu().detach().numpy(),
                                              val_affine, self.figures_dir, "Val_GT_heatmap", epoch, iteration)
                                    val_saver(val_batch_heatmap.squeeze().cpu().permute(1, 2, 3, 0).detach().numpy(),
                                              val_affine, self.figures_dir, "Val_GT_heatmap_separate", epoch, iteration)
                                    val_saver(
                                        val_batch_paf.sum(axis=-1).squeeze().cpu().permute(3, 1, 2, 0).detach().numpy(),
                                        val_affine, self.figures_dir, "Val_GT_paf", epoch, iteration)
                                    val_saver(val_batch_image.sum(axis=-1).squeeze().cpu().detach().numpy(),
                                              val_affine, self.figures_dir, "Val_Image", epoch, iteration)
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
                    print(f"Val Epoch {epoch}:")
                    print(f'Val BackBone: {np.mean(agg_loss):.3f}, '
                          f'Val Heatmap: {np.mean(agg_heatmap_loss):.3f}, '
                          f'Val PAF: {np.mean(agg_paf_loss):.3f}\n')

                    writerBB.add_scalar("Loss/Val_Backbone_Overall", np.mean(agg_loss), epoch)
                    writerBB.add_scalar("Loss/Val_Backbone_HeatMap", np.mean(agg_heatmap_loss), epoch)
                    writerBB.add_scalar("Loss/Val_Backbone_PAF", np.mean(agg_paf_loss), epoch)
                    del agg_loss, agg_heatmap_loss, agg_paf_loss
            self.scheduler.step()


class testTT:
    def __init__(self, models_dir, figures_dir, writer, inf_loader, opt, model,
                 selected_heatmaps, selected_pafs, debug):
        self.models_dir = models_dir
        self.figures_dir = figures_dir
        self.writer = writer
        self.inf_loader = inf_loader
        self.opt = opt
        self.model = model

        self.selected_heatmaps = selected_heatmaps
        self.selected_PAFs = selected_pafs

        self.debug = debug

    def test(self):
        if self.opt.weighted_sampling:
            from monai.inferers import SimpleInferer
            simple_inferer = SimpleInferer()
        else:
            from utils.utils import sliding_window_inference
        self.model.eval()
        for iteration, patch_s in enumerate(self.inf_loader):
            batch_image = patch_s['image'].to(device)

            # Affine: For output-saving purposes
            affine = patch_s['image_meta_dict']['affine'][0, ...]

            sub_name = patch_s['image_meta_dict']["filename_or_obj"][0]

            # heatmap_outputs, paf_outputs = sliding_window_inference(batch_image,
            #                                                         self.opt.patch_size,
            #                                                         1,
            #                                                         self.model,
            #                                                         mode="gaussian",
            #                                                         overlap=0.0)
            if self.opt.weighted_sampling:
                heatmap_outputs, paf_outputs = simple_inferer(batch_image,
                                                              self.model)
            else:
                heatmap_outputs, paf_outputs = sliding_window_inference(batch_image,
                                                                        self.opt.patch_size,
                                                                        1,
                                                                        self.model,
                                                                        mode="gaussian",
                                                                        overlap=0.0)

            val_saver(heatmap_outputs.squueze().cpu().detach().numpy(), affine, self.figures_dir,
                      f"HM_{sub_name}", 999, iteration)
            val_saver(paf_outputs.squeeze().permute(3, 1, 2, 0).cpu().detach().numpy(), affine, self.figures_dir,
                      f"PAF_{sub_name}", 999, iteration)
