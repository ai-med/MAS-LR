import os
import re
import time
import numpy as np
import pandas as pd
import torch
from glob import glob
from utils import losses as additional_losses
from torch.optim import lr_scheduler

import utils.common_utils as cu
from utils.log_utils import LogWriter
import utils.evaluator as eu

CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_EXTENSION = 'pth.tar'


class Solver(object):
    def __init__(self,
                 model,
                 exp_name,
                 device,
                 optim,
                 loss_func=additional_losses.CombinedLoss(),
                 model_name='quicknat',
                 classes=None,
                 num_classes=None,
                 num_epochs=10,
                 log_nth=5,
                 lr_scheduler_step_size=None,
                 lr_scheduler_gamma=None,
                 save_best_ckpt=True,
                 use_last_checkpoint=True,
                 checkpoint_path=None,
                 exp_dir='experiments',
                 log_dir='logs',
                 surrogate_reg_param=0,
                 average_weight_shifts=False):

        self.device = device
        self.model = model
        self.model_name = model_name
        self.classes = classes
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.loss_func = loss_func.cuda(device) if torch.cuda.is_available() else loss_func
        self.optim = optim
        self.scheduler = None if lr_scheduler_step_size is None else lr_scheduler.StepLR(self.optim,
                                                                                         step_size=lr_scheduler_step_size,
                                                                                         gamma=lr_scheduler_gamma)
        self.exp_dir_path = os.path.join(exp_dir, exp_name)
        self.log_nth = log_nth
        self.log_writer = LogWriter(['train', 'val'], log_dir, exp_name, use_last_checkpoint)
        self.save_best_ckpt = save_best_ckpt
        self.use_last_checkpoint = use_last_checkpoint
        self.checkpoint_path = checkpoint_path
        self.start_epoch = 1
        self.start_iteration = 1
        self.best_ds_mean = 0
        self.best_ds_mean_epoch = 0
        self.surrogate_reg_param = surrogate_reg_param
        self.average_weight_shifts = average_weight_shifts

        cu.create_if_not(os.path.join(self.exp_dir_path, CHECKPOINT_DIR))

        if use_last_checkpoint:
            self.load_checkpoint(self.checkpoint_path)

    def train(self, train_loader, val_loader):
        """
        Train a given model with the provided data.
        :param train_loader: Train data
        :param val_loader: Validation data
        """
        dataloaders = {'train': train_loader, 'val': val_loader}

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.model.cuda(self.device)

        current_iteration = self.start_iteration

        rnd_indicess = {phase: np.random.choice(len(dataloaders[phase].dataset), 3, replace=False) for phase in
                        ['train', 'val']}
        log_data = {phase: {'imgs': dataloaders[phase].dataset.imgs[rnd_indicess[phase]],
                            'labelss': dataloaders[phase].dataset.labelss[rnd_indicess[phase]]} for
                    phase in ['train', 'val']}

        loss_dict = {'loss': [], 'surr_loss': [], 'total_loss': []}

        start = time.time()
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            for phase in ['train', 'val']:
                print('\nTraining ...' if phase == 'train' else '\nValidating ...')
                out_list = []
                y_list = []

                if phase == 'train':
                    self.model.train()
                    if self.scheduler is not None:
                        self.scheduler.step()
                else:
                    self.model.eval()

                for i, (_, imgs, labelss, class_weightss) in enumerate(dataloaders[phase]):
                    imgs, labelss, class_weightss = imgs.type(torch.FloatTensor), labelss.type(
                        torch.LongTensor), class_weightss.type(torch.FloatTensor)

                    if self.model.is_cuda:
                        imgs = imgs.cuda(self.device, non_blocking=True)
                        labelss = labelss.cuda(self.device, non_blocking=True)
                        class_weightss = class_weightss.cuda(self.device, non_blocking=True)

                    output = self.model(imgs)

                    loss = self.loss_func(output, labelss, class_weightss)
                    surr_loss = self.model.surrogate_loss(
                        self.average_weight_shifts) if self.surrogate_reg_param != 0 else torch.tensor(0.)
                    total_loss = loss + self.surrogate_reg_param * surr_loss

                    loss_dict['loss'].append(loss.item())
                    loss_dict['surr_loss'].append(surr_loss.item())
                    loss_dict['total_loss'].append(total_loss.item())

                    if phase == 'train':
                        self.optim.zero_grad()
                        total_loss.backward()

                        if hasattr(self.model, 'freeze_masks') and self.model.freeze_masks != {}:
                            for name, freeze_mask in self.model.freeze_masks.items():
                                module_names = re.sub(r'\.(weight|bias)', '', name).split('.')
                                module = self.model
                                for module_name in module_names:
                                    module = module._modules[module_name]

                                if 'weight' in name:
                                    module.weight.grad *= (freeze_mask != 0.).type(torch.float)
                                else:
                                    module.bias.grad *= (freeze_mask != 0.).type(torch.float)

                        self.optim.step()

                        if i % self.log_nth == 0:
                            cu.print_progress(start, (epoch - 1) * len(dataloaders[phase]) + i,
                                              self.num_epochs * len(dataloaders[phase]),
                                              'Epoch: [{} / {}] - Batch: [{} / {}] - Train Loss: {:.4f}'.format(
                                                  epoch, self.num_epochs, i, len(dataloaders[phase]), total_loss))

                            for name, losses in loss_dict.items():
                                self.log_writer.writers[phase].add_scalar('loss/per_iteration/{}'.format(name),
                                                                          losses[-1], current_iteration)
                        current_iteration += 1

                    _, batch_output = torch.max(output, dim=1)
                    out_list.append(batch_output.cpu())
                    y_list.append(labelss.cpu())

                    del imgs, labelss, class_weightss, output, batch_output, loss, surr_loss, total_loss
                    torch.cuda.empty_cache()

                with torch.no_grad():
                    out_arr, y_arr = torch.cat(out_list), torch.cat(y_list)

                    for name, value in loss_dict.items():
                        self.log_writer.writers[phase].add_scalar(
                            'loss/per_epoch/{}'.format(name), value[-1] if phase == 'train' else np.mean(value), epoch)

                    self.log_writer.save_image(phase, 'sample_prediction/per_epoch', self.classes,
                                               self.model.predict(log_data[phase]['imgs'], self.device),
                                               log_data[phase]['labelss'], epoch)

                    cm_dice_scoress = eu.calc_cm_dice_scoress(out_arr[rnd_indicess[phase]], y_arr[rnd_indicess[phase]],
                                                              self.classes)
                    self.log_writer.plot_cm(phase, 'confusion_matrix/per_epoch', cm_dice_scoress, epoch)

                    if phase == 'train':
                        class_dice_scores = pd.DataFrame(
                            eu.calc_class_dice_scores(out_arr[rnd_indicess[phase]], y_arr[rnd_indicess[phase]],
                                                      self.classes), index=['Dice Score']).T
                    else:
                        class_dice_scores = pd.DataFrame(eu.calc_class_dice_scores(out_arr, y_arr, self.classes),
                                                         index=['Dice Score']).T

                    self.log_writer.plot_charts(phase, 'avg_class_dice_scores/per_epoch', class_dice_scores, epoch)

                    avg_dice_score = class_dice_scores.mean().iloc[0]
                    self.log_writer.writers[phase].add_scalar('avg_dice_score/per_epoch', avg_dice_score, epoch)

                    if phase == 'val':
                        if avg_dice_score > self.best_ds_mean or not self.save_best_ckpt or epoch == self.num_epochs:
                            print('Saving {}model with avg. score of {:.2f} at epoch {}'.format(
                                'best ' if self.save_best_ckpt and not epoch == self.num_epochs else '', avg_dice_score,
                                epoch))
                            print('at:', os.path.join(self.exp_dir_path, CHECKPOINT_DIR))
                            if self.save_best_ckpt and not epoch == self.num_epochs:
                                self.delete_checkpoint()

                            save_dict = {
                                'epoch': epoch + 1,
                                'start_iteration': current_iteration + 1,
                                'arch': self.model_name,
                                'state_dict': self.model.state_dict(),
                                'optimizer': self.optim.state_dict()
                            }
                            if self.scheduler is not None:
                                save_dict['scheduler'] = self.scheduler.state_dict()
                            torch.save(save_dict, os.path.join(self.exp_dir_path, CHECKPOINT_DIR,
                                                               '{}epoch-{}_score-{:.2f}.{}'.format(
                                                                   '' if not self.save_best_ckpt else 'best-', epoch,
                                                                   avg_dice_score, CHECKPOINT_EXTENSION)))

                        if avg_dice_score > self.best_ds_mean:
                            self.best_ds_mean = avg_dice_score
                            self.best_ds_mean_epoch = epoch

        self.log_writer.close()

        print('Finished training')

    def get_checkpoint_path(self, epoch=None):
        list_of_files = glob(os.path.join(self.exp_dir_path, CHECKPOINT_DIR, '*.' + CHECKPOINT_EXTENSION))

        if len(list_of_files) > 0:
            if epoch:
                checkpoint_path = [fn for fn in list_of_files if 'epoch-{}'.format(epoch) in fn]
                if len(checkpoint_path) == 1:
                    return checkpoint_path[0]
                else:
                    print('no checkpoint found at "{}" with epoch {}'.format(
                        os.path.join(self.exp_dir_path, CHECKPOINT_DIR), epoch))
                    return None
            else:
                return max(list_of_files, key=os.path.getctime)
        else:
            print('no checkpoint found at "{}"'.format(os.path.join(self.exp_dir_path, CHECKPOINT_DIR)))
            return None

    def delete_checkpoint(self, epoch=None):
        checkpoint_path = self.get_checkpoint_path(epoch)
        if checkpoint_path:
            os.remove(checkpoint_path)

    def load_checkpoint(self, checkpoint_path=None, epoch=None):
        file_path = checkpoint_path if checkpoint_path is not None else self.get_checkpoint_path(epoch)
        if file_path:
            print('=> loading checkpoint "{}"'.format(file_path))
            checkpoint = torch.load(file_path)
            self.start_epoch = checkpoint['epoch']
            self.start_iteration = checkpoint['start_iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer'])

            for state in self.optim.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)

            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            print('loaded checkpoint "{}" - epoch {}'.format(file_path, checkpoint['epoch']))
