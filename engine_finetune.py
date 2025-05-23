# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os, pdb
import math
from typing import Iterable, Optional

import torch
import torch.distributed as dist
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils
from utils import adjust_learning_rate
from scipy.special import softmax
from sklearn.metrics import (
    average_precision_score, 
    accuracy_score
)
import torch.nn.functional as F


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer_gn: torch.optim.Optimizer, optimizer_dt: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, 
                    log_writer=None, args=None):
    model.train(True)

    torch.autograd.set_detect_anomaly(True)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    update_freq = args.update_freq
    use_amp = args.use_amp
    optimizer_gn.zero_grad()

    for data_iter_step, (samples, value_matrixs, targets) in enumerate(metric_logger.log_every(data_loader, print_freq=100, header=header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % update_freq == 0:
            # adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
            adjust_learning_rate(optimizer_gn, data_iter_step / len(data_loader) + epoch, args)
            adjust_learning_rate(optimizer_dt, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        value_matrixs = value_matrixs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                # output = model(samples)
                output_fake_img, _, _ = model(samples, value_matrixs)

                # output = torch.cat([output_real_pred, output_fake_pred], dim=0)
                #
                # targets = torch.cat([
                #     torch.ones_like(output_real_pred),
                #     torch.zeros_like(output_fake_pred)
                # ], dim=0)


                # loss_dt = criterion(output, targets)

                loss_gn = F.l1_loss(output_fake_img, samples)
        else: # full precision
            # output = model(samples)
            # output_fake_img, output_real_pred, output_fake_pred = model(samples, value_matrixs)
            output_fake_img, _, _ = model(samples, value_matrixs)

            # output = torch.cat([output_real_pred, output_fake_pred], dim=0)
            #
            # targets = torch.cat([
            #     torch.ones_like(output_real_pred),
            #     torch.zeros_like(output_fake_pred)
            # ], dim=0)
            #
            # loss_dt = criterion(output, targets)

            loss_gn = F.l1_loss(output_fake_img, samples)

        torch.cuda.synchronize()
        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer_gn, 'is_second_order') and optimizer_gn.is_second_order
            loss_gn /= update_freq

            parameters_gn = [param for name, param in model.named_parameters() if
                             'encoder' in name or 'decoder' in name and param.requires_grad]

            # grad_norm = loss_scaler(loss_dt, optimizer_dt, clip_grad=max_norm,
            #                         parameters=model.parameters(), create_graph=is_second_order,
            #                         update_grad=(data_iter_step + 1) % update_freq == 0)

            grad_norm_gn = loss_scaler(loss_gn, optimizer_gn, clip_grad=max_norm,
                                       parameters=parameters_gn, create_graph=is_second_order,
                                       update_grad=(data_iter_step + 1) % update_freq == 0)

            if (data_iter_step + 1) % update_freq == 0:
                optimizer_gn.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else:  # full precision
            loss_gn /= update_freq
            loss_gn.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer_gn.step()
                optimizer_gn.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)


        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        optimizer_dt.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast():
                _, output_real_pred, output_fake_pred = model(samples, value_matrixs)

                output = torch.cat([output_real_pred, output_fake_pred], dim=0)

                targets = torch.cat([
                    torch.ones_like(output_real_pred),
                    torch.zeros_like(output_fake_pred)
                ], dim=0)

                loss_dt = criterion(output, targets)

                # loss_gn = F.l1_loss(output_fake_img, samples)
        else:  # full precision
            # output_fake_img, output_real_pred, output_fake_pred = model(samples, value_matrixs)
            _, output_real_pred, output_fake_pred = model(samples, value_matrixs)

            output = torch.cat([output_real_pred, output_fake_pred], dim=0)

            targets = torch.cat([
                torch.ones_like(output_real_pred),
                torch.zeros_like(output_fake_pred)
            ], dim=0)

            loss_dt = criterion(output, targets)

            # loss_gn = F.l1_loss(output_fake_img, samples)

        loss_value = loss_dt.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)


        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer_dt, 'is_second_order') and optimizer_dt.is_second_order
            loss_dt /= update_freq

            parameters_dt = [param for name, param in model.named_parameters() if ('encoder' in name or 'discriminator' in name) and param.requires_grad]

            # grad_norm = loss_scaler(loss_dt, optimizer_dt, clip_grad=max_norm,
            #                         parameters=model.parameters(), create_graph=is_second_order,
            #                         update_grad=(data_iter_step + 1) % update_freq == 0)

            grad_norm = loss_scaler(loss_dt, optimizer_dt, clip_grad=max_norm,
                                    parameters=parameters_dt, create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)

            if (data_iter_step + 1) % update_freq == 0:
                optimizer_dt.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss_dt /= update_freq

            # print("output.requires_grad:", output.requires_grad)
            # print("loss_dt.grad_fn:", loss_dt.grad_fn)

            loss_dt.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer_dt.step()
                optimizer_dt.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

        if mixup_fn is None:
            # class_acc = (output.max(-1)[-1] == targets).float().mean()

            pred_labels = (torch.sigmoid(output).flatten() > 0.5).float()
            class_acc = (pred_labels == targets.flatten()).float().mean()
        else:
            class_acc = None

        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer_dt.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer_dt.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)
        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, val=None, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for index, batch in enumerate(metric_logger.log_every(data_loader, 1000, header)):
        images = batch[0]
        value_matrixs = batch[1]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        value_matrixs = value_matrixs.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                # output = model(images)
                output_fake_image, output_real_pred, output_fake_pred = model(images, value_matrixs)

                # if isinstance(output, dict):
                #     output = output['logits']

                output = torch.cat([output_real_pred, output_fake_pred], dim=0)

                target = torch.cat([
                    torch.ones_like(output_real_pred),
                    torch.zeros_like(output_fake_pred)
                ], dim=0)

                loss = criterion(output, target)

        else:
            # output = model(images)
            output_fake_image, output_real_pred, output_fake_pred = model(images, value_matrixs) #[bs, num_cls]
            # if isinstance(output, dict):
            #     output = output['logits']

            output = torch.cat([output_real_pred, output_fake_pred], dim=0)

            target = torch.cat([
                torch.ones_like(output_real_pred),
                torch.zeros_like(output_fake_pred)
            ], dim=0)
            
            loss = criterion(output, target)
        
        if index == 0:
            predictions = output
            labels = target
        else:
            predictions = torch.cat((predictions, output), 0)
            labels = torch.cat((labels, target), 0)

        torch.cuda.synchronize()

        acc1, _ = [acc / 100 for acc in accuracy(output, target, topk=(1, 2))]

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.2%} loss {losses.global_avg:.4f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))

    output_ddp = [torch.zeros_like(predictions) for _ in range(utils.get_world_size())]
    dist.all_gather(output_ddp, predictions)
    labels_ddp = [torch.zeros_like(labels) for _ in range(utils.get_world_size())]
    dist.all_gather(labels_ddp, labels)

    output_all = torch.concat(output_ddp, dim=0)
    labels_all = torch.concat(labels_ddp, dim=0)

    y_pred = torch.sigmoid(output_all).cpu().numpy().squeeze()
    # y_pred = softmax(output_all.detach().cpu().numpy(), axis=1)[:, 1]
    y_true = labels_all.detach().cpu().numpy()
    y_true = y_true.astype(int)
  
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, acc, ap