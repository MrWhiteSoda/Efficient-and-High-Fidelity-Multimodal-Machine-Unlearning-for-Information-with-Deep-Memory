'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
 * Modified by Soda
'''
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.blip import blip_decoder
import utils
from utils import cosine_lr_schedule
from finetune_data import create_dataset, create_sampler, create_loader
from finetune_data.utils import save_result, coco_caption_eval


def save_mask(mask, result_dir, filename):
    mask_file = os.path.join(result_dir, '%s_rank%d.pt' % (filename, utils.get_rank()))
    final_mask_file = os.path.join(result_dir, '%s.pt' % filename)

    torch.save(mask, mask_file)
    dist.barrier()

    if utils.is_main_process():
        result = []
        for rank in range(utils.get_world_size()):
            mask_file = os.path.join(result_dir, '%s_rank%d.pt' % (filename, rank))
            result.append(torch.load(mask_file))

        torch.save(result, final_mask_file)
        print('mask file saved to %s' % final_mask_file)


def get_weight_mask(model, dataloader, result_dir, device):

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'],
                                  weight_decay=config['weight_decay'])
    gradients = {}
    for name, param in model.named_parameters():
        gradients[name] = 0

    model.eval()

    for image, caption, _ in dataloader:
        image = image.to(device)

        loss = model.module(image, caption)
        optimizer.zero_grad()
        loss.backward()



        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradient = param.grad.data.cpu()
                    gradients[name] += gradient

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])

        threshold_list = [0.5]
        for i in threshold_list:

            sorted_dict_positions = {}
            hard_dict = {}

            all_elements = - torch.cat(
                [tensor.flatten() for tensor in gradients.values()]
            )

            threshold_index = int(len(all_elements) * i)


            positions = torch.argsort(all_elements)
            ranks = torch.argsort(positions)

            start_index = 0
            for key, tensor in gradients.items():
                num_elements = tensor.numel()
                tensor_ranks = ranks[start_index: start_index + num_elements]

                sorted_positions = tensor_ranks.reshape(tensor.shape)
                sorted_dict_positions[key] = sorted_positions

                threshold_tensor = torch.zeros_like(tensor_ranks)
                threshold_tensor[tensor_ranks < threshold_index] = 1
                threshold_tensor = threshold_tensor.reshape(tensor.shape)
                hard_dict[key] = threshold_tensor
                start_index += num_elements

    return hard_dict


def train(original_model, unlearning_model, unlearning_loader, dismatch_loader, match_loader, optimizer, epoch, device, result_dir, mask):
    unlearning_model.train()
    original_model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('unlearning_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('remain_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Caption Epoch: [{}]'.format(epoch)
    print_freq = 50

    iter_dismatch = iter(dismatch_loader)
    iter_match = iter(match_loader)
    for i, (unlearning_image, unlearning_caption, _) in enumerate(metric_logger.log_every(unlearning_loader, print_freq, header)):
        _, dismatch_caption, _ = next(iter_dismatch)
        match_image, match_caption, _ = next(iter_match)
        match_image = match_image.to(device)
        unlearning_image = unlearning_image.to(device)

        remain_loss = unlearning_model.module(match_image, match_caption)

        unlearning_loss = unlearning_model.module(unlearning_image, dismatch_caption)

        loss = unlearning_loss + config['alpha'] * remain_loss

        optimizer.zero_grad()
        loss.backward()

        if mask:
            for name, param in unlearning_model.named_parameters():
                if param.grad is not None:
                    param.grad *= mask[name].to(param.grad.device)
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(unlearning_loss=unlearning_loss.item())
        metric_logger.update(remain_loss=remain_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, device, config):
    # evaluate
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 10

    result = []
    for image, image_id in metric_logger.log_every(data_loader, print_freq, header):

        image = image.to(device)

        captions = model.generate(image, sample=False, num_beams=config['num_beams'], max_length=config['max_length'],
                                  min_length=config['min_length'])

        for caption, img_id in zip(captions, image_id):
            result.append({"image_id": img_id.item(), "caption": caption})

    return result


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating captioning dataset")
    unlearning_dataset, dismatch_dataset, match_dataset, unlearning_val_dataset, val_dataset, test_dataset, remain_val_dataset = create_dataset(
        'caption_coco_salun', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(
            [unlearning_dataset, dismatch_dataset, match_dataset, unlearning_val_dataset, val_dataset, test_dataset,
             remain_val_dataset], [True, True, True, False, False, False, False], num_tasks,
            global_rank)
    else:
        samplers = [None, None, None, None, None, None, None]
    unlearning_loader, dismatch_loader, match_loader, unlearning_val_loader, val_loader, test_loader, remain_val_loader = create_loader(
        [unlearning_dataset, dismatch_dataset, match_dataset, unlearning_val_dataset, val_dataset, test_dataset,
         remain_val_dataset], samplers,
        batch_size=[config['batch_size']] * 7, num_workers=[4, 4, 4, 4, 4, 4, 4],
        is_trains=[True, True, True, False, False, False, False],
        collate_fns=[None, None, None, None, None, None, None])

    #### Model ####
    print("Creating model")
    original_model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'],
                                  vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                                  prompt=config['prompt'])
    unlearning_model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'],
                                    vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                                    prompt=config['prompt'])

    original_model = original_model.to(device)
    unlearning_model = unlearning_model.to(device)

    unlearning_model_without_ddp = unlearning_model
    if args.distributed:
        original_model = torch.nn.parallel.DistributedDataParallel(original_model, device_ids=[args.gpu])
        unlearning_model = torch.nn.parallel.DistributedDataParallel(unlearning_model, device_ids=[args.gpu])
        unlearning_model_without_ddp = unlearning_model.module


    optimizer = torch.optim.AdamW(params=unlearning_model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()
    mask = None
    for epoch in range(0, config['max_epoch']):
        if epoch == 0:
            start_mask_time = time.time()
            mask = get_weight_mask(original_model, unlearning_loader, args.result_dir, device)
            mask_time = time.time() - start_mask_time
            print(f"time of getting mask: {str(datetime.timedelta(seconds=int(mask_time)))}")
        if not args.evaluate:
            if args.distributed:
                unlearning_loader.sampler.set_epoch(epoch)
                dismatch_loader.sampler.set_epoch(epoch)
                match_loader.sampler.set_epoch(epoch)

            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            train_stats = train(original_model, unlearning_model, unlearning_loader, dismatch_loader, match_loader, optimizer, epoch, device, args.result_dir, mask)
        val_result = evaluate(unlearning_model_without_ddp, val_loader, device, config)
        val_result_file = save_result(val_result, args.result_dir, 'val_epoch%d' % epoch, remove_duplicate='image_id')
        test_result = evaluate(unlearning_model_without_ddp, test_loader, device, config)
        test_result_file = save_result(test_result, args.result_dir, 'test_epoch%d' % epoch,
                                       remove_duplicate='image_id')
        unlearning_result = evaluate(unlearning_model_without_ddp, unlearning_val_loader, device, config)
        unlearning_result_file = save_result(unlearning_result, args.result_dir, 'unlearning_val_epoch%d' % epoch,
                                      remove_duplicate='image_id')
        remain_result = evaluate(unlearning_model_without_ddp, remain_val_loader, device, config)
        remain_result_file = save_result(remain_result, args.result_dir, 'remain_epoch%d' % epoch,
                                         remove_duplicate='image_id')

        if utils.is_main_process():
            coco_val = coco_caption_eval(config['coco_gt_root'], val_result_file, filename=config['val_gt'])
            coco_test = coco_caption_eval(config['coco_gt_root'], test_result_file, filename=config['test_gt'])
            coco_unlearning_val = coco_caption_eval(config['coco_gt_root'], unlearning_result_file, filename=config['sensitive_val_gt'])
            coco_remain_val = coco_caption_eval(config['coco_gt_root'], remain_result_file,
                                                filename=config['remain_val_gt'])

            if args.evaluate:
                log_stats = {**{f'val_{k}': v for k, v in coco_val.eval.items()},
                             **{f'test_{k}': v for k, v in coco_test.eval.items()},
                             **{f'unlearning_val_{k}': v for k, v in coco_unlearning_val.eval.items()},
                             **{f'remain_val_{k}': v for k, v in coco_remain_val.eval.items()},
                             }
                with open(os.path.join(args.output_dir, "evaluate.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            else:
                save_obj = {
                    'model': unlearning_model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }

                if coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4'] > best:
                    best = coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4']
                    best_epoch = epoch
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in coco_val.eval.items()},
                             **{f'test_{k}': v for k, v in coco_test.eval.items()},
                             **{f'unlearning_val_{k}': v for k, v in coco_unlearning_val.eval.items()},
                             **{f'remain_val_{k}': v for k, v in coco_remain_val.eval.items()},
                             'epoch': epoch,
                             'best_epoch': best_epoch,
                             }
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        if args.evaluate:
            break
        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./unlearning_configs/train_configs/caption_coco_salun.yaml')
    parser.add_argument('--output_dir', default='./output/Caption_coco_salun_resume')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
