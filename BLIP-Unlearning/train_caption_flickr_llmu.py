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


def compute_logits(model, image, caption, device, config):
    image_embeds = model.module.visual_encoder(image)
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
    text = model.module.tokenizer(caption, padding='longest', truncation=True, max_length=60, return_tensors="pt").to(
        device)
    text.input_ids[:, 0] = model.module.tokenizer.bos_token_id
    decoder_targets = text.input_ids.masked_fill(text.input_ids == model.module.tokenizer.pad_token_id, -100)
    decoder_targets[:, :model.module.prompt_length] = -100
    logits = model.module.text_decoder(text.input_ids,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       labels=decoder_targets,
                                       return_logits=True,
                                       )
    logits = logits.view(-1, logits.shape[-1])
    return logits


def train(original_model, unlearning_model, unlearning_loader, dismatch_loader, maintain_loader, optimizer, epoch, device):
    # train
    unlearning_model.train()
    original_model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('unlearning_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('dismatch_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('maintain_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Caption Epoch: [{}]'.format(epoch)
    print_freq = 50

    epsilon1 = config['epsilon1']
    epsilon2 = config['epsilon2']
    epsilon3 = config['epsilon3']

    iter_dismatch = iter(dismatch_loader)
    iter_maintain = iter(maintain_loader)
    for i, (unlearning_image, unlearning_caption, _) in enumerate(metric_logger.log_every(unlearning_loader, print_freq, header)):
        dismatch_image, dismatch_caption, _ = next(iter_dismatch)
        dismatch_image = dismatch_image.to(device)
        maintain_image, maintain_caption, _ = next(iter_maintain)
        maintain_image = maintain_image.to(device)
        unlearning_image = unlearning_image.to(device)

        unlearning_loss = -unlearning_model.module(unlearning_image, unlearning_caption)

        dismatch_loss = unlearning_model.module(dismatch_image, dismatch_caption)

        unlarning_logits = compute_logits(unlearning_model, maintain_image, maintain_caption, device, config)
        original_logits = compute_logits(original_model, maintain_image, maintain_caption, device, config)
        maintain_loss = nn.KLDivLoss(reduction="batchmean")(F.log_softmax(unlarning_logits, dim=-1),
                                                          F.softmax(original_logits, dim=-1))

        loss = epsilon1 * unlearning_loss + epsilon2 * dismatch_loss + epsilon3 * maintain_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(unlearning_loss=unlearning_loss.item())
        metric_logger.update(dismatch_loss=dismatch_loss.item())
        metric_logger.update(maintain_loss=maintain_loss.item())
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

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating captioning dataset")
    unlearning_dataset, dismatch_dataset, maintain_dataset, unlearning_val_dataset, val_dataset, test_dataset, remain_val_dataset = create_dataset(
        'caption_flickr_llmu', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(
            [unlearning_dataset, dismatch_dataset, maintain_dataset, unlearning_val_dataset, val_dataset, test_dataset,
             remain_val_dataset], [True, True, True, False, False, False, False], num_tasks,
            global_rank)
    else:
        samplers = [None, None, None, None, None, None, None]
    unlearning_loader, dismatch_loader, maintain_loader, unlearning_val_loader, val_loader, test_loader, remain_val_loader = create_loader(
        [unlearning_dataset, dismatch_dataset, maintain_dataset, unlearning_val_dataset, val_dataset, test_dataset, remain_val_dataset], samplers,
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

    for params in original_model.parameters():
        params.requires_grad = False

    optimizer = torch.optim.AdamW(params=unlearning_model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()
    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:
            if args.distributed:
                unlearning_loader.sampler.set_epoch(epoch)
                dismatch_loader.sampler.set_epoch(epoch)
                maintain_loader.sampler.set_epoch(epoch)

            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            train_stats = train(original_model, unlearning_model, unlearning_loader, dismatch_loader, maintain_loader, optimizer, epoch, device)
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
            coco_val = coco_caption_eval(config['flickr_gt_root'], val_result_file, 'val')
            coco_test = coco_caption_eval(config['flickr_gt_root'], test_result_file, 'test')
            coco_unlearning_val = coco_caption_eval(config['flickr_gt_root'], unlearning_result_file, filename=config['sensitive_val_gt'])
            coco_remain_val = coco_caption_eval(config['flickr_gt_root'], remain_result_file,
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
    parser.add_argument('--config', default='./unlearning_configs/train_configs/caption_flickr_llmu.yaml')
    parser.add_argument('--output_dir', default='./output/Caption_flickr_llmu_resume')
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
