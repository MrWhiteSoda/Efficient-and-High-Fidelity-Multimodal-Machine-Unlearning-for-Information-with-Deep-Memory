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
# os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"
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
from torch.multiprocessing import Pool, set_start_method
import shap
import concurrent.futures
from torch.utils.data import DataLoader
import cuml
from cuml.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from models.blip import blip_decoder
import utils
from utils import cosine_lr_schedule
from finetune_data import create_dataset, create_sampler, create_loader, custom_collate_fn
from finetune_data.utils import save_result, coco_caption_eval
import warnings
warnings.simplefilter("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)


def custom_generate(model, image, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9,
                    repetition_penalty=1.0, output_scores=False, return_dict_in_generate=False):

    image_embeds = model.visual_encoder(image)

    if not sample:
        image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)

    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

    model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}

    prompt = [model.prompt] * image.size(0)
    input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device)

    input_ids[:, 0] = model.tokenizer.bos_token_id

    input_ids = input_ids[:, :-1]

    if sample:
        # nucleus sampling
        outputs = model.text_decoder.generate(input_ids=input_ids,
                                                     max_length=max_length,
                                                     min_length=min_length,
                                                     do_sample=True,
                                                     top_p=top_p,
                                                     num_return_sequences=1,
                                                     eos_token_id=model.tokenizer.sep_token_id,
                                                     pad_token_id=model.tokenizer.pad_token_id,
                                                     repetition_penalty=1.1,
                                                     output_scores=output_scores,
                                                     return_dict_in_generate=return_dict_in_generate,
                                                     **model_kwargs)
        outputs = outputs[:, model.prompt_length:]
    else:
        # beam search
        outputs = model.text_decoder.generate(input_ids=input_ids,
                                                     max_length=max_length,
                                                     min_length=min_length,
                                                     num_beams=num_beams,
                                                     eos_token_id=model.tokenizer.sep_token_id,
                                                     pad_token_id=model.tokenizer.pad_token_id,
                                                     repetition_penalty=repetition_penalty,
                                                     output_scores=output_scores,
                                                     return_dict_in_generate=return_dict_in_generate,
                                                     **model_kwargs)

    return outputs


def compute_disturbance_outputs(model, images, target_layers, device):

    ffn_outputs = []

    def capture_ffn_output(module, input, output):

        ffn_outputs.append(output[:, -1, :])

    def register_ffn_hook(model, target_layers):
        hooks = []
        for idx, layer in enumerate(model.text_decoder.bert.encoder.layer):
            if idx in target_layers:
                hook = layer.intermediate.register_forward_hook(capture_ffn_output)
                hooks.append(hook)
        return hooks

    hooks = register_ffn_hook(model, target_layers)
    _ = custom_generate(model, images, sample=True, max_length=1, min_length=1)

    disturbance_first_token_outputs = []
    for i in range(len(target_layers)):
        disturbance_first_token_outputs.append(ffn_outputs[i])

    for hook in hooks:
        hook.remove()
    # torch.cuda.empty_cache()
    return disturbance_first_token_outputs


def compute_ffn_outputs_only_input(model, unlearning_images, disturbance_images, target_layers, device):
    unlearning_first_token_outputs = compute_disturbance_outputs(model, unlearning_images, target_layers, device)
    disturbance_first_token_outputs = compute_disturbance_outputs(model, disturbance_images, target_layers, device)
    return unlearning_first_token_outputs, disturbance_first_token_outputs


def parameters_input_solver(unlearning_outputs, disturbance_outputs, target_layers, device):

    layer_indices = []

    unlearning_outputs_tensor = torch.stack(unlearning_outputs, dim=0)
    disturbance_outputs_tensor = torch.stack(disturbance_outputs, dim=0)


    scores = torch.mean(torch.abs(unlearning_outputs_tensor - disturbance_outputs_tensor), dim=1).to(device)
    for idx, layer_idx in enumerate(target_layers):
        layer_indices.append(torch.topk(scores[idx], 1000).indices.tolist())

    return layer_indices


def get_parameter_indices_only_input(model, unlearning_images, disturbance_images, target_layers, device):
    model.eval()

    unlearning_first_token_outputs, disturbance_outputs = compute_ffn_outputs_only_input(
            model, unlearning_images, disturbance_images, target_layers, device)

    parameter_indices = parameters_input_solver(unlearning_first_token_outputs, disturbance_outputs,
                                                      target_layers, device)

    return parameter_indices


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


def train(original_model, unlearning_model, unlearning_loader, remain_loader, optimizer, epoch, params_pool, batch_params_indices, target_layers, device, config):
    unlearning_model.train()
    original_model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('unlearning_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('remain_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('unlearning_remain_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train Caption Epoch: [{}]'.format(epoch)
    print_freq = 50

    gamma1 = config['gamma1']
    gamma2 = config['gamma2']
    gamma3 = config['gamma3']


    iter_remain = iter(remain_loader)
    for i, (unlearning_image, disturbance_image, unlearning_caption, _) in enumerate(
            metric_logger.log_every(unlearning_loader, print_freq, header)):
        remain_image, remain_caption, _ = next(iter_remain)
        remain_image = remain_image.to(device)
        unlearning_image = unlearning_image.to(device)
        disturbance_image = disturbance_image.to(device)

        parameter_indices = get_parameter_indices_only_input(unlearning_model.module, unlearning_image, disturbance_image, target_layers, device)
        params_in_pool = []
        for layer_idx, indices in enumerate(parameter_indices):
            weight_name = f'text_decoder.bert.encoder.layer.{target_layers[layer_idx]}.intermediate.dense.weight'
            if weight_name in params_pool.keys():
                param_pool = params_pool[weight_name]
                indices_tensor = torch.tensor(indices, device=param_pool.device, dtype=torch.long)
                mask = param_pool[indices_tensor, :].any(dim=1)
                params_in_pool.append(indices_tensor[mask].tolist())
        if epoch == 0:
            batch_params_indices[i] = params_in_pool
        else:
            last_params_in_pool = batch_params_indices[i]
            need_update_param = [list(set(current).difference(set(last))) for last, current in zip(last_params_in_pool, params_in_pool)]
            for layer_idx, indices in enumerate(need_update_param):
                weight_name = f'text_decoder.bert.encoder.layer.{target_layers[layer_idx]}.intermediate.dense.weight'
                if weight_name in params_pool.keys():
                    params_pool[weight_name][indices, :] = 0
            batch_params_indices[i] = params_in_pool


        unlearning_loss = -1.0 * unlearning_model.module(unlearning_image, unlearning_caption)

        unlearning_logits = compute_logits(unlearning_model, remain_image, remain_caption, device, config)
        original_logits = compute_logits(original_model, remain_image, remain_caption, device, config)
        prob_p = torch.nn.functional.softmax(original_logits, dim=-1)
        prob_q = torch.nn.functional.softmax(unlearning_logits, dim=-1)

        remain_loss = (prob_p * (torch.log(prob_p + 1e-12) - torch.log(prob_q + 1e-12))).sum(dim=-1).mean()

        unlearning_remain_loss = unlearning_model.module(unlearning_image, remain_caption)

        loss = gamma1 * unlearning_loss + gamma2 * remain_loss + gamma3 * unlearning_remain_loss
        optimizer.zero_grad()
        loss.backward()

        if params_pool:
            for name, param in unlearning_model.module.named_parameters():
                if name in params_pool.keys():

                    param.grad *= params_pool[name].to(device)

        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(unlearning_loss=unlearning_loss.item())
        metric_logger.update(remain_loss=remain_loss.item())
        metric_logger.update(unlearning_remain_loss=unlearning_remain_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, device, config):
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
    unlearning_dataset, remain_dataset, unlearning_val_dataset, val_dataset, test_dataset, remain_val_dataset = create_dataset(
        'caption_flickr_emmu', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(
            [unlearning_dataset, remain_dataset, unlearning_val_dataset, val_dataset, test_dataset, remain_val_dataset],
            [True, True, False, False, False, False], num_tasks,
            global_rank)
    else:
        samplers = [None, None, None, None, None, None]
    print(len(unlearning_dataset))
    print(len(remain_dataset))
    unlearning_loader, remain_loader, unlearning_val_loader, val_loader, test_loader, remain_val_loader = create_loader(
        [unlearning_dataset, remain_dataset, unlearning_val_dataset, val_dataset, test_dataset, remain_val_dataset],
        samplers,
        batch_size=[config['batch_size']] * 6, num_workers=[4, 4, 4, 4, 4, 4],
        is_trains=[True, True, False, False, False, False],
        collate_fns=[None, None, None, None, None, None])

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

    for params in unlearning_model.parameters():
        params.requires_grad = False

    target_layers = config['target_layers']
    if target_layers == 'None':
        target_layers = list(range(12))
        print(target_layers)
    else:
        target_layers = [int(x) for x in target_layers.split(",")]

    for name, param in unlearning_model.named_parameters():
        for layer_idx in target_layers:
            if f'layer.{layer_idx}.intermediate' in name:
                param.requires_grad = True
                break

    optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, unlearning_model.parameters()),
                                  lr=config['init_lr'], weight_decay=config['weight_decay'])

    best = 0
    best_epoch = 0

    if config['param_indices'] is not None:
        param_indices = json.load(open(config['param_indices'], 'r'))
        params_pool = {}
        for idx, indices in enumerate(param_indices):
            layer = unlearning_model_without_ddp.text_decoder.bert.encoder.layer[target_layers[idx]].intermediate
            param_pool = torch.zeros_like(layer.dense.weight)
            param_pool[indices, :] = 1
            params_pool[f'text_decoder.bert.encoder.layer.{target_layers[idx]}.intermediate.dense.weight'] = param_pool
    else:
        params_pool = None
    batch_params_indices = {}
    print("Start training")
    start_time = time.time()
    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:
            if args.distributed:
                unlearning_loader.sampler.set_epoch(epoch)
                remain_loader.sampler.set_epoch(epoch)

            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            train_stats = train(original_model, unlearning_model, unlearning_loader, remain_loader, optimizer,
                                epoch, params_pool, batch_params_indices, target_layers, device, config)
        val_result = evaluate(unlearning_model_without_ddp, val_loader, device, config)
        val_result_file = save_result(val_result, args.result_dir, 'val_epoch%d' % epoch, remove_duplicate='image_id')
        test_result = evaluate(unlearning_model_without_ddp, test_loader, device, config)
        test_result_file = save_result(test_result, args.result_dir, 'test_epoch%d' % epoch,
                                       remove_duplicate='image_id')
        unlearning_result = evaluate(unlearning_model_without_ddp, unlearning_val_loader, device, config)
        unlearning_result_file = save_result(unlearning_result, args.result_dir, 'unlearning_val_epoch%d' % epoch,
                                             remove_duplicate='image_id')
        remain_result = evaluate(unlearning_model_without_ddp, remain_val_loader, device, config)
        remain_result_file = save_result(remain_result, args.result_dir, 'remain_val_epoch%d' % epoch,
                                         remove_duplicate='image_id')


        if utils.is_main_process():
            coco_val = coco_caption_eval(config['ann_root'], val_result_file, 'val')
            coco_test = coco_caption_eval(config['ann_root'], test_result_file, 'test')
            coco_unlearning_val = coco_caption_eval(config['ann_root'], unlearning_result_file,
                                                    filename=config['sensitive_val_gt'])
            coco_remain_val = coco_caption_eval(config['ann_root'], remain_result_file,
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
    parser.add_argument('--config', default='./unlearning_configs/emmu_configs/caption_flickr_sensitive.yaml')
    parser.add_argument('--output_dir', default='./output/Caption_flickr_emmu_dbi_only_input')
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
