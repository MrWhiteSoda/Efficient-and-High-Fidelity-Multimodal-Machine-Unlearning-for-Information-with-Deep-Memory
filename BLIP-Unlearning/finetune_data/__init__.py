import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from finetune_data.flickr_caption_dataset import flickr_karpathy_caption_train, flickr_karpathy_caption_eval, \
    flickr_karpathy_retrieval_eval, flickr_karpathy_caption_train_event, flickr_karpathy_caption_train_disturbance
from data.nocaps_dataset import nocaps_eval
# from data.flickr30k_dataset import flickr30k_train, flickr30k_retrieval_eval
from finetune_data.vqa_dataset import vqa_dataset
from data.nlvr_dataset import nlvr_dataset
from data.pretrain_dataset import pretrain_dataset
from transform.randaugment import RandomAugment


def create_dataset(dataset, config, min_scale=0.5):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(config['image_size'], scale=(min_scale, 1.0),
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 5, isPIL=True, augs=['Identity', 'AutoContrast', 'Brightness', 'Sharpness', 'Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size']), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    if dataset == 'pretrain':
        dataset = pretrain_dataset(config['train_file'], config['laion_path'], transform_train)
        return dataset

    elif dataset == 'caption_flickr':
        train_dataset = flickr_karpathy_caption_train(transform_train, config['image_root'], config['ann_root'],
                                                      prompt=config['prompt'])
        val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'test')
        return train_dataset, val_dataset, test_dataset

    elif dataset == 'caption_adversarial':
        val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], filename=config['filename'])
        return val_dataset

    elif dataset == 'caption_flickr_sensitive':
        train_dataset = flickr_karpathy_caption_train(transform_train, config['image_root'], config['ann_root'],
                                                      prompt=config['prompt'], filename=config['sensitive_train'])
        val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'test')
        sensitive_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                             filename=config['sensitive_val'])
        remain_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                      filename=config['remain_val'])
        return train_dataset, val_dataset, test_dataset, sensitive_val_dataset, remain_dataset

    elif dataset == 'caption_coco_sensitive':
        train_dataset = flickr_karpathy_caption_train(transform_train, config['image_root'], config['ann_root'],
                                                      prompt=config['prompt'], filename=config['sensitive_train'])
        val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                   filename=config['val'])
        test_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                    filename=config['test'])
        sensitive_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                             filename=config['sensitive_val'])
        remain_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                      filename=config['remain_val'])
        return train_dataset, val_dataset, test_dataset, sensitive_val_dataset, remain_dataset

    elif dataset == 'caption_flickr_finetune':
        train_dataset = flickr_karpathy_caption_train(transform_train, config['image_root'], config['ann_root'],
                                                      prompt=config['prompt'], filename=config['finetune_train'])
        val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'test')
        unlearning_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                              filename=config['sensitive_val'])
        remain_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                      filename=config['remain_val'])
        return train_dataset, val_dataset, test_dataset, unlearning_val_dataset, remain_dataset

    elif dataset == 'caption_coco_finetune':
        train_dataset = flickr_karpathy_caption_train(transform_train, config['image_root'], config['ann_root'],
                                                      prompt=config['prompt'], filename=config['finetune_train'])
        val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                   filename=config['val'])
        test_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                    filename=config['test'])
        unlearning_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                              filename=config['sensitive_val'])
        remain_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                      filename=config['remain_val'])
        return train_dataset, val_dataset, test_dataset, unlearning_val_dataset, remain_dataset

    elif dataset == 'caption_flickr_retrain':
        train_dataset = flickr_karpathy_caption_train(transform_train, config['image_root'], config['ann_root'],
                                                      prompt=config['prompt'], filename=config['retrain'])
        val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'test')
        remain_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                      filename=config['remain_val'])
        dbi_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                       split=None, filename=config['dbi_val'])
        edu_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                       split=None, filename=config['edu_val'])
        fi_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                      split=None, filename=config['fi_val'])
        idc_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                       split=None, filename=config['idc_val'])
        mi_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                      split=None, filename=config['mi_val'])
        resume_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                          split=None, filename=config['resume_val'])
        return train_dataset, val_dataset, test_dataset, remain_dataset, dbi_val_dataset, edu_val_dataset, fi_val_dataset, idc_val_dataset, mi_val_dataset, resume_val_dataset

    elif dataset == 'caption_coco_retrain':
        train_dataset = flickr_karpathy_caption_train(transform_train, config['image_root'], config['ann_root'],
                                                      prompt=config['prompt'], filename=config['retrain'])
        val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                   filename=config['val'])
        test_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                    filename=config['test'])
        remain_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                      filename=config['remain_val'])
        dbi_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                       split=None, filename=config['dbi_val'])
        edu_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                       split=None, filename=config['edu_val'])
        fi_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                      split=None, filename=config['fi_val'])
        idc_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                       split=None, filename=config['idc_val'])
        mi_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                      split=None, filename=config['mi_val'])
        resume_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                          split=None, filename=config['resume_val'])
        return train_dataset, val_dataset, test_dataset, remain_dataset, dbi_val_dataset, edu_val_dataset, fi_val_dataset, idc_val_dataset, mi_val_dataset, resume_val_dataset

    elif dataset == 'caption_flickr_kul':
        train_dataset = flickr_karpathy_caption_train(transform_train, config['image_root'], config['ann_root'],
                                                      prompt=config['prompt'], filename=config['sensitive_train'])
        val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'test')
        unlearning_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                              split=None, filename=config['sensitive_val'])
        remain_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                      filename=config['remain_val'])
        return train_dataset, val_dataset, test_dataset, unlearning_val_dataset, remain_dataset

    elif dataset == 'caption_coco_kul':
        train_dataset = flickr_karpathy_caption_train(transform_train, config['image_root'], config['ann_root'],
                                                      prompt=config['prompt'], filename=config['sensitive_train'])
        val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                   filename=config['val'])
        test_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                    filename=config['test'])
        unlearning_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                              split=None, filename=config['sensitive_val'])
        remain_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                      filename=config['remain_val'])
        return train_dataset, val_dataset, test_dataset, unlearning_val_dataset, remain_dataset

    elif dataset == 'caption_flickr_mmu':
        unlearning_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
                                                           ann_root=config['ann_root'], prompt=config['prompt'],
                                                           filename=config['sensitive_train'])
        remain_dismatch_dataset = flickr_karpathy_caption_train(transform=transform_train,
                                                                image_root=config['image_root'],
                                                                ann_root=config['ann_root'], prompt=config['prompt'],
                                                                filename=config['remain_dismatch'])
        remain_match_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
                                                             ann_root=config['ann_root'], prompt=config['prompt'],
                                                             filename=config['remain_match'])
        unlearning_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                              filename=config['sensitive_val'])
        val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'test')
        remain_val_dataset = flickr_karpathy_caption_eval(transform=transform_test, image_root=config['image_root'],
                                                          ann_root=config['ann_root'], filename=config['remain_val'])
        return unlearning_dataset, remain_dismatch_dataset, remain_match_dataset, unlearning_val_dataset, val_dataset, test_dataset, remain_val_dataset

    elif dataset == 'caption_coco_mmu':
        unlearning_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
                                                           ann_root=config['ann_root'], prompt=config['prompt'],
                                                           filename=config['sensitive_train'])
        remain_dismatch_dataset = flickr_karpathy_caption_train(transform=transform_train,
                                                                image_root=config['image_root'],
                                                                ann_root=config['ann_root'], prompt=config['prompt'],
                                                                filename=config['remain_dismatch'])
        remain_match_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
                                                             ann_root=config['ann_root'], prompt=config['prompt'],
                                                             filename=config['remain_match'])
        unlearning_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                              filename=config['sensitive_val'])
        val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                   filename=config['val'])
        test_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                    filename=config['test'])
        remain_val_dataset = flickr_karpathy_caption_eval(transform=transform_test, image_root=config['image_root'],
                                                          ann_root=config['ann_root'], filename=config['remain_val'])
        return unlearning_dataset, remain_dismatch_dataset, remain_match_dataset, unlearning_val_dataset, val_dataset, test_dataset, remain_val_dataset

    elif dataset == 'caption_flickr_mmu_part':
        unlearning_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
                                                           ann_root=config['ann_root'], prompt=config['prompt'],
                                                           filename=config['sensitive_train'])
        remain_dismatch_dataset = flickr_karpathy_caption_train(transform=transform_train,
                                                                image_root=config['image_root'],
                                                                ann_root=config['ann_root'], prompt=config['prompt'],
                                                                filename=config['remain_dismatch'])
        remain_match_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
                                                             ann_root=config['ann_root'], prompt=config['prompt'],
                                                             filename=config['remain_match'])
        unlearning_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                              filename=config['sensitive_val'])
        val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'test')
        remain_val_dataset = flickr_karpathy_caption_eval(transform=transform_test, image_root=config['image_root'],
                                                          ann_root=config['ann_root'], filename=config['remain_val'])
        unlearning_part_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                               filename=config['sensitive_remain_val'])

        return unlearning_dataset, remain_dismatch_dataset, remain_match_dataset, unlearning_val_dataset, val_dataset, test_dataset, remain_val_dataset, unlearning_part_dataset

    elif dataset == 'caption_flickr_mmu_part_augmentation':
        unlearning_dataset = flickr_karpathy_caption_train(transform=transform_train,
                                                           image_root=config['augmentation_root'],
                                                           ann_root=config['ann_root'], prompt=config['prompt'],
                                                           filename=config['sensitive_train'])
        remain_dismatch_dataset = flickr_karpathy_caption_train(transform=transform_train,
                                                                image_root=config['image_root'],
                                                                ann_root=config['ann_root'], prompt=config['prompt'],
                                                                filename=config['remain_dismatch'])
        remain_match_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
                                                             ann_root=config['ann_root'], prompt=config['prompt'],
                                                             filename=config['remain_match'])
        unlearning_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                              filename=config['sensitive_val'])
        unlearning_aug_val_dataset = flickr_karpathy_caption_eval(transform_test, config['augmentation_root'],
                                                                  config['ann_root'],
                                                                  filename=config['sensitive_val'])
        val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'test')
        remain_val_dataset = flickr_karpathy_caption_eval(transform=transform_test, image_root=config['image_root'],
                                                          ann_root=config['ann_root'], filename=config['remain_val'])
        unlearning_part_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                               filename=config['sensitive_remain_val'])
        unlearning_part_aug_dataset = flickr_karpathy_caption_eval(transform_test, config['augmentation_root'],
                                                                   config['ann_root'],
                                                                   filename=config['sensitive_remain_val'])

        return unlearning_dataset, remain_dismatch_dataset, remain_match_dataset, unlearning_val_dataset, unlearning_aug_val_dataset, val_dataset, test_dataset, remain_val_dataset, unlearning_part_dataset, unlearning_part_aug_dataset

    elif dataset == 'caption_coco_mmu_part':
        unlearning_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
                                                           ann_root=config['ann_root'], prompt=config['prompt'],
                                                           filename=config['sensitive_train'])
        remain_dismatch_dataset = flickr_karpathy_caption_train(transform=transform_train,
                                                                image_root=config['image_root'],
                                                                ann_root=config['ann_root'], prompt=config['prompt'],
                                                                filename=config['remain_dismatch'])
        remain_match_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
                                                             ann_root=config['ann_root'], prompt=config['prompt'],
                                                             filename=config['remain_match'])
        unlearning_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                              filename=config['sensitive_val'])
        val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                   filename=config['val'])
        test_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                    filename=config['test'])
        remain_val_dataset = flickr_karpathy_caption_eval(transform=transform_test, image_root=config['image_root'],
                                                          ann_root=config['ann_root'], filename=config['remain_val'])
        unlearning_part_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                               filename=config['sensitive_remain_val'])
        return unlearning_dataset, remain_dismatch_dataset, remain_match_dataset, unlearning_val_dataset, val_dataset, test_dataset, remain_val_dataset, unlearning_part_dataset

    elif dataset == 'caption_coco_mmu_part_augmentation':
        unlearning_dataset = flickr_karpathy_caption_train(transform=transform_train,
                                                           image_root=config['augmentation_root'],
                                                           ann_root=config['ann_root'], prompt=config['prompt'],
                                                           filename=config['sensitive_train'])
        remain_dismatch_dataset = flickr_karpathy_caption_train(transform=transform_train,
                                                                image_root=config['image_root'],
                                                                ann_root=config['ann_root'], prompt=config['prompt'],
                                                                filename=config['remain_dismatch'])
        remain_match_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
                                                             ann_root=config['ann_root'], prompt=config['prompt'],
                                                             filename=config['remain_match'])
        unlearning_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                              filename=config['sensitive_val'])
        unlearning_aug_val_dataset = flickr_karpathy_caption_eval(transform_test, config['augmentation_root'],
                                                                  config['ann_root'],
                                                                  filename=config['sensitive_val'])
        val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                   filename=config['val'])
        test_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                    filename=config['test'])
        remain_val_dataset = flickr_karpathy_caption_eval(transform=transform_test, image_root=config['image_root'],
                                                          ann_root=config['ann_root'], filename=config['remain_val'])
        unlearning_part_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                               filename=config['sensitive_remain_val'])
        unlearning_part_aug_dataset = flickr_karpathy_caption_eval(transform_test, config['augmentation_root'],
                                                                   config['ann_root'],
                                                                   filename=config['sensitive_remain_val'])
        return unlearning_dataset, remain_dismatch_dataset, remain_match_dataset, unlearning_val_dataset, unlearning_aug_val_dataset, val_dataset, test_dataset, remain_val_dataset, unlearning_part_dataset, unlearning_part_aug_dataset

    elif dataset == 'caption_flickr_llmu':
        unlearning_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
                                                           ann_root=config['ann_root'], prompt=config['prompt'],
                                                           filename=config['sensitive_train'])
        dismatch_dataset = flickr_karpathy_caption_train(transform=transform_train,
                                                         image_root=config['image_root'],
                                                         ann_root=config['ann_root'], prompt=config['prompt'],
                                                         filename=config['dismatch'])
        match_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
                                                      ann_root=config['ann_root'], prompt=config['prompt'],
                                                      filename=config['match'])
        unlearning_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                              filename=config['sensitive_val'])
        val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'test')
        remain_val_dataset = flickr_karpathy_caption_eval(transform=transform_test, image_root=config['image_root'],
                                                          ann_root=config['ann_root'], filename=config['remain_val'])
        return unlearning_dataset, dismatch_dataset, match_dataset, unlearning_val_dataset, val_dataset, test_dataset, remain_val_dataset

    elif dataset == 'caption_coco_llmu':
        unlearning_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
                                                           ann_root=config['ann_root'], prompt=config['prompt'],
                                                           filename=config['sensitive_train'])
        dismatch_dataset = flickr_karpathy_caption_train(transform=transform_train,
                                                         image_root=config['image_root'],
                                                         ann_root=config['ann_root'], prompt=config['prompt'],
                                                         filename=config['dismatch'])
        match_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
                                                      ann_root=config['ann_root'], prompt=config['prompt'],
                                                      filename=config['match'])
        unlearning_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                              filename=config['sensitive_val'])
        val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                   filename=config['val'])
        test_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                    filename=config['test'])
        remain_val_dataset = flickr_karpathy_caption_eval(transform=transform_test, image_root=config['image_root'],
                                                          ann_root=config['ann_root'], filename=config['remain_val'])
        return unlearning_dataset, dismatch_dataset, match_dataset, unlearning_val_dataset, val_dataset, test_dataset, remain_val_dataset

    elif dataset == 'caption_flickr_emmu':
        # unlearning_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
        #                                                    ann_root=config['ann_root'], prompt=config['prompt'],
        #                                                    filename=config['sensitive_train'])
        unlearning_dataset = flickr_karpathy_caption_train_disturbance(transform=transform_train,
                                                                       image_root=config['image_root'],
                                                                       disturbance_root=config[
                                                                           'disturbance_image_root'],
                                                                       ann_root=config['ann_root'],
                                                                       filename=config['sensitive_train'])
        remain_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
                                                       ann_root=config['ann_root'], prompt=config['prompt'],
                                                       filename=config['remain'])
        unlearning_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                              filename=config['sensitive_val'])
        val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'test')
        remain_val_dataset = flickr_karpathy_caption_eval(transform=transform_test, image_root=config['image_root'],
                                                          ann_root=config['ann_root'], filename=config['remain_val'])
        # disturbance_val_dataset = flickr_karpathy_caption_eval(transform=transform_test, image_root=config['disturbance_image_root'],
        #                                            ann_root=config['ann_root'], filename=config['unlearning_val'])
        return unlearning_dataset, remain_dataset, unlearning_val_dataset, val_dataset, test_dataset, remain_val_dataset

    elif dataset == 'caption_flickr_emmu_part':
        # unlearning_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
        #                                                    ann_root=config['ann_root'], prompt=config['prompt'],
        #                                                    filename=config['unlearning'])
        unlearning_dataset = flickr_karpathy_caption_train_disturbance(transform=transform_train,
                                                                       image_root=config['image_root'],
                                                                       disturbance_root=config[
                                                                           'disturbance_image_root'],
                                                                       ann_root=config['ann_root'],
                                                                       filename=config['sensitive_train'])
        remain_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
                                                       ann_root=config['ann_root'], prompt=config['prompt'],
                                                       filename=config['remain'])
        unlearning_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                              filename=config['sensitive_val'])
        val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'test')
        remain_val_dataset = flickr_karpathy_caption_eval(transform=transform_test, image_root=config['image_root'],
                                                          ann_root=config['ann_root'], filename=config['remain_val'])
        unlearning_part_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                               filename=config['sensitive_remain_val'])

        return unlearning_dataset, remain_dataset, unlearning_val_dataset, val_dataset, test_dataset, remain_val_dataset, unlearning_part_dataset

    elif dataset == 'caption_flickr_emmu_part_augmentation':
        # unlearning_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
        #                                                    ann_root=config['ann_root'], prompt=config['prompt'],
        #                                                    filename=config['unlearning'])
        unlearning_dataset = flickr_karpathy_caption_train_disturbance(transform=transform_train,
                                                                       image_root=config['augmentation_root'],
                                                                       disturbance_root=config[
                                                                           'disturbance_image_root'],
                                                                       ann_root=config['ann_root'],
                                                                       filename=config['sensitive_train'])
        remain_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
                                                       ann_root=config['ann_root'], prompt=config['prompt'],
                                                       filename=config['remain'])
        unlearning_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                              filename=config['sensitive_val'])
        unlearning_aug_val_dataset = flickr_karpathy_caption_eval(transform_test, config['augmentation_root'],
                                                                  config['ann_root'],
                                                                  filename=config['sensitive_val'])
        val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'test')
        remain_val_dataset = flickr_karpathy_caption_eval(transform=transform_test, image_root=config['image_root'],
                                                          ann_root=config['ann_root'], filename=config['remain_val'])
        unlearning_part_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                               filename=config['sensitive_remain_val'])
        unlearning_part_aug_dataset = flickr_karpathy_caption_eval(transform_test, config['augmentation_root'],
                                                                   config['ann_root'],
                                                                   filename=config['sensitive_remain_val'])

        return unlearning_dataset, remain_dataset, unlearning_val_dataset, unlearning_aug_val_dataset, val_dataset, test_dataset, remain_val_dataset, unlearning_part_dataset, unlearning_part_aug_dataset

    elif dataset == 'caption_flickr_emmu_part_augmentation_composite':
        # unlearning_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
        #                                                    ann_root=config['ann_root'], prompt=config['prompt'],
        #                                                    filename=config['unlearning'])
        unlearning_dataset = flickr_karpathy_caption_train_disturbance(transform=transform_train,
                                                                       image_root=config['image_root'],
                                                                       disturbance_root=config[
                                                                           'disturbance_image_root'],
                                                                       ann_root=config['ann_root'],
                                                                       filename=config['sensitive_train'])
        augmentation_dataset = flickr_karpathy_caption_train_disturbance(transform=transform_train,
                                                                         image_root=config['augmentation_root'],
                                                                         disturbance_root=config[
                                                                             'disturbance_image_root'],
                                                                         ann_root=config['ann_root'],
                                                                         filename=config['sensitive_train'])
        remain_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
                                                       ann_root=config['ann_root'], prompt=config['prompt'],
                                                       filename=config['remain'])
        unlearning_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                              filename=config['sensitive_val'])
        val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'test')
        remain_val_dataset = flickr_karpathy_caption_eval(transform=transform_test, image_root=config['image_root'],
                                                          ann_root=config['ann_root'], filename=config['remain_val'])
        unlearning_part_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                               filename=config['sensitive_remain_val'])

        return unlearning_dataset, augmentation_dataset, remain_dataset, unlearning_val_dataset, val_dataset, test_dataset, remain_val_dataset, unlearning_part_dataset

    elif dataset == 'caption_coco_emmu':
        # unlearning_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
        #                                                    ann_root=config['ann_root'], prompt=config['prompt'],
        #                                                    filename=config['sensitive_train'])
        unlearning_dataset = flickr_karpathy_caption_train_disturbance(transform=transform_train,
                                                                       image_root=config['image_root'],
                                                                       disturbance_root=config[
                                                                           'disturbance_image_root'],
                                                                       ann_root=config['ann_root'],
                                                                       filename=config['sensitive_train'])
        remain_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
                                                       ann_root=config['ann_root'], prompt=config['prompt'],
                                                       filename=config['remain'])
        unlearning_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                              filename=config['sensitive_val'])
        val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                   filename=config['val'])
        test_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                    filename=config['test'])
        remain_val_dataset = flickr_karpathy_caption_eval(transform=transform_test, image_root=config['image_root'],
                                                          ann_root=config['ann_root'], filename=config['remain_val'])
        # disturbance_val_dataset = flickr_karpathy_caption_eval(transform=transform_test, image_root=config['disturbance_image_root'],
        #                                            ann_root=config['ann_root'], filename=config['unlearning_val'])
        return unlearning_dataset, remain_dataset, unlearning_val_dataset, val_dataset, test_dataset, remain_val_dataset

    elif dataset == 'caption_coco_emmu_part':
        unlearning_dataset = flickr_karpathy_caption_train_disturbance(transform=transform_train,
                                                                       image_root=config['image_root'],
                                                                       disturbance_root=config[
                                                                           'disturbance_image_root'],
                                                                       ann_root=config['ann_root'],
                                                                       filename=config['sensitive_train'])
        remain_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
                                                       ann_root=config['ann_root'], prompt=config['prompt'],
                                                       filename=config['remain'])
        unlearning_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                              filename=config['sensitive_val'])
        val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                   filename=config['val'])
        test_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                    filename=config['test'])
        remain_val_dataset = flickr_karpathy_caption_eval(transform=transform_test, image_root=config['image_root'],
                                                          ann_root=config['ann_root'], filename=config['remain_val'])
        # disturbance_val_dataset = flickr_karpathy_caption_eval(transform=transform_test, image_root=config['disturbance_image_root'],
        #                                            ann_root=config['ann_root'], filename=config['unlearning_val'])
        unlearning_part_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                               filename=config['sensitive_remain_val'])
        return unlearning_dataset, remain_dataset, unlearning_val_dataset, val_dataset, test_dataset, remain_val_dataset, unlearning_part_dataset

    elif dataset == 'caption_coco_emmu_part_augmentation':
        unlearning_dataset = flickr_karpathy_caption_train_disturbance(transform=transform_train,
                                                                       image_root=config['augmentation_root'],
                                                                       disturbance_root=config[
                                                                           'disturbance_image_root'],
                                                                       ann_root=config['ann_root'],
                                                                       filename=config['sensitive_train'])
        remain_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
                                                       ann_root=config['ann_root'], prompt=config['prompt'],
                                                       filename=config['remain'])
        unlearning_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                              filename=config['sensitive_val'])
        unlearning_aug_val_dataset = flickr_karpathy_caption_eval(transform_test, config['augmentation_root'],
                                                                  config['ann_root'],
                                                                  filename=config['sensitive_val'])
        val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                   filename=config['val'])
        test_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                    filename=config['test'])
        remain_val_dataset = flickr_karpathy_caption_eval(transform=transform_test, image_root=config['image_root'],
                                                          ann_root=config['ann_root'], filename=config['remain_val'])
        # disturbance_val_dataset = flickr_karpathy_caption_eval(transform=transform_test, image_root=config['disturbance_image_root'],
        #                                            ann_root=config['ann_root'], filename=config['unlearning_val'])
        unlearning_part_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                               filename=config['sensitive_remain_val'])
        unlearning_part_aug_dataset = flickr_karpathy_caption_eval(transform_test, config['augmentation_root'],
                                                                   config['ann_root'],
                                                                   filename=config['sensitive_remain_val'])
        return unlearning_dataset, remain_dataset, unlearning_val_dataset, unlearning_aug_val_dataset, val_dataset, test_dataset, remain_val_dataset, unlearning_part_dataset, unlearning_part_aug_dataset

    elif dataset == 'caption_coco_emmu_part_augmentation_composite':
        unlearning_dataset = flickr_karpathy_caption_train_disturbance(transform=transform_train,
                                                                       image_root=config['image_root'],
                                                                       disturbance_root=config[
                                                                           'disturbance_image_root'],
                                                                       ann_root=config['ann_root'],
                                                                       filename=config['sensitive_train'])
        augmentation_dataset = flickr_karpathy_caption_train_disturbance(transform=transform_train,
                                                                         image_root=config['augmentation_root'],
                                                                         disturbance_root=config[
                                                                             'disturbance_image_root'],
                                                                         ann_root=config['ann_root'],
                                                                         filename=config['sensitive_train'])
        remain_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
                                                       ann_root=config['ann_root'], prompt=config['prompt'],
                                                       filename=config['remain'])
        unlearning_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                              filename=config['sensitive_val'])
        val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                   filename=config['val'])
        test_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                    filename=config['test'])
        remain_val_dataset = flickr_karpathy_caption_eval(transform=transform_test, image_root=config['image_root'],
                                                          ann_root=config['ann_root'], filename=config['remain_val'])
        # disturbance_val_dataset = flickr_karpathy_caption_eval(transform=transform_test, image_root=config['disturbance_image_root'],
        #                                            ann_root=config['ann_root'], filename=config['unlearning_val'])
        unlearning_part_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                               filename=config['sensitive_remain_val'])
        return unlearning_dataset, augmentation_dataset, remain_dataset, unlearning_val_dataset, val_dataset, test_dataset, remain_val_dataset, unlearning_part_dataset

    elif dataset == 'caption_flickr_salun':
        unlearning_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
                                                           ann_root=config['ann_root'], prompt=config['prompt'],
                                                           filename=config['sensitive_train'])
        dismatch_dataset = flickr_karpathy_caption_train(transform=transform_train,
                                                         image_root=config['image_root'],
                                                         ann_root=config['ann_root'], prompt=config['prompt'],
                                                         filename=config['dismatch'])
        match_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
                                                      ann_root=config['ann_root'], prompt=config['prompt'],
                                                      filename=config['match'])
        unlearning_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                              filename=config['sensitive_val'])
        val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'test')
        remain_val_dataset = flickr_karpathy_caption_eval(transform=transform_test, image_root=config['image_root'],
                                                          ann_root=config['ann_root'], filename=config['remain_val'])
        return unlearning_dataset, dismatch_dataset, match_dataset, unlearning_val_dataset, val_dataset, test_dataset, remain_val_dataset

    elif dataset == 'caption_flickr_salun_part':
        unlearning_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
                                                           ann_root=config['ann_root'], prompt=config['prompt'],
                                                           filename=config['sensitive_train'])
        dismatch_dataset = flickr_karpathy_caption_train(transform=transform_train,
                                                         image_root=config['image_root'],
                                                         ann_root=config['ann_root'], prompt=config['prompt'],
                                                         filename=config['dismatch'])
        match_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
                                                      ann_root=config['ann_root'], prompt=config['prompt'],
                                                      filename=config['match'])
        unlearning_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                              filename=config['sensitive_val'])
        val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'test')
        remain_val_dataset = flickr_karpathy_caption_eval(transform=transform_test, image_root=config['image_root'],
                                                          ann_root=config['ann_root'], filename=config['remain_val'])
        unlearning_part_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                               filename=config['sensitive_remain_val'])

        return unlearning_dataset, dismatch_dataset, match_dataset, unlearning_val_dataset, val_dataset, test_dataset, remain_val_dataset, unlearning_part_dataset

    elif dataset == 'caption_flickr_salun_part_augmentation':
        unlearning_dataset = flickr_karpathy_caption_train(transform=transform_train,
                                                           image_root=config['augmentation_root'],
                                                           ann_root=config['ann_root'], prompt=config['prompt'],
                                                           filename=config['sensitive_train'])
        dismatch_dataset = flickr_karpathy_caption_train(transform=transform_train,
                                                         image_root=config['image_root'],
                                                         ann_root=config['ann_root'], prompt=config['prompt'],
                                                         filename=config['dismatch'])
        match_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
                                                      ann_root=config['ann_root'], prompt=config['prompt'],
                                                      filename=config['match'])
        unlearning_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                              filename=config['sensitive_val'])
        unlearning_aug_val_dataset = flickr_karpathy_caption_eval(transform_test, config['augmentation_root'],
                                                                  config['ann_root'],
                                                                  filename=config['sensitive_val'])
        val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'test')
        remain_val_dataset = flickr_karpathy_caption_eval(transform=transform_test, image_root=config['image_root'],
                                                          ann_root=config['ann_root'], filename=config['remain_val'])
        unlearning_part_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                               filename=config['sensitive_remain_val'])
        unlearning_part_aug_dataset = flickr_karpathy_caption_eval(transform_test, config['augmentation_root'],
                                                                   config['ann_root'],
                                                                   filename=config['sensitive_remain_val'])

        return unlearning_dataset, dismatch_dataset, match_dataset, unlearning_val_dataset, unlearning_aug_val_dataset, val_dataset, test_dataset, remain_val_dataset, unlearning_part_dataset, unlearning_part_aug_dataset

    elif dataset == 'caption_coco_salun':
        unlearning_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
                                                           ann_root=config['ann_root'], prompt=config['prompt'],
                                                           filename=config['sensitive_train'])
        dismatch_dataset = flickr_karpathy_caption_train(transform=transform_train,
                                                         image_root=config['image_root'],
                                                         ann_root=config['ann_root'], prompt=config['prompt'],
                                                         filename=config['dismatch'])
        match_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
                                                      ann_root=config['ann_root'], prompt=config['prompt'],
                                                      filename=config['match'])
        unlearning_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                              filename=config['sensitive_val'])
        val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                   filename=config['val'])
        test_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                    filename=config['test'])
        remain_val_dataset = flickr_karpathy_caption_eval(transform=transform_test, image_root=config['image_root'],
                                                          ann_root=config['ann_root'], filename=config['remain_val'])
        return unlearning_dataset, dismatch_dataset, match_dataset, unlearning_val_dataset, val_dataset, test_dataset, remain_val_dataset

    elif dataset == 'caption_coco_salun_part':
        unlearning_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
                                                           ann_root=config['ann_root'], prompt=config['prompt'],
                                                           filename=config['sensitive_train'])
        dismatch_dataset = flickr_karpathy_caption_train(transform=transform_train,
                                                         image_root=config['image_root'],
                                                         ann_root=config['ann_root'], prompt=config['prompt'],
                                                         filename=config['dismatch'])
        match_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
                                                      ann_root=config['ann_root'], prompt=config['prompt'],
                                                      filename=config['match'])
        unlearning_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                              filename=config['sensitive_val'])
        val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                   filename=config['val'])
        test_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                    filename=config['test'])
        remain_val_dataset = flickr_karpathy_caption_eval(transform=transform_test, image_root=config['image_root'],
                                                          ann_root=config['ann_root'], filename=config['remain_val'])
        unlearning_part_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                               filename=config['sensitive_remain_val'])

        return unlearning_dataset, dismatch_dataset, match_dataset, unlearning_val_dataset, val_dataset, test_dataset, remain_val_dataset, unlearning_part_dataset

    elif dataset == 'caption_coco_salun_part_augmentation':
        unlearning_dataset = flickr_karpathy_caption_train(transform=transform_train,
                                                           image_root=config['augmentation_root'],
                                                           ann_root=config['ann_root'], prompt=config['prompt'],
                                                           filename=config['sensitive_train'])
        dismatch_dataset = flickr_karpathy_caption_train(transform=transform_train,
                                                         image_root=config['image_root'],
                                                         ann_root=config['ann_root'], prompt=config['prompt'],
                                                         filename=config['dismatch'])
        match_dataset = flickr_karpathy_caption_train(transform=transform_train, image_root=config['image_root'],
                                                      ann_root=config['ann_root'], prompt=config['prompt'],
                                                      filename=config['match'])
        unlearning_val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                              filename=config['sensitive_val'])
        unlearning_aug_val_dataset = flickr_karpathy_caption_eval(transform_test, config['augmentation_root'],
                                                                  config['ann_root'],
                                                                  filename=config['sensitive_val'])
        val_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                   filename=config['val'])
        test_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                    filename=config['test'])
        remain_val_dataset = flickr_karpathy_caption_eval(transform=transform_test, image_root=config['image_root'],
                                                          ann_root=config['ann_root'], filename=config['remain_val'])
        unlearning_part_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'],
                                                               filename=config['sensitive_remain_val'])
        unlearning_part_aug_dataset = flickr_karpathy_caption_eval(transform_test, config['augmentation_root'], config['ann_root'],
                                                               filename=config['sensitive_remain_val'])

        return unlearning_dataset, dismatch_dataset, match_dataset, unlearning_val_dataset, unlearning_aug_val_dataset, val_dataset, test_dataset, remain_val_dataset, unlearning_part_dataset, unlearning_part_aug_dataset

    elif dataset == 'caption_flickr_auc':
        unlearning_dataset = flickr_karpathy_caption_train(transform_train, config['image_root'], config['ann_root'],
                                                           filename=config['unlearning'])
        remain_dataset = flickr_karpathy_caption_train(transform_train, config['image_root'], config['ann_root'],
                                                       filename=config['remain'])
        return unlearning_dataset, remain_dataset

    elif dataset == 'caption_flickr_mia':
        pos_dataset = flickr_karpathy_caption_train(transform_train, config['image_root'], config['ann_root'],
                                                    filename=config['positive'])
        neg_dataset = flickr_karpathy_caption_train(transform_train, config['image_root'], config['ann_root'],
                                                    filename=config['negative'])
        unlearning_dataset = flickr_karpathy_caption_train(transform_train, config['image_root'], config['ann_root'],
                                                           filename=config['unlearning'])
        return pos_dataset, neg_dataset, unlearning_dataset

    elif dataset == 'caption_flickr_unlearning_val':
        unlearning_val_1k_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'],
                                                                 config['ann_root'], split=None,
                                                                 filename=config['unlearning_val_1k'])
        unlearning_val_2k_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'],
                                                                 config['ann_root'], split=None,
                                                                 filename=config['unlearning_val_2k'])
        unlearning_val_3k_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'],
                                                                 config['ann_root'], split=None,
                                                                 filename=config['unlearning_val_3k'])
        unlearning_val_4k_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'],
                                                                 config['ann_root'], split=None,
                                                                 filename=config['unlearning_val_4k'])
        unlearning_val_5k_dataset = flickr_karpathy_caption_eval(transform_test, config['image_root'],
                                                                 config['ann_root'], split=None,
                                                                 filename=config['unlearning_val_5k'])
        return unlearning_val_1k_dataset, unlearning_val_2k_dataset, unlearning_val_3k_dataset, unlearning_val_4k_dataset, unlearning_val_5k_dataset

    elif dataset == 'nocaps':
        val_dataset = nocaps_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = nocaps_eval(transform_test, config['image_root'], config['ann_root'], 'test')
        return val_dataset, test_dataset

    # elif dataset == 'retrieval_':
    #     train_dataset = coco_karpathy_train(transform_train, config['image_root'], config['ann_root'])
    #     val_dataset = coco_karpathy_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'val')
    #     test_dataset = coco_karpathy_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'test')
    #     return train_dataset, val_dataset, test_dataset

    elif dataset == 'retrieval_flickr':
        train_dataset = flickr_karpathy_caption_train(transform_train, config['image_root'], config['ann_root'])
        val_dataset = flickr_karpathy_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = flickr_karpathy_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'test')
        return train_dataset, val_dataset, test_dataset

    elif dataset == 'retrieval_flickr_mmu':
        unlearning_dataset = flickr_karpathy_caption_train(transform_train, config['image_root'], config['ann_root'],
                                                           filename=config['unlearning'])
        remain_dismatch_dataset = flickr_karpathy_caption_train(transform_train, config['image_root'],
                                                                config['ann_root'],
                                                                filename=config['remain_dismatch'])
        remain_match_dataset = flickr_karpathy_caption_train(transform_train, config['image_root'], config['ann_root'],
                                                             filename=config['remain_match'])
        unlearning_val_dataset = flickr_karpathy_retrieval_eval(transform_test, config['image_root'],
                                                                config['ann_root'],
                                                                split=None, filename=config['unlearning_val'])
        val_dataset = flickr_karpathy_retrieval_eval(transform_test, config['image_root'], config['val_root'], 'val')
        test_dataset = flickr_karpathy_retrieval_eval(transform_test, config['image_root'], config['val_root'], 'test')
        return unlearning_dataset, remain_dismatch_dataset, remain_match_dataset, unlearning_val_dataset, val_dataset, test_dataset

    elif dataset == 'retrieval_flickr_emmu':
        unlearning_dataset = flickr_karpathy_caption_train(transform_train, config['image_root'], config['ann_root'],
                                                           filename=config['unlearning'])
        remain_match_dataset = flickr_karpathy_caption_train(transform_train, config['image_root'], config['ann_root'],
                                                             filename=config['remain_match'])
        unlearning_val_dataset = flickr_karpathy_retrieval_eval(transform_test, config['image_root'],
                                                                config['ann_root'],
                                                                split=None, filename=config['unlearning_val'])
        val_dataset = flickr_karpathy_retrieval_eval(transform_test, config['image_root'], config['val_root'], 'val')
        test_dataset = flickr_karpathy_retrieval_eval(transform_test, config['image_root'], config['val_root'], 'test')
        return unlearning_dataset, remain_match_dataset, unlearning_val_dataset, val_dataset, test_dataset

    elif dataset == 'retrieval_flickr_emmu1':
        unlearning_dataset = flickr_karpathy_caption_train(transform_train, config['unlearning_image_root'],
                                                           config['unlearning_ann_root'],
                                                           filename=config['unlearning'])
        generalization_dataset = flickr_karpathy_caption_train(transform_train, config['generalization_image_root'],
                                                               config['generalization_ann_root'],
                                                               filename=config['generalization'])
        unlearning_val_dataset = flickr_karpathy_retrieval_eval(transform_test, config['unlearning_image_root'],
                                                                config['unlearning_ann_root'],
                                                                split=None, filename=config['unlearning_val'])
        val_dataset = flickr_karpathy_retrieval_eval(transform_test, config['unlearning_image_root'],
                                                     config['val_root'], 'val')
        test_dataset = flickr_karpathy_retrieval_eval(transform_test, config['unlearning_image_root'],
                                                      config['val_root'], 'test')
        return unlearning_dataset, generalization_dataset, unlearning_val_dataset, val_dataset, test_dataset

    elif dataset == 'retrieval_flickr_retrain':
        train_dataset = flickr_karpathy_caption_train(transform_train, config['image_root'], config['ann_root'],
                                                      filename=config['retrain'])
        val_dataset = flickr_karpathy_retrieval_eval(transform_test, config['image_root'], config['val_root'], 'val')
        test_dataset = flickr_karpathy_retrieval_eval(transform_test, config['image_root'], config['val_root'], 'test')
        unlearning_dataset = flickr_karpathy_retrieval_eval(transform_test, config['image_root'], config['ann_root'],
                                                            split=None, filename=config['unlearning_val'])
        return train_dataset, val_dataset, test_dataset, unlearning_dataset

    elif dataset == 'retrieval_flickr_unlearning_val':
        unlearning_val_1k_dataset = flickr_karpathy_retrieval_eval(transform_test, config['image_root'],
                                                                   config['ann_root'], split=None,
                                                                   filename=config['unlearning_val_1k'])
        unlearning_val_2k_dataset = flickr_karpathy_retrieval_eval(transform_test, config['image_root'],
                                                                   config['ann_root'], split=None,
                                                                   filename=config['unlearning_val_2k'])
        unlearning_val_3k_dataset = flickr_karpathy_retrieval_eval(transform_test, config['image_root'],
                                                                   config['ann_root'], split=None,
                                                                   filename=config['unlearning_val_3k'])
        unlearning_val_4k_dataset = flickr_karpathy_retrieval_eval(transform_test, config['image_root'],
                                                                   config['ann_root'], split=None,
                                                                   filename=config['unlearning_val_4k'])
        unlearning_val_5k_dataset = flickr_karpathy_retrieval_eval(transform_test, config['image_root'],
                                                                   config['ann_root'], split=None,
                                                                   filename=config['unlearning_val_5k'])
        return unlearning_val_1k_dataset, unlearning_val_2k_dataset, unlearning_val_3k_dataset, unlearning_val_4k_dataset, unlearning_val_5k_dataset


    elif dataset == 'retrieval_flickr_test':
        test_dataset = flickr_karpathy_caption_train_event(transform_train, config['image_root'], config['ann_root'],
                                                           filename=config['test'])
        return test_dataset

    elif dataset == 'vqa':
        train_dataset = vqa_dataset(transform_train, config['ann_root'], config['vqa_root'], config['vg_root'],
                                    train_files=config['train_files'], split='train')
        test_dataset = vqa_dataset(transform_test, config['ann_root'], config['vqa_root'], config['vg_root'],
                                   split='test')
        return train_dataset, test_dataset

    elif dataset == 'nlvr':
        train_dataset = nlvr_dataset(transform_train, config['image_root'], config['ann_root'], 'train')
        val_dataset = nlvr_dataset(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = nlvr_dataset(transform_test, config['image_root'], config['ann_root'], 'test')
        return train_dataset, val_dataset, test_dataset


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                      shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders


def custom_collate_fn(batch):
    # Unzip the batch
    images, disturbance_images, captions, ids, scenarios = zip(*batch)
    # Check if all scenarios are the same
    if len(set(scenarios)) > 1:
        raise ValueError("Scenarios in a batch are not the same.")
    # Stack images and disturbance_images into tensors
    images = torch.stack(images, 0)
    disturbance_images = torch.stack(disturbance_images, 0)
    # Return the batch
    return images, disturbance_images, captions, ids, scenarios[0]
