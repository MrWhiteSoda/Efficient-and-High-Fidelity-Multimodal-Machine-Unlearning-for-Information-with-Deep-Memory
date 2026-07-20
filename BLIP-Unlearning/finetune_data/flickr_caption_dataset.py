import os
import json

from torch.utils.data import Dataset

from PIL import Image

from finetune_data.utils import pre_caption


class flickr_karpathy_caption_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt='', filename=None):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''
        if filename is None:
            filename = 'flickr30k_train.json'
        # print(filename)
        self.annotation = json.load(open(os.path.join(ann_root, filename), 'r'))[320:]  # 加载json文件
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.image_root, ann['image'])
        try:
            image = Image.open(image_path).convert('RGB')
        except OSError as e:
            print(f"Failed to open image: {image_path} with error {e}")
            raise
        # image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        caption = self.prompt + pre_caption(ann['caption'], self.max_words)
        return image, caption, int(ann['image_id'])


class flickr_karpathy_caption_train_disturbance(Dataset):
    def __init__(self, transform, image_root, disturbance_root, ann_root, max_words=30, prompt='', filename=None):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''
        if filename is None:
            filename = 'flickr30k_train.json'

        self.annotation = json.load(open(os.path.join(ann_root, filename), 'r'))[320:]  # 加载json文件
        self.transform = transform
        self.image_root = image_root
        self.disturbance_root = disturbance_root
        self.max_words = max_words
        self.prompt = prompt

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        disturbance_path = os.path.join(self.disturbance_root, ann['image'])
        disturbance_image = Image.open(disturbance_path).convert('RGB')
        disturbance_image = self.transform(disturbance_image)
        caption = self.prompt + pre_caption(ann['caption'], self.max_words)

        return image, disturbance_image, caption, int(ann['image_id'])

class flickr_karpathy_caption_train_event(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt='', filename=None):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''
        if filename is None:
            filename = 'flickr30k_train.json'

        self.annotation = json.load(open(os.path.join(ann_root, filename), 'r'))  # 加载json文件
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        caption = self.prompt + pre_caption(ann['caption'], self.max_words)
        new_caption = self.prompt + pre_caption(ann['new_caption'], self.max_words)
        event = ann['class']
        return image, caption, int(ann['image_id']), new_caption, event


class flickr_karpathy_caption_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split=None, filename=None):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        filenames = {'val': 'flickr30k_val.json', 'test': 'flickr30k_test.json'}
        if filename is None:
            self.annotation = json.load(open(os.path.join(ann_root, filenames[split]), 'r'))
        else:
            self.annotation = json.load(open(os.path.join(ann_root, filename), 'r'))
        self.transform = transform
        self.image_root = image_root

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        # try:
        #     image = Image.open(image_path).convert('RGB')
        # except OSError as e:
        #     print(f"Failed to open image: {image_path} with error {e}")
        #     raise
        image = self.transform(image)
        img_id = ann['image_id']
        return image, int(img_id)


class flickr_karpathy_retrieval_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=30, filename=None):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        filenames = {'val': 'flickr30k_val.json', 'test': 'flickr30k_test.json'}
        if filename is None:
            self.annotation = json.load(open(os.path.join(ann_root, filenames[split]), 'r'))
        else:
            self.annotation = json.load(open(os.path.join(ann_root, filename), 'r'))
        self.transform = transform
        self.image_root = image_root

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption, max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.annotation[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index
