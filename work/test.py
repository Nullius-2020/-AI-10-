import argparse
import glob
import collections.abc
import os.path
from transforms import Normalize, Resize
from PIL import Image
from transforms import Compose, Compose_test
import paddle
import paddle.nn as nn
import cv2
import numpy as np
from src.config import Config
from main import set_model_log_output_dir
from src.MRTR import MRTR

import time
import os
def load_config(mode=None):
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    # train mode
    if mode == 1:
        # create checkpoints path if does't exist
        if not os.path.exists(config.LOG_DIR):
            os.makedirs(config.LOG_DIR)

        if not os.path.exists(config.MODEL_DIR):
            os.makedirs(config.MODEL_DIR)

        config.MODE = 1
        if args.model:
            config.MODEL = args.model

    # test mode
    elif mode == 2:
        config.MODE = 2
        #config.MODEL = args.model if args.model is not None else 3
        # Hack
        config.INPUT_SIZE = 0
        #
        config._dict['WORD_BB_PERCENT_THRESHOLD'] = 0
        config._dict['CHAR_BB_PERCENT_THRESHOLD'] = 0
        config._dict['MASK_CORNER_OFFSET'] = 5

        # TODO: update this part
        if args.input is not None:
            config.TEST_FLIST = args.input

        if args.mask is not None:
            config.TEST_MASK_FLIST = args.mask

        if args.edge is not None:
            config.TEST_EDGE_FLIST = args.edge

        if args.output is not None:
            config.RESULTS = args.output

    # eval mode
    elif mode == 3:
        config.MODE = 3
        config.MODEL = args.model if args.model is not None else 3

    return config

class Dataset_test(paddle.io.Dataset):
    def __init__(self, dataset_root=None, transforms=None):
        if dataset_root is None:
            raise ValueError("dataset_root is None")
        self.dataset_root = dataset_root
        self.transforms = Compose_test(transforms)
        self.input_img = glob.glob(os.path.join(self.dataset_root, "*.jpg"))

        self.input_img.sort()

 

    def __getitem__(self, index):
        input_path = self.input_img[index]  
        input, h, w= self.transforms(input_path)
        
        return input, h, w, input_path

    def __len__(self):
        return len(self.input_img)


def parse_args():
    parser = argparse.ArgumentParser(description='Model testing')

    parser.add_argument(
            '--config', type=str, 
            default='./config/config.yml', 
            help='model config file')

    parser.add_argument('--model', type=int, 
            choices=[1, 2, 3, 4], 
            help='1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model')

    parser.add_argument(
        '--dataset_root',
        dest='dataset_root',
        help='The path of dataset root',
        type=str,
        default=None)

    parser.add_argument(
        '--pretrained',
        dest='pretrained',
        help='The pretrained of model',
        type=str,
        default=None)

    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='batch_size',
        type=int,
        default=1
    )

    parser.add_argument(
        '--save_path',
        dest='save_path',
        help='save_path',
        type=str,
        default='test_result'
    )

    return parser.parse_args()

def slide(model, im, crop_size, stride):
    """
    Infer by sliding window.

    Args:
        model (paddle.nn.Layer): model to get logits of image.
        im (Tensor): the input image.
        crop_size (tuple|list). The size of sliding window, (w, h).
        stride (tuple|list). The size of stride, (w, h).

    Return:
        Tensor: The logit of input image.
    """
    h_im, w_im = im.shape[-2:]
    w_crop, h_crop = crop_size
    w_stride, h_stride = stride
    # calculate the crop nums
    rows = np.int(np.ceil(1.0 * (h_im - h_crop) / h_stride)) + 1
    cols = np.int(np.ceil(1.0 * (w_im - w_crop) / w_stride)) + 1
    # prevent negative sliding rounds when imgs after scaling << crop_size
    rows = 1 if h_im <= h_crop else rows
    cols = 1 if w_im <= w_crop else cols
    # TODO 'Tensor' object does not support item assignment. If support, use tensor to calculation.
    final_logit = None
    count = np.zeros([1, 1, h_im, w_im])
    mask = np.ones(crop_size)
    mask=paddle.to_tensor(mask,dtype='float32')
    mask = mask.unsqueeze(0).unsqueeze(0)
    for r in range(rows):
        for c in range(cols):
            h1 = r * h_stride
            w1 = c * w_stride
            h2 = min(h1 + h_crop, h_im)
            w2 = min(w1 + w_crop, w_im)
            h1 = max(h2 - h_crop, 0)
            w1 = max(w2 - w_crop, 0)
            im_crop = im[:, :, h1:h2, w1:w2]
            logits,_,_ = model(im_crop,mask)
            logit = logits.numpy()
            if final_logit is None:
                final_logit = np.zeros([1, logit.shape[1], h_im, w_im])
            final_logit[:, :, h1:h2, w1:w2] += logit[:, :, :h2 - h1, :w2 - w1]
            count[:, :, h1:h2, w1:w2] += 1
    if np.sum(count == 0) != 0:
        raise RuntimeError(
            'There are pixel not predicted. It is possible that stride is greater than crop_size'
        )
    final_logit = final_logit / count
    final_logit = paddle.to_tensor(final_logit)
    return final_logit

def main(args):
    
    config = Config(args.config)
    config.MODE = 2
    config.INPUT_SIZE = 0
        #
    config._dict['WORD_BB_PERCENT_THRESHOLD'] = 0
    config._dict['CHAR_BB_PERCENT_THRESHOLD'] = 0
    config._dict['MASK_CORNER_OFFSET'] = 5
    config.G_MODEL_PATH= args.pretrained
    config = set_model_log_output_dir(config)
    # As it is a small file, I saved in both log and model directory
    ## TODO save as yml
    config.save(config.CONFIG_DIR)
    model = MRTR(config)
    model.maskpreinpaint_model.load()

    transforms = [
        #Resize(target_size=(4096, 4096))
    ]
    dataset = Dataset_test(dataset_root=args.dataset_root,transforms=transforms)
    dataloader = paddle.io.DataLoader(dataset, 
                                    batch_size = args.batch_size, 
                                    num_workers = 0,
                                    shuffle = False,
                                    return_list = True)
    model.maskpreinpaint_model.eval()
    save_path = args.save_path
    if not os.path.exists(save_path):
            os.makedirs(save_path)

    # inference
    start = time.time()
    for i, (img, h, w, path) in enumerate(dataloader):
        print(path[0],h.numpy(),w.numpy())
        name= os.path.join(save_path,path[0].split('/')[-1].replace('jpg','png'))
        path = name
        #mask =  np.ones([2048, 2048])
        img=paddle.to_tensor(img/255,dtype='float32')
        #mask=paddle.to_tensor(mask,dtype='float32')
        #mask = mask.unsqueeze(0).unsqueeze(0)
        #print(mask,img)
        #output_images, output_pre_images, output_masks = model.maskpreinpaint_model(img, mask)
        output_images=slide(model.maskpreinpaint_model,img,[640, 640],[600,600])
        #print(output_images.shape, output_pre_images.shape, output_masks.shape)
        img_out = nn.functional.interpolate(output_images, size = [h,w], mode = 'nearest')
        #print(img_out)
        img_out = img_out.squeeze(0)
        
        img_out = paddle.clip(img_out* 255.0, 0, 255)
        img_out = paddle.transpose(img_out, [1,2,0])
        img_out = np.uint8(img_out)  
        img_out=Image.fromarray(img_out)
        img_out.save(path)

    end = time.time()
    time_one = (end - start)/i
    print('The running time of an image is : {:2f} s'.format(time_one))



if __name__=='__main__':
    args = parse_args()
    main(args)