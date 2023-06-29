# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch

# from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding

# from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
#                            make_exp_dirs)
# from basicsr.utils.options import dict2str

from glob import glob

from natsort import natsorted

def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)

    img_path = opt['img_path'].get('input_img')

    img_folder = '/content/dirve/MyDrive/test_noisy_folder/'

    out_folder = '/content/drive/MyDrive/HINetResult/denoise/'

    os.makedirs(out_folder, exist_ok=True)

    files = natsorted(glob(os.path.join(img_folder, '*.tiff')))

    for _file in files:
        
        file_client = FileClient('disk')

        img_bytes = file_client.get(img_path, None)
        try:
            img = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("path {} not working".format(img_path))

        img = img2tensor(img, bgr2rgb=True, float32=True)
        
        model = create_model(opt)

        output_path = os.path.join(out_folder,
                                         'result-' + 
                                         os.path.split(_file)[-1])
        
        model.single_image_inference(img, output_path)

    
          
        

    '''
    output_path = opt['img_path'].get('output_img')


    ## 1. read image
    file_client = FileClient('disk')

    img_bytes = file_client.get(img_path, None)
    try:
        img = imfrombytes(img_bytes, float32=True)
    except:
        raise Exception("path {} not working".format(img_path))

    img = img2tensor(img, bgr2rgb=True, float32=True)



    ## 2. run inference
    model = create_model(opt)
    model.single_image_inference(img, output_path)

    print('inference {} .. finished.'.format(img_path))
    '''

if __name__ == '__main__':
    main()
