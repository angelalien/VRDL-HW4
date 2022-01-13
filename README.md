# VRDL_HW4

Code for Selected Topics in Visual Recognition using Deep Learning(2021 Autumn NYCU) 
Homework4: Image super resolution.  
I use [super-image](https://eugenesiow.github.io/super-image/) library to handle this assignment.

## Requirements

I use Google Colab as the environment.
To install and import the required library and tools, run the command on Google Colab:

```setup
!pip install super-image

import os
import shutil
from google_drive_downloader import GoogleDriveDownloader as gdd
from google.colab import files
from PIL import Image
from torchvision.transforms import transforms
from pathlib import Path
from super_image.data import EvalDataset, TrainDataset
from super_image import Trainer, TrainingArguments, EdsrModel, EdsrConfig
from super_image import ImageLoader
```
> This command has been included in train.ipynb file.
## Dataset Preparation

To download the training dataset, run this command in the Google Colab:
```
gdd.download_file_from_google_drive(file_id='1GL_Rh1N-WjrvF_-YOKOyvq0zrV6TF4hb',
                                      dest_path='./dataset/HW_data.zip',
                                      unzip=True)
```
Then do the data pre-processing method for data augmentation.
> This command and pre-processing method have been included in train.ipynb file.

## Training 

To train the model, run the train.ipynb file on Google Colab.

> This process has been included in train.ipynb file.

After running, you will get some output files: config.json, pytorch_model_3x.pt. The files structure is:
```
results
  +- config.json
  +- pytorch_model_3x.pt
```

## Pre-trained Models

You can download my pretrained models (trained on given high resolution images dataset) here:

- [My EDSR model weight](https://drive.google.com/file/d/10edK-zM_1l7e2Fvf3FR-UFw5cbKeEXba/view?usp=sharing)
- [My EDSR model config](https://drive.google.com/file/d/1JVDZbANgLu_Oo85iSP22bRrULDS87pis/view?usp=sharing)
  

Model's hyperparameter setting:

- n_resblocks = 32 (number of residual blocks)
- n_feats = 256 (number of filters)
- res_scale = 0.1 (residual scaling)
- patch_size = 12 (the size I set for randomly crop the image)
- num_train_epochs = 300




## Make Submission

To make the submission file, run the [inference.ipynb](https://colab.research.google.com/drive/1yyMStT51een3V101005Eg3yxcJ8MEtu2?usp=sharing).
After running, you will get an 'output' file including predicted images for submission.

## Result

My model achieves the following performance on CodaLab:
| Model name  | Top 1 PSNR   |
| ----------- |------------- |
| ESDR        |    28.1565   |


## Reference
- https://eugenesiow.github.io/super-image
- https://github.com/2KangHo/vdsr_pytorch
- https://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.pdf
