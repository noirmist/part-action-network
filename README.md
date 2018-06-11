# Part Action Network Tensorflow Implementation.

This project made by tensorflow which descibes the Part Action Network proposed in the paper in ICCV2017: Single Image Action Recognition using Semantic Part Actions, Zhichen Zhao, Huimin Ma and Shaodi You.

In general, the main purpose of this paper is to capture "part action" cues to improve the body action recognition. We view a body action as a combination of several part actions.
![](https://github.com/ZhichenZhao/part-action-network/raw/master/imgs/framework.jpg)
Some part actions are shown as follows:
![](https://github.com/ZhichenZhao/part-action-network/raw/master/imgs/parts.jpg)

## Part actions
we define 5 kinds of parts: head, torso, lower body, two arms and two hands. For each of them, we define some actions, such as "head: looking up", "hand: half holding" etc.

|index|part|action|index|part|action|
|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
|1|head|breathing|18|lower body|standing|
|2|head|drinking|19|lower body|walking|
|3|head|laughing|20|arms|curving down|
|4|head|looking down|21|arms|curving up|
|5|head|looking through|22|arms|straight down|
|6|head|looking up|23|arms|straight up|
|7|head|normal|24|hands|cutting|
|8|head|speaking|25|hands|half holding|
|9|head|brushing teeth|26|hands|fully holding|
|10|torso|bending|27|hands|merging|
|11|torso|fading away|28|hands|slack|
|12|torso|normal|29|hands|printing|
|13|torso|lying|30|hands|proping|
|14|lower body|crouching|31|hands|supporting|
|15|lower body|forking|32|hands|washing|
|16|lower body|running|33|hands|waving|
|17|lower body|sitting|34|hands|writing|


The part action set we have collected is not perfect now, if you find annotation errors or you have good ideas on how to design the set, please feel free to contact me.

## Prerequisite
1. matcaffe (caffe and matlab)
2. tensorflow 
3. tensorflow-hub
4. other python library(tqdm, scikit-learn, numpy, opencv-python, python-tk, sklearn)

## Annotations
The annotations are provided as "txt" files, in each of them, we label part actions in order of head-torso-lower_body-left_arm-right_arm-left_hand-right_hand. Since in any case you need to locate part locations in the test phase by algprithms, we do not provide part locations in the training set, which keeps consistency for the part localization.

Download the annotations:
[Annotations](https://drive.google.com/file/d/0B9BLbZk6ZRS0cVdxbGRPQzJHRGs/view?usp=sharing)

### Models
you can download the model from my google drive: [PAN of Stanford40 for TF]
(https://drive.google.com/file/d/1C9nVhlhvOCqRXEayM6p6SzZiKlTHNQEo/view?usp=sharing)

## Testing
To test the network, you need to follow the steps:
1. Download the Stanford-40 dataset in data/stanford40
2. Use tools/PersonImage.m to generate bbox images in BBOXImages/(the whole images are stored in JPEGImages/).
3. Use tools/Realtime_Multi_Person_Pose_Estimation-mater/testing/demo.m to generate parts in PARTImages/, these programs are modified from the Part Affinity Field Network (see citations).
4. Change every path to your files in pan.py
5. run python pan.py
6. The test result automatically generate in test_result_save_folder

## Training
To train the network, you need to use my modified caffe as below at first.
you can follow the steps to generate your own part action network:
1. Through tensorflow hub ResNet v1-50 automatically loaded.
2. Generate BBOXImages/, JPEGImages/, PARTImages/ folders, by running tools/PersonImage_train.m , tools/Realtime_Multi-Person_Pose_Estimation-master/testing/demo.m
3. Change all path to your files in pan.py
4. Set check_point_file='' in pan.py
5. Set Train = True in pan.py
6. Run python pan.py

### Our modified Caffe
https://github.com/ZhichenZhao/my-modified-caffe/tree/master

If you find that our paper or this project help, please cite the paper:
```
@InProceedings{Zhao_2017_ICCV,
author = {Zhao, Zhichen and Ma, Huimin and You, Shaodi},
title = {Single Image Action Recognition Using Semantic Body Part Actions},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2017}
}
```
