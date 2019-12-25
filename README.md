# Action Recognition on Realtime Application

This project is RGB base action recognition system that focus on real time application. We use [KARD dataset](https://data.mendeley.com/datasets/k28dtm7tr6/1) to train this model. You can check the full demo version in this youtube [link](https://www.youtube.com/channel/UChJg8ndTnT_gEyhd43Ki40Q/featured?view_as=subscriber)

![alt text](https://github.com/peachman05/action-recognition-tutorial/blob/master/media/demo.gif "demo")

# Overview

* [Requirement](#requirement)
* [Dataset](#dataset)
* [Demo](#demo)
* [Evaluation](#evaluation)
* [Training](#training)
* [Performance](#performance)
* [Method](#method)
* [Note](#note)
* [Reference](#reference)

## Requirement

- keras

- tensorflow 1.14 (for using CuDNNLSTM)

- CUDA 10.0 

## Dataset
You can download dataset from [here](https://data.mendeley.com/datasets/k28dtm7tr6/1). Skeleton joints and depth data is not used in this project. Only RGB part is needed. For preparing dataset, you should make the structure of your folder to be in this form
```
KARD-split
├── a01                   
│   ├── a01_s01_e01.mp4             
│   ├── a01_s01_e02.mp4            
│   ├── ...           
│   ├── a01_s10_e03     
├── a02                   
│   ├── a02_s01_e01.mp4             
│   ├── a02_s01_e02.mp4            
│   ├── ...           
│   ├── a02_s10_e03      
├── ....
├── ....
├── a18                   
│   ├── a18_s01_e01.mp4             
│   ├── a18_s01_e02.mp4            
│   ├── ...           
│   ├── a18_s10_e03   
└── ...
```
You can see more detail in dataset_list/trainlist.txt and dataset_list/testlist.txt

## Demo

Webcam or any camera is required for using demo. You can run demo by using the following command
```
python webcam.py
```
## Evaluation

If you want to see accuracy and confusion matrix of pretrain model, you can run the evaluation part by using below command.
```
python evaluate_model.py
```
## Training

You can try to train model by run this command.

```
python train.py
```
If you want to change any parameter, you can find it in train.py file

## Performance

Accuracy: around 87-89% (depend on which part of test set that is random)

Confusion Matrix: 
![alt text](https://github.com/peachman05/action-recognition-tutorial/blob/master/media/confusion_matrix.png "Confusion Matrix")

## Method

Input: 8 RGB frames

Output: 18 action classes

* This project use RGB Difference as input. The idea is from this [paper](https://arxiv.org/abs/1705.02953) and this [project](https://github.com/AhmedGamal1496/online-action-recognition#Introduction). 
* This project just use only simple model to solve it. I use just only LSTM as core of model and use MobileNetV2 as feature extraction part. You can see the detail of architecture in model_ML.py
* while testing and evaluation, I will random n_sequence frames from each video file. So, n_sequence frames is "1 sample". while testing, if we random only 1 sample per 1 video file is not good because the accuracy will be unstable. So, we need to random more sample per one file. For example in evaluate_model.py, I set 'n_mul_test' to 2. It mean I will random 2 sample per one video file. You can change n_mul_test to be any value. If value is high, the accuracy will be stable but it need more testing time.

## Note
* The hyperparameters of train.py, evaluate_model.py and webcam.py is located in header of file. You can adjust it.
* If you face the out of memory problem when you try to evaluate or train, you can decrease n_batch and n_sequence to reduce memory consumption. I suggest you should not use n_batch = 1 because the accuracy will be very swing and cannot converge


## Spacial Thanks
#### Beijing University of Post and Telecommunication(BUPT)

My supervisor: Assoc Prof Dr. Hui Gao

My mentor: ChaiXinXin

#### Tsinghua University

Co-supervisor: Prof Dr.Xin Su



## Reference

### Example code

[https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly) Data generator on keras
[https://github.com/eriklindernoren/Action-Recognition](https://github.com/eriklindernoren/Action-Recognition) Sampling Idea

[https://github.com/AhmedGamal1496/online-action-recognition#Introduction](https://github.com/AhmedGamal1496/online-action-recognition#Introduction) RGB Difference Example

### Paper

Temporal Segment Networks for Action Recognition in Videos, Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, and Luc Van Gool, TPAMI, 2018. [Arxiv Preprint](https://arxiv.org/abs/1705.02953)

### Dataset
[https://data.mendeley.com/datasets/k28dtm7tr6/1](https://data.mendeley.com/datasets/k28dtm7tr6/1)



