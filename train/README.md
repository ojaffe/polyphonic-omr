# Experiment_code
Code for training models for Optical Music Recognition (OMR) and making predictions

# Instructions
There is currently only one model available.

## RNNDecoder
Training
```
python train.py -voc_p <path to pitch vocabulary> -voc_r <path to rhythm vocabulary>
```
Inference
```
python predict.py -voc_p <path to pitch vocab> -voc_r <path to rhythm vocab> -model <path to trained model> -images <path to directory of images to predict> -out <directory to output predictions>
```

## Pretrained Models
Please download pretrained models from https://drive.google.com/drive/folders/1OVPg_oSsb1X9YaXI5mB7nxhO1WQx53u9?usp=sharing
