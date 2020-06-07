# Mask R-CNN for Object Detection and Segmentation

This is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

### Raw segmentation
http://www.robots.ox.ac.uk/~vgg/software/via/via-1.0.6.html

### Training model
```
python foots.py train --dataset=/Users/ACEBALLOS/Developer/FootsModel/dataset --weights=coco

```

### Run tensorBoard
```
tensorboard --logdir='./logs' --port 6006 --host=127.0.0.1
```

### Find model
Go to logs and find the file mask_rcnn_foots_0010.h5, then copy this file and paste in Mask_RCNN

### Run model 
Go to file inspect_balloon_model.ipynb into the foots folder and run jupyter notebook
```
jupyter notebook

```
