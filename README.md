# PRAD-AI: Automatic Fashion Style Transfer Using CNN

## Data
[Fashionpedia](https://github.com/cvdfoundation/fashionpedia):
* [Images](https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip)
* [Annotations](https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json)

## Possible approaches:
1. learn per-attribute style image, that can be used to do style transfers like in the paper
2. learn classification of attributes, then perform gradient descent to maximize the desired attributes on an input image.