# PRAD_AI: Automatic Fashion Style Transfer Using CNN

## Group 33 - Students

Paul Frolke, 4888693, <p.r.frolkee@student.tudelft.nl> \
Gijs Admiraal, 4781669, <g.j.admiraal@student.tudelft.nl>

## Table of contents

[TOC]

## Introduction

The fashion industry is a worldwide industry that involves a large and diverse set of people. In order to accommodate the diverse tastes and preferences of their target audience, fashion designers frequently develop multiple styles or variations of their clothing. Examples of styles include the fabric type, the color, and the fit of a piece of clothing. However, the traditional process of designing and physically prototyping these new styles is time-consuming, resource-intensive, and very wasteful. In this project, we were inspired to help alleviate these issues by exploring deep learning style transfer algorithms that can digitally apply styles to a clothing image.

## Style Transfer Algorithms

![example style transfer](https://assets-global.website-files.com/5d7b77b063a9066d83e1209c/613ebc713d78bd6d64e763b8_IJBbdBDp67rtR1bfd0GVlcu5kQIIOX5ExT3I8w7f7UGV90_-SwP-lOfF61k6Npq_4SqPiXnQnVgRXwFmNe8c0OctDfb0p_ScrJGWNkgu7S1UKZmk_BsYb-_C11OGXciC8IjqMfO2%3Ds0.png)
*Figure 1. An example image of applied style transfers on the Mona Lisa*

Style transfer algorithms are algorithms that can be used to change the visual style of an input image to match the style of a reference image while preserving the higher level content. An example of the results of a style transfer algorithm can be seen in figure 1. Different types of deep learning methods have been used to do these style transfers. Early work involved Convolutional Neural Networks (CNN) to transform the styles of famous paintings onto photographs [[1]](#1). Another work proposed an Auto Encoder method, where a learned "filter bank" is applied in the latent space with which an input image is decoded to have a certain style [[4]](#4). Later works used the generative abilities of Generative Adversarial Networks (GAN) to perform an image translation from one image to another [[2]](#2) or to have images adapt a cartoon style[[3]](#3).

Furthermore, researchers have explored applying style transfer methods to transfer fashion styles in clothing pieces. Most research thus far has relied on GANs to perform style transfers. A first research direction that was explored looked at the generation of new clothing on a wearer through generative adversarial learning, while preserving the wearer's pose and adhering to specific language descriptions[[5]](#5). A later work that addresses the specific challenge of person-to-person clothing swapping by proposing a multistage deep generative approach [[6]](#6).

In this project we followed the CNN-based approach [[1]](#1). We chose to explore this method because it requires much less computational resources than the Auto-Encoder or GAN methods.

With the rise of generative diffusion models like Stable diffusion [[9]](#9), we also explored the possibility of using diffusion models for style transfer. Diffusion based models are known to be able to generate high quality through diffusion steps, that to slowly add random noise to data. They then learn to reverse the diffusion process to construct desired data samples from the noise. A diffusion based model called CoAdapter, which can be used to perform style transfer, was recently released [[10]](#10). In this blog we also see if we can apply diffusion based style transfers on the segmented clothing pieces from the Fashionpedia dataset using CoAdapter. Since these diffusion based models are very computationally expensive, we only explore this model through manual experimentation using a Gradio interface

## The Fashionpedia Dataset

A recent fashion dataset in combination with a fashion garment segmentation and classifier models was released called Fashionpedia [[7]](#7). The dataset consists out of around 45.000 images of people wearing clothing. Each image has detailed annotations for each garment, which contain a class label out of 27 categories and fine-grained features of the garment from a list of 294 attributes.

From this dataset, we selected garments that where classified to a class belonging to the `upperbody`-superclass. Additionally, we required each clothing piece in our selection to have a minimum area of `25%` of the image size to filter out small detail pieces.

## CNN-Based Style Transfer

The pre-trained Convolutional Neural Network (CNN), VGG-19 [[8]](#8), can be used to manipulate a content image to adopt the style of a style reference image, by following [[1]](#1). The authors find that the representations of image content and image style within a CNN are separable. By reconstructing input images from the activations at different layers in the CNN, they show how deeper layers capture the high-level content of an image, while shallower layers capture a style representation. Specifically, layer `conv_4_2` is used for the content representation, and layers `conv_1_1`, `conv_2_1`, `conv_3_1`, `conv_4_1`, and `conv_5_1` are used for the style representation.

To transfer style from a style image to a content image, we perform a forward pass on the pre-trained CNN with the content and the style images and save the layer activations. Next, we perform a forward pass on the input image, which starts off as a copy of the content image. Then, we calculate the loss and iteratively perform gradient descent on the input image.

The loss function is a weighted combination of a content loss and a style loss:
$$\ell_{total}(p,a,x)=\alpha \ell_{content}+\beta \ell_{style}$$
The content loss is given by the squared difference between the activations at the chosen content representation layer of the content image ($P_{ij}^l$) and the input image ($F_{ij}^l$):
$$\ell_{content}(p,x)=\frac{1}{2}\sum_{i,j}(F_{ij}^l - P_{ij}^l)^2$$
The style loss is given by the mean-squared distance between the Gram matrices of the activations at the style representation layers of the style image ($A_{ij}^l$) and the input image ($G_{ij}^l$):
$$\ell_{content}(a,x)=\frac{1}{|L|}\sum_{l\in L}\frac{1}{4N_l^2M_l^2}\sum_{i,j}(F_{ij}^l - P_{ij}^l)^2$$

## CoAdapter Style Transfer

The CoAdapter model allows as input a style image and up to four target images. These five images are fed to T21-Adapters to extract their respective information. The style image is used to extract the style information, while the target images are used to extract the content information. The information that is extracted from these target images are a depth map, a sketch image, canny edges and a spatial color palette. These four target images can all be different images or be the same, depending on the wanted result. For our experiments we used the same image for all target images but the the color palette target image. The color palette target image was set to nothing since we wanted to use the style image as the main color palette.

Additionally to the five input images, the CoAdapter model also allows for an input prompt and a negative input prompt. These prompts are a text description of the wanted and unwanted style transfer. Since the attributes and classes of the Fashionpedia dataset are very detailed, we will try to see if this can help the model to perform better style transfers.

Lastly, three hyperparameters can be set for the CoAdapter model. The first hyperparameter is the number of diffusion steps. The second hyperparameter is the guidance scale, where a lower guidance scale allows to model to generate more diverse and creative outputs. A lower guidance makes the model condition more faithfully to the provided guidance or reference information. The last hyperparameter is used to control the timing and duration of the adapter's influence. The longer the apter is applied the more the model receives  guidance and conditioning information resulting in a more faithful output. During our experiments we will play around with these hyperparameters to see how they affect the results.

## Results

### CNN-Based Results

### CoAdapter Results


## Conclusion

## References

<a id="1">[1]</a> Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). A neural algorithm of artistic style. arXiv preprint arXiv:1508.06576.

<a id="2">[2]</a>Zhu, J. Y., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired image-to-image translation using cycle-consistent adversarial networks. In Proceedings of the IEEE international conference on computer vision (pp. 2223-2232).

<a id="3">[3]</a>Chen, Y., Lai, Y. K., & Liu, Y. J. (2018). Cartoongan: Generative adversarial networks for photo cartoonization. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 9465-9474).

<a id="4">[4]</a>Chen, D., Yuan, L., Liao, J., Yu, N., & Hua, G. (2017). Stylebank: An explicit representation for neural image style transfer. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1897-1906).

<a id="5">[5]</a>Zhu, S., Urtasun, R., Fidler, S., Lin, D., & Change Loy, C. (2017). Be your own prada: Fashion synthesis with structural coherence. In Proceedings of the IEEE international conference on computer vision (pp. 1680-1688).

<a id="6">[6]</a>Liu, Y., Chen, W., Liu, L., & Lew, M. S. (2019). Swapgan: A multistage generative approach for person-to-person fashion style transfer. IEEE Transactions on Multimedia, 21(9), 2209-2222.

<a id="7">[7]</a>Jia, M., Shi, M., Sirotenko, M., Cui, Y., Cardie, C., Hariharan, B., ... & Belongie, S. (2020). Fashionpedia: Ontology, segmentation, and an attribute localization dataset. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part I 16 (pp. 316-332). Springer International Publishing.

<a id="8">[8]</a>Simonyan, K., and Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.

<a id="9">[9]</a>Rombach, R ., Blattmann, A., Lorenz, D., Esser, P., Ommer, B. (2021). High-Resolution Image Synthesis with Latent Diffusion Models. . arXiv preprint arXiv:2112.10752.

<a id="10">[10]</a>TencentARC, T2I-Adapter, (2023), GitHub repository, <https://github.com/TencentARC/T2I-Adapter>


DATASET: <https://fashionpedia.github.io/home/index.html>
DATASET RESEARCH: <https://arxiv.org/pdf/2004.12276v2.pdf>

AUTOENCODER STYLE TRANSFER RESEARCH: <https://openaccess.thecvf.com/content_cvpr_2017/papers/Chen_StyleBank_An_Explicit_CVPR_2017_paper.pdf>
AUTOENCODER STYLE TRANSFER BLOG: <https://medium.com/analytics-vidhya/lets-discuss-encoders-and-style-transfer-c0494aca6090>
AUTOENCODE FASHION STYLE TRANSFER RESEARCH: <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9903255&tag=1>

GAN STYLE TRANSFER: <https://arxiv.org/pdf/1406.2661.pdf>
PERSON TO PERSON GAN FASHION STYLE TRANSFER: <https://ieeexplore.ieee.org/abstract/document/8636265>
IMAGE ANALOGY PROBING GAN FASHION STYLE: <https://arxiv.org/pdf/1709.04695.pdf>
FASHIONGAN BLOG: <https://towardsdatascience.com/deepstyle-part-2-4ca2ae822ba0>

CNN STYLE TRANSFER RESEARCH: <https://arxiv.org/pdf/1508.06576.pdf>
CNN FASHION STYLE TRANSFER BLOG: <https://towardsdatascience.com/a-brief-introduction-to-neural-style-transfer-d05d0403901d>
CNN FASHION STYLE TRANSFER RESEARCH: <https://www.hindawi.com/journals/cin/2020/8894309/>
VGG RESEARCH: <https://arxiv.org/pdf/1409.1556.pdf>

COADAPTER STABBLE DIFFUSION STYLE TRANSFER: <https://github.com/TencentARC/T2I-Adapter/blob/main/README.md>