# PRAD_AI: Automatic Fashion Style Transfer Using CNN

## Group 33 - Students

Paul Frolke, 4888693, <p.r.frolkee@student.tudelft.nl> \
Gijs Admiraal, 4781669, <g.j.admiraal@student.tudelft.nl>

## Table of contents

[TOC]

## Introduction

The world of fashion is diverse and large and thus encompasses a variety of styles to suit the taste and preferences of each individual. Fashion designers often create multiple styles or variations of their clothing to cater to the different tastes and preferences of their target audience. Examples of styles include the fabric type, the color, and the fit of a piece of clothing. However, the traditional process of designing and physically prototyping these new styles is both time-consuming and resource-intensive, which leads to a wasteful process. To help alleviate these issues deep learning algorithms called style transfer algorithms can be utilized.

![example style transfer](https://assets-global.website-files.com/5d7b77b063a9066d83e1209c/613ebc713d78bd6d64e763b8_IJBbdBDp67rtR1bfd0GVlcu5kQIIOX5ExT3I8w7f7UGV90_-SwP-lOfF61k6Npq_4SqPiXnQnVgRXwFmNe8c0OctDfb0p_ScrJGWNkgu7S1UKZmk_BsYb-_C11OGXciC8IjqMfO2%3Ds0.png)
*Figure 1. An example image of applied style transfers on the Mona Lisa*

Style transfer algorithms are algorithms that have the ability to adopt the visual style of a style image by manipulating a target image such that this target image will be in the visual style of the style image. An example of how the results of such a style transfer algorithm looks like can be seen in figure 1. Different types of deep learning methods have been used to do these style transfers. Early work began with using Convolution Neural Networks (CNN) to transform the styles of famous painting onto photographs [[1]](#1). Another work proposed the use of an Auto-Encoder network as a key component for style transfer, focusing on encoding and decoding image features while incorporating style information into the learned representations [[4]](#4). Later work used the generative abilities of Generative Adversarial Networks (GAN) to perform an image translation from one image to another [[2]](#2) or to have images adapt a cartoon style[[3]](#3). 

Research has already been done to see if these style transfer algorithms can be used to transfer fashion styles. Most research has thus far relied on GANs to perform style transfers. A first research direction that was explored looked at the generation of new clothing on a wearer through generative adversarial learning, while preserving the wearer's pose and adhering to specific language descriptions[[5]](#5). A later work that addresses the specific challenge of person-to-person clothing swapping by proposing a multistage deep generative approach [[6]](#6).

A recent fashion dataset in combination with a fashion garment segmentation and classifier models was released called Fashionpedia [[7]](7). The dataset consists out of around 45 thousand images of people wearing clothing. Each image has been segmented per garment and classified based on 27 categories and 294 attributes. 

Our research will focus on if we can do style transfers from 

## Model



## Results

## Conclusion

## References

<a id="1">[1]</a> Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). A neural algorithm of artistic style. arXiv preprint arXiv:1508.06576.

<a id="2">[2]</a>Zhu, J. Y., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired image-to-image translation using cycle-consistent adversarial networks. In Proceedings of the IEEE international conference on computer vision (pp. 2223-2232).

<a id="3">[3]</a>Chen, Y., Lai, Y. K., & Liu, Y. J. (2018). Cartoongan: Generative adversarial networks for photo cartoonization. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 9465-9474).

<a id="4">[4]</a>Chen, D., Yuan, L., Liao, J., Yu, N., & Hua, G. (2017). Stylebank: An explicit representation for neural image style transfer. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1897-1906).

<a id="5">[5]</a>Zhu, S., Urtasun, R., Fidler, S., Lin, D., & Change Loy, C. (2017). Be your own prada: Fashion synthesis with structural coherence. In Proceedings of the IEEE international conference on computer vision (pp. 1680-1688).

<a id="6">[6]</a>Liu, Y., Chen, W., Liu, L., & Lew, M. S. (2019). Swapgan: A multistage generative approach for person-to-person fashion style transfer. IEEE Transactions on Multimedia, 21(9), 2209-2222.

<a id="7">[7]</a>Jia, M., Shi, M., Sirotenko, M., Cui, Y., Cardie, C., Hariharan, B., ... & Belongie, S. (2020). Fashionpedia: Ontology, segmentation, and an attribute localization dataset. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part I 16 (pp. 316-332). Springer International Publishing.


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
