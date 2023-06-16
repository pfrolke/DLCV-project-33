# PRAD-AI: Automatic Fashion Style Transfer Using CNN

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

Style transfer algorithms are algorithms that can be used to change the visual style of an input image to match the style of a reference image while preserving the higher level content. An example of the results of a style transfer algorithm can be seen in figure 1. Different types of deep learning methods have been used to do these style transfers. Early work involved Convolutional Neural Networks (CNN) to transform the styles of famous painting onto photographs [[1]](#1). Another work proposed an Auto Encoder method, where a learned "filter bank" is applied in the latent space with which an input image is decoded to have a certain style [[4]](#4). Later works used the generative abilities of Generative Adversarial Networks (GAN) to perform an image translation from one image to another [[2]](#2) or to have images adapt a cartoon style[[3]](#3). 

Furthermore, researchers have explored applying style transfer methods to transfer fashion styles in clothing pieces. Most research thus far has relied on GANs to perform style transfers. A first research direction that was explored looked at the generation of new clothing on a wearer through generative adversarial learning, while preserving the wearer's pose and adhering to specific language descriptions[[5]](#5). A later work that addresses the specific challenge of person-to-person clothing swapping by proposing a multistage deep generative approach [[6]](#6).

In this project we followed the CNN-based approach [[1]](#1). We chose to explore this method because it requires much less computational resources than the Auto-Encoder or GAN methods.

## The Fashionpedia Dataset
A dataset of fashion images, called Fashionpedia [7], was recently released in combination with fashion garment segmentation and classifier models. The dataset consists out of around 45.000 images of people wearing clothing. Each image has detailed annotations for each garment, which contain a class label out of 27 categories and fine-grained features of the garment from a list of 294 attributes.

From this dataset, we selected garments that where classified to a class belonging to the `upperbody`-superclass. Additionally, we required each clothing piece in our selection to have a minimum area of `25%` of the image size to filter out small detail pieces. From this selection, we use the segmented clothing images after scaling to `240x240` and normalizing.

## CNN-Based Style Transfer
A pre-trained Convolutional Neural Network (CNN), in this case VGG-19 [[8]](#8), can be used to manipulate a content image to adopt the style of a style reference image by following [[1]](#1). The authors find that the representations of image content and image style within a CNN are separable. By reconstructing input images from the activations at different layers in the CNN, they show how deeper layers capture the high-level content of an image, while shallower layers capture a style representation. Specifically, layer `conv_4_2` is used for the content representation, and layers `conv_1_1`, `conv_2_1`, `conv_3_1`, `conv_4_1`, and `conv_5_1` are used for the style representation.

To transfer style from a style image to a content image, we do a forward pass of the pre-trained CNN with the content and the style images and save the layer activations. Next, we do a forward pass on the input image, which starts off as a copy of the content image. Then, we calculate the loss and iteratively perform gradient descent on the input image.

The loss function is a weighted combination of a content loss function and a style loss function:
$$\ell_{total}(p,a,x)=\alpha \ell_{content}+\beta \ell_{style}$$
The content loss is given by the squared difference between the activations at the chosen content representation layer of the content image ($P_{ij}^l$) and the input image ($F_{ij}^l$):
$$\ell_{content}(p,x)=\frac{1}{2}\sum_{i,j}(F_{ij}^l - P_{ij}^l)^2$$
The style loss is given by the mean-squared distance between the Gram matrices of the activations at the style representation layers of the style image ($A_{ij}^l$) and the input image ($G_{ij}^l$):
$$\ell_{style}(a,x)=\frac{1}{|L|}\sum_{l\in L}\frac{1}{4N_l^2M_l^2}\sum_{i,j}(A_{ij}^l - G_{ij}^l)^2$$

## Content & Style Extraction
We first experimented with extracting just the content or the style of an image from the dataset, to gain an understanding of how the fashion images work with this method. 
 
![style and content extraction](/blog/imgs/loss.png)
*Figure 2. Extracting style and content to an RGB noise image input (top-left). The generated style representation (top-middle). The generated content representation (top-right). The style-reference (bottom-middle). The content reference (bottom-right).*

We manipulate an RGB noise image input using just the style loss or the content loss for 50000 iterations of gradient descent (Adam optimizer; lr=$10^{-3}$). The results in figure 2 show how the style loss is able to add the blue checkered patterns in the image, without introducing the structural content of the reference clothing. Contrastingly, the content loss is able to introduce the structure of the content image, while keeping the noise pattern intact throughout the image.

Figure 2 also highlights an issue with using the segmented pieces for style transfer. The generated style representation shows big bright spots in the image, which are likely caused by the white background of the style reference image also being treated as "style". An idea to crop the style images to reduce the amount of visible background was explored, however this reduced the quality of the reduced the quality of the generated images due to the lower resolution of the crop. Therefore, we use the uncropped style image in our following experiments.

## Quantitative Evaluation by FashionPedia Classifier
In order to measure the quantitative performance of the style transfer method, we apply the pre-trained FashionPedia classifier to the generated images. The classification accuracy on the generated image of the class matching the content image provides an insight into how much the generated clothing piece's structural content has changed. Similarly, the classification accuracy of the goal attribute of the style image on the generated image provides an insight into how much the generated clothing piece's style has changed.

Ideally, when classifying the generated image the accuracy of the clothing category ($C$) would match that of the content image, and accuracy of the style attribute ($A$) would match that of the style image. Therefore, we report a PRADAPoints error (PPe) as the mean square error of these accuracies:

$$ PPe=\frac{1}{2}(C_{style} - C_{generated})^2+\frac{1}{2}(A_{style} - A_{generated})^2 $$

It's important to note that this score does not give the whole picture. The FashionPedia classifier is trained on pictures only, so the artifacts that come from the style transfer can completely throw it off even though it would be acceptable for a human.

## Tuning the Settings
![style and content extraction](/blog/imgs/params.png)
*Figure 3. Style transfer results for multiple settings of the ratio $\alpha/\beta$ vs. the number of iterations. Same content and style image as in fig 2.*

Figure 3 shows the results of the style transfer method for different settings of the ratio $\alpha/\beta$, which is used in the loss function, and the number of iterations the algorithm is ran for. The PPe scores are shown in the table below. Contrary to the scores, we used $\alpha/\beta=10^{-3}$ and 10000 iterations for further experiments, since we found this had a good balance in showing the style while keeping the structural content intact.

|      | 5000       | 10000  | 15000  | 20000  |
|------|------------|--------|--------|--------|
| 1e-2 | **0.3393** | 0.4704 | 0.4701 | 0.47   |
| 1e-3 | 0.4697     | 0.4727 | 0.4754 | 0.4754 |
| 1e-4 | 0.4727     | 0.4719 | 0.4754 | 0.4754 |

*Table 1. PPe scores for different settings.*

## Fashion Style Transfer Results
![style transfer results](/blog/imgs/transfers.png)

## Multiple References Style Transfer
Additionally, we carried out experiments by incorporating multiple style images in each iteration. The idea behind this approach was to utilize several style images with a common attribute, aiming to uncover the underlying generalized representation of that style and transfer it to the input image.

![multiple style transfer results](/blog/imgs/multi.png)

*Figure 5. Style transfer with 10 style reference images. 3000 iterations.*

However, this method seems to generate worse results (figure 5). The styles from each style reference image seem to get transferred onto the target image in a fragmented way. And since these style images are still highly diverse, this results in a combined style that is unnatural.

## Conclusion


## References

<a id="1">[1]</a> Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). A neural algorithm of artistic style. arXiv preprint arXiv:1508.06576.

<a id="2">[2]</a>Zhu, J. Y., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired image-to-image translation using cycle-consistent adversarial networks. In Proceedings of the IEEE international conference on computer vision (pp. 2223-2232).

<a id="3">[3]</a>Chen, Y., Lai, Y. K., & Liu, Y. J. (2018). Cartoongan: Generative adversarial networks for photo cartoonization. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 9465-9474).

<a id="4">[4]</a>Chen, D., Yuan, L., Liao, J., Yu, N., & Hua, G. (2017). Stylebank: An explicit representation for neural image style transfer. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1897-1906).

<a id="5">[5]</a>Zhu, S., Urtasun, R., Fidler, S., Lin, D., & Change Loy, C. (2017). Be your own prada: Fashion synthesis with structural coherence. In Proceedings of the IEEE international conference on computer vision (pp. 1680-1688).

<a id="6">[6]</a>Liu, Y., Chen, W., Liu, L., & Lew, M. S. (2019). Swapgan: A multistage generative approach for person-to-person fashion style transfer. IEEE Transactions on Multimedia, 21(9), 2209-2222.

<a id="7">[7]</a>Jia, M., Shi, M., Sirotenko, M., Cui, Y., Cardie, C., Hariharan, B., ... & Belongie, S. (2020). Fashionpedia: Ontology, segmentation, and an attribute localization dataset. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part I 16 (pp. 316-332). Springer International Publishing.

<a id="7">[8]</a>Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).


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