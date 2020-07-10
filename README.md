# Transparent Shape Dataset Creation

This repository contains the code to create the transparent shape dataset in paper [Through the Looking Glass: Neural 3D Reconstruction of Transparent Shapes, CVPR 2020](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20Transparent/). Please check our [webpage](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20Transparent/) for more information. Please consider citing our paper if you find this code useful. Part of the code is inherited from the 2 prior projects:
* Xu, Z., Sunkavalli, K., Hadap, S., & Ramamoorthi, R. (2018). Deep image-based relighting from optimal sparse samples. ACM Transactions on Graphics (TOG), 37(4), 1-13.
* Li, Z., Xu, Z., Ramamoorthi, R., Sunkavalli, K., & Chandraker, M. (2018). Learning to reconstruct shape and spatially-varying reflectance from a single image. ACM Transactions on Graphics (TOG), 37(6), 1-11.

## Overview 
We create transparent shape dataset by procedurally combining shape primitives to create complex sceenes. An overview of our dataset creation pipeline is shown below. Please refer to our [paper](https://arxiv.org/abs/2004.10904) for more details. 
![](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20Transparent/github/dataset.png)

## Prerequsites 
In order to run the code, you will need to prepare:
* Laval Indoor scene dataset:
* Optix Renderer:
* Colmap: 
* Meshlab: 

## Instructions 
We will first go through the process of creating training set for 10 views reconstruction. The instructions to create 5-view and 20-view datasets will be given below. 

### Creating testing set

### Creating 5-view and 20-view dataset
