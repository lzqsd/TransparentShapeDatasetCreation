# Transparent Shape Dataset Creation

This repository contains the code to create the transparent shape dataset in paper [Through the Looking Glass: Neural 3D Reconstruction of Transparent Shapes, CVPR 2020](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20Transparent/). Please check our [webpage](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20Transparent/) for more information. Please consider citing our paper if you find this code useful. Part of the code is inherited from the 2 prior projects:
* Xu, Z., Sunkavalli, K., Hadap, S., & Ramamoorthi, R. (2018). Deep image-based relighting from optimal sparse samples. ACM Transactions on Graphics (TOG), 37(4), 1-13.
* Li, Z., Xu, Z., Ramamoorthi, R., Sunkavalli, K., & Chandraker, M. (2018). Learning to reconstruct shape and spatially-varying reflectance from a single image. ACM Transactions on Graphics (TOG), 37(6), 1-11.

## Overview 
We create transparent shape dataset by procedurally combining shape primitives to create complex sceenes. An overview of our dataset creation pipeline is shown below. Please refer to our [paper](https://arxiv.org/abs/2004.10904) for more details. 
![](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20Transparent/github/dataset.png)

## Prerequsites 
In order to run the code, you will need:
* Laval Indoor scene dataset: Please download the dataset from this [link](http://indoor.hdrdb.com/). We use 1499 environment map for training and 645 environment map for testing. Please turn the `.exr` files into `.hdr` files, since our renderer does not support loading `.exr` files yet. Please save the training set and testing set in `./Envmap/train` and `./Envmap/test` separately.
* Optix Renderer: Please download our Optix-based renderer from this [link](https://github.com/lzqsd/OptixRenderer). There is an Optix renderer included in this repository. But it is the renderer specifically modified to render the two-bounce normal. Please use the renderer from the [link](https://github.com/lzqsd/OptixRenderer) to render images. We will refer to the renderer in this repository as renderer-twobounce and the renderer from the [link](https://github.com/lzqsd/OptixRenderer) as renderer-general in the following to avoid confusion. 
* Colmap: Please install Colmap from this [link](https://colmap.github.io/). We use Colmap to reconstruct mesh from point cloud. 
* Meshlab: Please install [Meshlab](https://www.meshlab.net/). We use the subdivision algorithm in Meshlab to smooth the surface so that there is no artifacts when rendering transparent shape. 

## Instructions 
We will first go through the process of creating training set for 10 views reconstruction. The instructions to create 5-view and 20-view datasets will be given below. 
1. `python createShape.py --mode train --rs 0 --re 3000`
  * Create 3000 randomly generated scene as the training set. The data will be stored under the directory `./Shapes`
2. `python createRenderFilesForDepths.py --mode train --rs 0 --re 3000`
  * Create the camera poses and the xml files for rendering depth maps. For each shape, it will uniformly sample 75 poses surronding the shape. 
3. `python renderAndIntegrate.py --mode train --rs 0 --re 3000 --renderProgram ABSOLUTE_PATH_TO_RENDERER_GENERAL`
  * For each shape, we render 75 depth maps from different views and fuse the depth map together to generate a mesh. After that, we use subdivision to smooth the generated surface. The purpose of this step is to remove the intersection of the randomly generated scene and keep only the outersurface. 

### Creating testing set
To create the testing set, please set `--mode` to `test` and `--re` to 600 and rerun all the above commands again.

### Creating 5-view and 20-view dataset
