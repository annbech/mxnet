
# Object Detection Using Faster R-CNN with Distributed Implementation and Data Parallelization

Region Proposal Network (RPN) solves object detection as a regression problem
from the objectness perspective. Bounding boxes are predicted by applying
learned bounding box deltas to base boxes, namely anchor boxes across
different positions in feature maps. The Training process directly learns a
mapping from raw image intensities to bounding box transformation targets.

Fast R-CNN treats general object detection as a classification problem and
bounding box prediction as a regression problem. Classifying cropped region
feature maps and predicting bounding box displacements together yields
detection results. Cropping feature maps instead of image input accelerates
computation by using shared convolution maps. Bounding box displacements
are simultaneously learned in the training process.

Faster R-CNN uses an alternate optimization training process between RPN
and Fast R-CNN. Fast R-CNN weights are used to initiate RPN for training.

## Getting Started
You can get the source code for this example tutorial on [
GitHub](https://github.com/dmlc/mxnet/tree/master/example/rcnn). For links to other resources, see [Downloadable Resources](#downloadable-resources).


1. Install the following Python packages: `easydict`, `cv2`, and `matplotlib`. MXNet require `NumPy`.
* Install MXNet using a version no later than commit 8a3424e, preferably the latest master.
  Follow the instructions at http://mxnet.readthedocs.io/en/latest/how_to/build.html. Install the Python interface.
* Try out detection result by running:

 	`python demo.py --prefix final --epoch 0 --image myimage.jpg --gpu 0`.
  
For this tutorial, we assume that you have downloaded the pretrained network and placed the extracted file `final-0000.params` in this folder and there is an image named `myimage.jpg`.

## Training Faster R-CNN

1. Install the Python package `scipy`.
* Download Pascal VOC data and place it in the `data` folder according to `Data Folder Structure`.
  You might want to create a symbolic link to the VOCdevkit folder by using the following command:

	`ln -s /path/to/your/VOCdevkit data/VOCdevkit`

* Download the VGG16 pretrained model, use `mxnet/tools/caffe_converter` to convert it,
  rename it to `vgg16-symbol.json` and `vgg16-0001.params`, and place it in the `model` folder. MXNet uses the
  `model` folder to place model checkpoints along the training process.
* After VOCdevkit is ready, start training by running:

	`python train_alternate.py` 

  For example, a typical command would be:

	`python train_alternate.py --gpus 0`
 
This trains the network on the VOC07 trainval.
  For more control of the training process, see the argparse help, which you can access by running `python train_alternate.py -h`.
* Start testing by running `python test.py` after completing the training process.
  For example, a typical command would be `python test.py --has_rpn --prefix model/final --epoch 8`. This tests the network on the VOC07 test.
  Adding a `--vis` turns on visualization, and adding `-h` shows help as it does in the training process.

## Testing Faster R-CNN


1. Download the Pascal VOC data and place it in the `data` folder, according to `Data Folder Structure`.
  You might want to create a symbolic link to the VOCdevkit folder by running:
`ln -s /path/to/your/VOCdevkit data/VOCdevkit`.
* Download the precomputed selective search data, and place it in the `data` folder, according to `Data Folder Structure`.
* Download the VGG16 pretrained model, use `mxnet/tools/caffe_converter` to convert it,
  rename to `vgg16-symbol.json` and `vgg16-0001.params`, and place it in  the `model` folder.
  The `model` folder will be used to place model checkpoints along the training process.
* To use the selective search proposal, start training by running
`python -m tools.train_rcnn --proposal ss` .
* Start testing by running
`python -m tools.test_rcnn --proposal ss`.

## Using Approximate Joint Training
Train the  faster-rcnn model using the end2end training method, which is implemented by approximate joint training and is similar to [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn).

Start end2end training by running `python -u train_end2end.py`. For information about how to set the training paramters, use `python train_end2end.py --help`.  For example, help explains how to set the step of dropping lr by using `--factor-step`.

## Downloadable Resources
Links to trained model:

* Baidu Yun: http://pan.baidu.com/s/1boRhGvH (ixiw) 
	
* Dropbox: https://www.dropbox.com/s/jrr83q0ai2ckltq/final-0000.params.tar.gz?dl=0
 
Links to Pascal VOC and precomputed selective search proposals:

* Pascal VOCdevkit:

  	http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
  	http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  	http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar

* selective_search_data (by Ross Girshick):

  	https://github.com/rbgirshick/fast-rcnn/blob/master/data/scripts/fetch_selective_search_data.sh
  ```

Data folder structure (create a `data` folder if there is none):

      ```
      VOCdevkit
      -- VOC + year (JPEG images and annotations)
      -- results (will be created by evaluation)
      ---- VOC + year
      ------ main
      -------- comp4_det_val_aeroplane.txt
      selective_search_data
      rpn_data (will be created by rpn)
      cache (will be created by imdb)
      ```

## Disclaimer
This repository uses code from [MXNet](https://github.com/dmlc/mxnet),
[Fast R-CNN](https://github.com/rbgirshick/fast-rcnn),
[Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn), and
[caffe](https://github.com/BVLC/caffe). Training data are from
[Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and
[ImageNet](http://image-net.org/). The model comes from
[VGG16](http://www.robots.ox.ac.uk/~vgg/research/very_deep/).

## References
1. Tianqi Chen, Mu Li, Yutian Li, Min Lin, Naiyan Wang, Minjie Wang, Tianjun Xiao, Bing Xu, Chiyuan Zhang, and Zheng Zhang. MXNet: A Flexible and Efficient Machine Learning Library for Heterogeneous Distributed Systems. In Neural Information Processing Systems, Workshop on Machine Learning Systems, 2015
2. Ross Girshick. "Fast R-CNN." In Proceedings of the IEEE International Conference on Computer Vision, 2015.
3. Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. "Faster R-CNN: Towards real-time object detection with region proposal networks." In Advances in Neural Information Processing Systems, 2015.
4. Yangqing Jia, Evan Shelhamer, Jeff Donahue, Sergey Karayev, Jonathan Long, Ross Girshick, Sergio Guadarrama, and Trevor Darrell. "Caffe: Convolutional architecture for fast feature embedding." In Proceedings of the ACM International Conference on Multimedia, 2014.
5. Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. "The pascal visual object classes (voc) challenge." International journal of computer vision 88, no. 2 (2010): 303-338.
6. Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. "ImageNet: A large-scale hierarchical image database." In Computer Vision and Pattern Recognition, IEEE Conference on, 2009.
7. Karen Simonyan, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).

# Next Steps
* [MXNet tutorials index](http://mxnet.io/tutorials/index.html)