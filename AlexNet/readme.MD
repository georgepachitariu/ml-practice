## Task: Implement Alexnet to classify the images in Imagenet2012

I followed the architecture described in the paper: _"ImageNet Classification with Deep Convolutional Neural Networks"_ by Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton. 

My best performance was top-5 error of 38% on the dataset with 1.2 million train images and 1000 categories. The training had 45 cycles over the augmented dataset and it took 16 hours on a computer with GeForce RTX 2070 with 7 GB memory, 16GB RAM and data stored on SSD.

Please read `alexnet.py` to see the details of the implementation.

  

### To train:
1. Use the Dockerfile in this repository to create and start a Docker container. The commands I used:
```
docker build --tag gpachitariu_docker:1.0 .
docker run -p 8888:8888 -v /home/gpachitariu/git:/home/gpachitariu/git \
-v /home/gpachitariu/SSD/data:/home/gpachitariu/SSD/data \
--gpus all -it gpachitariu_docker:1.0
```
2. In the container start training:
```
cd git/ml-practice/AlexNet
python alexnet.py
```
