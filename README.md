# SPP-Net

## Paper
[《Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition》](https://arxiv.org/pdf/1406.4729.pdf)
![alt text](docs/1.jpg "title")
![alt text](docs/2.jpg "title")
![alt text](docs/3.jpg "title")

```
layer = (13*13)
level = [3,2,1]

pool 3x3
window size = ceil[13/3] = 5
stride size = floor[13/3] = 4

pool 2x2
window size = ceil[13/3] = 7
stride size = floor[13/3] = 6

pool 1x1
window size = ceil[13/3] = 13
stride size = floor[13/3] = 13
```

![alt text](docs/4.jpg "title")
![alt text](docs/5.jpg "title")
![alt text](docs/6.jpg "title")

## Dataset
```
http://www.robots.ox.ac.uk/~vgg/data/flowers/102/
1. Dataset images ----> 102flowers.tgz
2. The image labels ----> imagelabels.mat
3. The data splits ----> setid.mat
```

## Train
#### Single-size training
// TODO

#### Multi-size training
// TODO
