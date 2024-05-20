# Unbiased Multi-Label Learning from Crowdsourced Annotations

This is a PyTorch implementation of our ICML 2024 paper CLEAR.

## An example of running CLEAR on *Image*


```shell
# For M=5, [T01, T10]=[0.2, 0.2]
CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'Image' --noise_rate '0.1,0.1,0.2,0.2,0.2,0.2,0.2,0.2,0.3,0.3' --loss_type 'unbiased' --no-verbose

# For M=5, [T01, T10]=[0.2, 0.5]
CUDA_VISIBLE_DEVICES=1 python main.py --dataset 'Image' --noise_rate '0.1,0.4,0.2,0.5,0.2,0.5,0.2,0.5,0.3,0.6' --loss_type 'unbiased' --no-verbose

# For M=5, [T01, T10]=[0.5, 0.2]
CUDA_VISIBLE_DEVICES=2 python main.py --dataset 'Image' --noise_rate '0.4,0.1,0.5,0.2,0.5,0.2,0.5,0.2,0.6,0.3' --loss_type 'unbiased' --no-verbose
```

## Acknowledgement
This paper is supported by [Netease Youling Crowdsourcing Platform](https://fuxi.163.com). As the importance of data continues rising, Netease Youling Crowdsourcing Platform is dedicated to utilizing various advanced algorithms to provide high-quality, low-noise labeled samples. Feel free to contact us for more information.