---
title: "RetinaNet"
date: 2024-01-02T15:06:43+09:00
draft: false
categories: "Object Detection"
mathjax: true
---

 
# 1. overview

## 1.1 objective

- anchor box를 없애보자.

## 1.2 background

### 1.2.1 anchor box의 문제점

1. 너무 많은 수의 anchor box (ex. DSS에는 40k 이상, RetinaNet에는 100k 이상) \\(\Rightarrow\\) positive sample과 negative sample의 imbalance가 training 속도 느리게 함.
2. hyperparameter \\(\uparrow\\) (ex. ratio, box size, feature map size)  

# 2. main

## 2.1 architecture

![image](https://github.com/ownvoy/ownogatari/assets/96481582/f7c687b2-d679-41e0-aa02-141f64bd799e)

backbone(Hourglass Networ) + prediction Module(Corner pooling, Heatmaps, Embeddings, offsets)으로 이루어져 있다.

### 2.1.1 Corner pooling
![image](https://github.com/ownvoy/ownogatari/assets/96481582/93ef350c-4ba7-4a40-8f9d-9aed9be42bce)
각각의 코너를 top-left corner와 bottom-right corner라 한다.

![image](https://github.com/ownvoy/ownogatari/assets/96481582/8c405fe6-be7c-402e-9c37-d3117099d1c8)
주황색 코너를 봐보면, 사람에 대한 local 정보가 없다.
사람에 대한 정보를 얻기 위해서는 오른쪽과 아래 pixele들을 봐야한다. (Top-left 기준)

![image](https://github.com/ownvoy/ownogatari/assets/96481582/9b47ef2f-5d71-4e57-b75e-0ae3e827dfdd)
수평과 수직 방향으로 max값들을 취한다. Bottom-Right도 마찬가지로 왼쪽과 위쪽을 봐주면 된다.

![image](https://github.com/ownvoy/ownogatari/assets/96481582/35c91433-0c18-4b61-bb79-19d92b28e0e7)

Corner Pooling을 한 후, backbone의 feqturemap과 다시 residual connect를 해준다.
![image](https://github.com/ownvoy/ownogatari/assets/96481582/2a5b9f47-221b-467a-956f-cfb8f8c28a02)
### 2.1.2 Corners and Heatmaps

Corner Pooling 후 heatmap과 embedding, offset을 예측한다.

![image](https://github.com/ownvoy/ownogatari/assets/96481582/3181d7fb-afbe-444e-8c2f-0a06f2e063c6)

- top-left corner와 bottom-right corner에 대한 heatmap이 있다.
- 각각의 size는 \\(C \times W \times H\\)(category for \\(C\\))이다.

__heatmap에 대한 focal loss__

$$L_{d e t}=\frac{-1}{N}\sum_{c=1}^{C}\sum_{i=1}^{H}\sum_{j=1}^{W}\left\{\!\!\!\begin{array}{c}{{(1-p_{c i j})^{\alpha}\log\left(p_{c i j}\right)\qquad\mathrm{if~}y_{c i j}=1}}\\ {{(1-y_{c i j})^{\beta}\left(p_{c i j}\right)^{\alpha}\log\left(1-p_{c i j}\right)\,\mathrm{otherwise}}}\end{array}\right.\quad(1)$$

- \\(p_{cij}\\): \\(c\\)에 대한 prediction
- \\(y_{cij}\\): \\(c\\)에 대한 ground truth
- negative sample: 덜 중요한 sample이니 가중치 많이 줄임.
- negative이면서 corner와 가까운 sample: 정답과 가까이 있으니까, 가중치 적당히 줄임.
- positive sample: 중요한 sample이니 가중치 조금만 줄임.



### 2.1.3 Grouping Corners and Embeddings

![image](https://github.com/ownvoy/ownogatari/assets/96481582/7fff9950-8626-43fd-aa5f-f2a13f3c2e83)
여러 개의 object가 있을 때에는 각각의 쌍을 맞춰줘야 한다.
\\(\Rightarrow\\)각 포인트의 임베딩 벡터로 유사도 구한 후, distance 측정.

$$L_{p u l l}=\frac{1}{N}\sum_{k=1}^{N}\left[\left(e_{t_{k}}-e_{k}\right)^{2}+\left(e_{b_{k}}-e_{k}\right)^{2}\right]$$
같은 group은 pull(비슷하게)하도록 학습한다.


$$L_{p u s h}=\frac1{N(N-1)}\sum_{k=1}^{N}\sum_{\stackrel{j=1}{j\neq k}}^{N}\operatorname*{max}\left(0,\Delta-\vert e_{k}-e_{j}\vert\right)$$
다른 group은 psuh(멀어지게)하도록 학습한다.

### 2.1.4 Offset

offset은 corner의 정확한 위치를 조정한다.

__offset에 대한 loss__

$${L}_{off}=\frac{1}{N}\sum_{k=1}^{N}\mathrm{SmoothL1Loss}\left({\boldsymbol{o}}_{k},{\hat{\boldsymbol{o}}}_{k}\right)$$

__최종 loss__

$${L}={L}_{det}+\alpha{L}_{pull}+\beta{L}_{push}+\gamma{L}_{off}$$

# 3. experiments

![image](https://github.com/ownvoy/ownogatari/assets/96481582/3a6f439b-5985-4e7c-a172-ad0cf8453913)
RetinaNet의 경우 one-stage detector보다는 다 좋고, two-stage detector와는 비등비등하다.

![image](https://github.com/ownvoy/ownogatari/assets/96481582/15335eb7-8a2c-4df0-a0d8-828824f77612)
corner pooling시 object의 보다 정확한 box를 구하는 것을 확인 할 수 있다. 특히 큰 물체와 관련하여서, 점수가 좋다.

![image](https://github.com/ownvoy/ownogatari/assets/96481582/dee1457d-eddf-4976-a0c6-b6e26ac9bc71) 

그러나, corner를 잘못 잡거나 embedding이 올바르지 않을 때에는 error를 보여준다.


