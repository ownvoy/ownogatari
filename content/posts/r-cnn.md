---
title: "R-CNN"
date: 2023-11-12
draft: false
katex: true
mathjax: true
categories: "Object Detection"
tags: ["CV"]
---

# 1. overview

## 1.1 objective

CNN을 object detection에 적용해보자.

## 1.2 background

### 1.2.1 object detection

- object detection은 사물이 뭔지, 어디에 있는지 찾는 것.
- classification과 localization을 동시에 하는 것.

### 1.2.2 way to localize object
1. 가장 greedy한 방법: sliding window 
    이미지 크기: \\(H\times W\\)
    박스 크기: \\(h \times w\\)
    가능한 \\(x\\) 위치: \\(W-w+1\\)
    가능한 \\(y\\) 위치: \\(H-h +1\\)
    가능한 박스 위치: \\((W-w+1)(H-h+1)\\)
    가능한 총 경우의 수
$$\sum_{h=1}^{H}\sum_{w=1}^{W}(W-w+1)(H-h+1)$$
is equal to
$$\frac{H(H+1)}{2}\frac{W(W+1)}{2}$$

만약 이미지의 크기가 300*300이라고 하면 81,000,000의 경우의 수가 나옴.  \\(\Rightarrow\\) computation cost가 너무 큼.

2. selective search

![image](https://github.com/ownvoy/ownogatari/assets/96481582/06f1d1eb-fdea-481c-9b3e-d06f12f4932a)

R-CNN에서는 2000개의 bounding box를 뽑음. 이 과정을 region proposal이라고 함.

### 1.2.3 object detection의 종류

1. proposal-based model 
	-  Two-stage model
	- region proposal을 먼저 뽑고, 그 다음에 classification, localization
	- `R-CNN`, `Fast R-CNN` , `Faster R-CNN`, `R-FCN`

2. proposal-free model
	- Single-stage model
	- region proposal 없이 바로 classification, localization
	- `YOLO`, `SSD`, `DETR`

### 1.2.4 IoU (Intersection over Union)
ground truth와 prediction이 얼마나 많이 겹치느냐

![image](https://github.com/ownvoy/ownogatari/assets/96481582/8bb6dd91-7006-4dd2-abad-cf1c25682e5d)

- threshold를 0.5라 두면, 0.5 이상이면 positive, 0.5 이하면 negative으로 labeling

- negative sample이 압도적으로 많기에, positive 위주로 sampling

### 1.2.5 NMS(Non-Maximum Suppression)

![image](https://github.com/ownvoy/ownogatari/assets/96481582/03f17b38-5962-456f-b2bc-49a5d2b7b745)

하나의 object에 대해 여러 개의 bounding box가 나올 수 있음. 이 중에서 가장 좋은 bounding box를 고르는 것.


### 1.2.6 Recall and Precision

|          | positive(predict) | negative(predict) |
| -------- | -------- | -------- |
| positive(g.t) |   TP (있는 걸 있다고 함)       |    FN (있는데 없다고 함)      |
| negative(g.t) |   FP (없는데 있다고 함)      |    TN (없는걸 없다고 함)      |

![image](https://github.com/ownvoy/ownogatari/assets/96481582/cf9568c9-3e0d-4eea-89c8-569c12ed3947)
- 1,4,5는 TP
- 3은 FP
- 가운데 검정 강아지 박스는 FN
- TN은 object detection에서 고려 x(없는 걸 없다고 판정하는 모델이 아니기에)

$$\text{Precision} : \frac{TP}{TP+FP}$$
$$\text{Recall}: \frac{TP}{TP+FN}$$





# 2. main
## 2.1 architecture
region proposal + cnn
![image](https://github.com/ownvoy/ownogatari/assets/96481582/50c3af46-b485-445c-b5e6-3647f01fd571)

region proposal된 2000개 각각을 cnn에 넣은 후 SVM을 통해 classify 


### 2.1.1 region proposal
어떤 알고리즘을 써도 무방. `selective search`를 사용.

### 2.1.2 cnn
imagenet 사용

![image](https://github.com/ownvoy/ownogatari/assets/96481582/a09c021b-2ef7-4954-a7b6-c87e12e7dce1)

이미지의 크기를 cnn에 넣기 위해, 227*227로 resize

![image](https://github.com/ownvoy/ownogatari/assets/96481582/722d28d4-5866-4a1b-a244-c30d6872be43)

- bottom은 패딩을 추가한건데 성능이 더 좋음.

### 2.1.3 bounding box regression


![image](https://github.com/ownvoy/ownogatari/assets/96481582/e4588995-839d-45dd-83e1-ed3e26a74b8d)

bounding box는 \\(x,y,h,w\\)로 구성이 됨.

![image](https://github.com/ownvoy/ownogatari/assets/96481582/98abaa2a-cb66-4bc2-a61d-cfcd32e318e7)

proposal로 나온 bounding box를 \\(P_x,P_y,P_w, P_h\\)라 하자.

![image](https://github.com/ownvoy/ownogatari/assets/96481582/ef1459b6-656a-4ae1-a11e-c607201b5ca1)

ground truth로 나온 bounding box를 \\(G_x,G_y,G_w, G_h\\)라 하자.

P에서 G로 가는 transformation을 알아야 함.
$$\begin{array}{l}{{t_{x}=(G_{x}-P_{x})/P_{w}}}\\ {{t_{y}=(G_{y}-P_{y})/P_{h}}}\\ {{t_{w}=\log(G_{w}/P_{w})}}\\ {{t_{h}=\log(G_{h}/P_{h}).}}\end{array}$$

\\(\phi_{5}(P^{i})\\)이 \\(t_x, t_y, t_w,t_h\\)에 대한 linear function이라 할 떄, \\(\bf w\\)를 배우는 것이 목표. 이 때 \\(\bf w\\)가 큰 값이 나오지 않도록 regularization을 해줌.



$${\bf w}_{\star}=\mathop{\mathrm{argmin}}_{\hat{\bf w}_{\star}}\sum_{i}^{N}(t_{\star}^{i}-\hat{\bf w}_{\star}^{\mathrm{T}}\phi_{5}(P^{i}))^{2}+\lambda\left\|\hat{\bf w}_{\star}\right\|^{2}.$$


# 3. experiments

## 3.1 PASCAL VOC 2010

![image](https://github.com/ownvoy/ownogatari/assets/96481582/0fcfe868-106f-4f8d-9808-c296fafbf8da)


## 3.2 fine-tuning 없이도 잘 되는가?

![image](https://github.com/ownvoy/ownogatari/assets/96481582/0f9c07e2-1e71-40b4-9f55-1fa68dcbac90)

- CNN만으로도 생각보다 잘 됨. 근데 fine-tuning을 하면 더 잘 됨.
- ILSVRC(pretrain) => PASCAL(fine-tuning)의 순서로 학습.


