---
title: "DETR"
date: 2024-01-19T13:28:40+09:00
draft: false
math: katex
katex: true
categories: "Object Detection"
tags: ["CV"]
---

# 1. overview

## 1.1 objective

- detection에서 transformer를 사용해보자.
- detection에서 end to end 학습을 해보자.

NMS, anchor setting의 부차적인 과정들이 학습의 결과에 많은 영향을 미친다. Transformer를 통해 위 두 개의 과정을 없앨 수 있게 된다.

# 2. main

## 2.1 Transformer
![image](https://github.com/ownvoy/ownogatari/assets/96481582/3c6be95f-25a9-466f-a7cd-32a6946f8cef)
DETR은 기본적으로 Transforemer 구조를 따른다. DeTR에서 Encoder와 Decoder가 무슨 역할을 하는지 보도록 한다.
### 2.1.1 Encoder
- cnn을 통과한 feature-map이 쪼개져서 input으로 들어온다. (ViT 스타일)
- 그 후, attention 계산을 통해 그림 전체에 대한 정보를 본다.

- Encoder에서 self attention이다. 물체들을 구별하는 것을 볼 수 있다.
![image](https://github.com/ownvoy/ownogatari/assets/96481582/5c4633b1-d90e-4385-862c-0e8ba375e9c4)


### 2.1.2 Decoder

- decoder의 query는 output으로 box와 class를 내놓는다.(decoder의 query가 일반 Transformer에서 EOS1, EOS2, ...,EOSN과 같은 느낌을 받음. 
- output이 sequence 형태가 아니니까 inference할 때도, parallel하게 할 수 있음. DETR은 auto-regressive하지 않음. 
- object query는 각각의 고유한 것을 예측해야하기 때문에 learnable positional embedding임.
![image](https://github.com/ownvoy/ownogatari/assets/96481582/7514a9b5-bd64-4991-b90f-b98ae5b31bca)
- encoder의 사진 전체에 대한 정보를 참고하여, output을 뽑는다. output의 attention은 사물의 가장자리를 보는 것을 알 수 있다.
- __img에서 prediction이 나옴으로, anchor setting에서 자유롭다. anchor 자체가 없다.__

## 2.2 Set prediction

decoder의 쿼리가 들어가면 set으로 (box,class)가 나온다.

decoder의 쿼리 수 \\(N\\)은 무조건 사진의 object 수보다 크게 설정한다. object가 없을 경우 \\(\emptyset\\)(no object)으로 매칭
![image](https://github.com/ownvoy/ownogatari/assets/96481582/899559be-99ec-409d-a73a-0dce868eae8d)

### 2.2.1 Set prediction의 이점
- __일대일 매칭이어서 NMS를 피할 수 있다.__


### 2.2.2 Set prediction loss

\\(y\\)랑 \\(\hat{y}\\) 를 가능한 1-1 매칭을 해보고, loss가 가장 적은 정책을 \\(\hat{\sigma}\\)라고 한다. 이는 Hungarian algorithm를 통해 사용.

\\[\hat{\sigma}={\underset{\sigma\in S_{N}}{\arg\min}}\sum_{i}^{N}L_{match}(y_{i},\hat{y}_{\sigma(i)})\\]

가장 최적의 \\(\hat{\sigma}\\)를 통해 답과의 loss를 구한다.


//[L_{Hungarian}(y,\hat{y})=\sum_{i=1}^N\left[-\log\hat{p}_{\hat{\sigma}(i)}(c_i)+1_{c_i\neq\emptyset}L_{box}(b_{i},\hat{b}_{\hat{\sigma}}(i))\right]//]

class가 맞으면, Loss가 작아지는 식의 Cross Entropy Loss + box의 차이가 작으면, loss가 작아지는 식이다.

bounding box loss

//[\lambda_{iou}L_{iou}(b_i,\hat{b}_{\sigma(i)})+ \lambda_{L1}\mid\mid b_{i}-\hat{b}_{\sigma(i)}\mid\mid_1//]

그냥 L1 loss만으로는 scale에 대해 영향을 많이 받으므로 generalized iou 도입.

# 3. experiments

![image](https://github.com/ownvoy/ownogatari/assets/96481582/9cff582c-fa59-486d-b616-00b448bbf9ff)

DETR는 비슷 한수의 param과 FLOPS를 가진 Faster RCNN보다 성능이 좋은 것을 볼 수 있다. 다만, 작은 물체의 경우 Faster RCNN의 결과가 좋다.


