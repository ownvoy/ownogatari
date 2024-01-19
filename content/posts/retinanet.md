---
title: "RetinaNet"
date: 2024-01-03T05:27:37+09:00
draft: false
mathjax: true
categories: "Object Detection"
tags: ["CV"]
---

 
# 1. overview

## 1.1 objective

배경과 사물의 imbalance를 loss를 새롭게 만듦으로써 해결해보자.

## 1.2 background

### 1.2.1 Class Imbalance

object는 몇 개 안 되는데, 배경은 \\(10^4\sim10^5\\)개 정도 있다. 배경을 많이 학습하는 것은 모델 성능 향상에 큰 도움이 안 된다.


### 1.2.2 Robust Loss

Robust Loss: outlier(hard examples)를 down weight 해주는 loss
Focal Loss: inliner(easy examples)를 down weight해주는 loss

### 1.2.3 Cross Entropy Loss

#### Binary Cross Entropy Loss

$$ - (y\log(p)+ (1-y)\log(1-p)), \ where \  y\in \lbrace 0,1 \rbrace$$

이 식은 아래의 식과 똑같다.
![image](https://github.com/ownvoy/ownogatari/assets/96481582/1f2d4f0d-c828-4939-b59d-1be6597dccd3)


\\(p_t\\)를 아래와 정의 했을 때,

![image](https://github.com/ownvoy/ownogatari/assets/96481582/115af2ab-e721-441e-b3ba-5d33519ecb08)



 최종적으로 Cross entropy는 다음과 같다.
$$CE(p,y)= -\log(p_t)$$

Corss Entropy의 문제점은 easy example(\\(p_{t}> 0.5\\))에 대해서 그 loss값이 크다는 것이다.(아래 그림에서 파란색 plot) 이 많은 easy example이 모이면, 총 loss가 상당히 커지고, 모델의 성능을 저해한다.

![image](https://github.com/ownvoy/ownogatari/assets/96481582/6212b6c1-5841-4bf8-aa5d-fb6c98969d8e)

Focal Loss는 easy examples들을 down weight하는 것이다. 

> Note
- 기존의 해결방법: anchors들을 sampling(RPN, OHEM, SSD) 
- Focal Loss: 모든 anchor들 사용

# 2. main

## 2.1 Focal Loss

$$\mathrm{FL}(p_{\mathrm{t}})=-(1-p_{\mathrm{t}})^{\gamma}\log(p_{\mathrm{t}}).$$
1.  \\(p_t\\)가 작은 경우(hard positive example): \\((1-p_t)\\)가 거의 \\(1\\)이기에 loss는 영향을 받지 않는다.
2. \\(p_t\\)가 큰 경우(easy negative example): (1-\\(p_t\\))가 거의 0이기에 loss가 down weight된다.

\\(\gamma\\)가 \\(2\\)일 때에 성능이 가장 좋았다고 한다.
 
## 2.2 RetinaNet

![image](https://github.com/ownvoy/ownogatari/assets/96481582/43e7bbd0-1cc0-45a0-90c9-862b6b646acd)

ResNet(FPN) + class와 box를 예측하는 subnets


### 2.2.1 FPN

기존의 [FPN](https://ownogatari.xyz/posts/fpn/)과 다른 점은 계산상의 이유로 \\(P_2\\) level을 사용하지 않았다는 것과 추가적인 convolution을 통해, \\(P_6\\)과 \\(P_7\\)까지 사용한 것이다.

### 2.2.2 Class & box subnet

FCN의 구조를 통해 prediction을 수행함. \\(K\\)는 class의 개수이며, \\(A\\)는 anchor 개수이다.


# 3. experiments

![image](https://github.com/ownvoy/ownogatari/assets/96481582/014a4b0d-fdba-4662-9b5b-7a77ad39c529)

speed와 정확도 측면에서 좋은 성능을 보인다.


