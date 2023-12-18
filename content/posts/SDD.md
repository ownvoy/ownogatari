---
title: "SDD"
date: 2023-12-19
draft: false
katex: true
mathjax: true
ShowToc: true
---

# 1. overview

## 1.1 objective
1. multiple scale에 대해 다뤄 보자. (큰 물체, 작은 물체)
\\(\Rightarrow\\) 장점: low resolution에 대한 문제를 해결
\\(\Rightarrow\\) 방법: 여러 개의 conv layer 도입

2. different shape에 대해 다뤄 보자. (박스 형태, 2:1, 1:1)
\\(\Rightarrow\\) 장점: 다양한 형태의 객체 검출
\\(\Rightarrow\\) 방법: 박스 형태 다양하게


## 1.2 background
### YOLO
YOLO 같은 경우 feature map이 \\(7\times7\\)짜리 1개이다. box개수는 각 cell마다 2개 있어서 총 98개이다.

![image](https://github.com/ownvoy/ownogatari/assets/96481582/da6b123f-8458-4380-bf54-2d0fde560084)

# 2. main
## 2.1 architecture
pretrained model + convolutional layer
논문에서는 VGG를 pretrained model로 사용

\\(\text{Conv}4_{3}\rightarrow \text{Conv}6 \rightarrow \text{Conv}7 \rightarrow \text{Conv}8_{2}\rightarrow \text{Conv}9_{2}\rightarrow\text{Conv}10_2\\)
라 할 때

| Conv  |    feature map size    |      filter size(다음 conv로)       |      number of box      |             output size              |
|:-----:|:----------------------:|:----------------------:|:-----------------------:|:------------------------------------:|
| Conv4 | \\(38\times38\times 512\\) | \\(3\times 3\times 1024\\) | \\(38 \times 38 \times 4\\) | \\(38\times38\times4\times(C+4)\\) |
|  Conv6     |    \\(19 \times19\times1024\\)                    |          \\(1\times1\times1024\\)              |            \\(19\times19\times6\\)             |             \\(19 \times 19 \times 6 \times (C+4)\\)                         |

각 feature map의 cell 마다 bounding box가 4 or 6개가 있음.
\\(\Rightarrow\\) YOLO보다 압도적으로 많은 bounding box 개수
![image](https://github.com/ownvoy/ownogatari/assets/96481582/05f2fc35-e7c8-42a8-a48e-c6776ce45e4b)



큰 feature map 더 많은 디테일을 담고 있으므로 작은 물체를, 작은 feature map은 큰 물체를 검출.

# 3. other detail

## 3.1 loss
bounding box offset하는 거 4개랑 confidence loss weighted sum해서 Smooth L1 loss 적용
$$L(x,c,l,g)=\frac{1}{N}(L_{c o n f}(x,c)+\alpha L_{l o c}(x,l,g))$$

## 3.2 Hard Negative Mining
negative sample이 압도적으로 많으니까, confidence가 높은 순서로 정렬해서 하나만 사용
\\(\rightarrow\\) negative:positive = 3:1 

## 3.3 bounding box 만들기
각각의 bounding box의 크기랑 비율이 상이.

#### scale 정하기

![image](https://github.com/ownvoy/ownogatari/assets/96481582/e2c6a68f-cf96-476e-b96c-a19c15c3b515)

앞 단의 feature map(lower layer)은 box scale이 작음. 반면 뒷단의 feautre map(higher layer)은 box scale이 큼.
$$s_{k}=s_{{min}}+{\frac{s_{{max}}-s_{{min}}}{m-1}}(k-1),\quad k\in[1,m]$$
- 6개의 feature map이 있다고 할 때, lowest layer은 box scale이 \\(s_{min}=0.2\\) 이고 highest layer은 box scale이 \\(s_{max}= 0.9\\)

### box 형태 정하기
$$a_{r}\;\in\;\{1,2,3,\frac{1}{2},\frac{1}{3}\}$$
를 기준으로 \\((w_{k}^{a}=s_{k}\sqrt{a_{r}})\;\;,(h_{k}^{a}=s_{k}/\sqrt{a_{r}})\\) 를 총 5개의 박스를 만들어줌.

\\(a_r=1\\)일때 \\(s_k=\sqrt{s_ks_{k+1}}\\) 도 추가하여 총 6개의 box 설정

기본적으로 6개의 box를 사용하였는데, 4개의 box를 쓸 때에는, 
\\(a_{r}\;\in\;\{1,2,\frac{1}{2}\}\\) 로 설정.


# 4. experimental results

![image](https://github.com/ownvoy/ownogatari/assets/96481582/b67df078-ee95-4065-96a8-ad982613da4a)
성능이 R-CNN 계열 , YOLO보다 좋다~

![image](https://github.com/ownvoy/ownogatari/assets/96481582/f8b7ced1-6b7a-4bdd-9a19-2e733bf00658)
Inference할 때 정확도도 높으면서 속도도 빠르다~
