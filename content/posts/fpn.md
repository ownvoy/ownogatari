---
title: "FPN"
date: 2023-12-26T06:25:59+09:00
draft: false
katex: true
mathjax: true
categories: "Object Detection"
tags: "paper"
---


# 1. overview


## 1.1 objective

- resolution과 semantic의 trade-off를 줄여보자.
\\(\Leftrightarrow\\) 모든 resolution(scale)에서 강한 semantic 정보를 얻어보자.

## 1.2 background

### 1.2.1 resolution semantic trade-off

\\(\text{resolution} \downarrow \ \ \Rightarrow \ \ \text{semantic} \uparrow (😀)  , \ \ \text{detail} \downarrow (🙁)\\)
\\(\text{resolution} \uparrow \ \ \Rightarrow \ \ \text{semantic} \downarrow (🙁) \ \, \text{detail} \uparrow (😀)\\)

아래 table의 각 col들은 같은 의미임. 

|                  |                   |
| ----------------- | ------------------ |
| higher resolution | lower resolution   |
| weak semantic     | strong semantic    |
| low level feature | high level feature |
| detail            | abstract           |
| lower layer                  |        higher layer            |

### 1.2.2 related works

![image](https://github.com/ownvoy/ownogatari/assets/96481582/e6d0a70f-9704-4734-ad15-92b7f1872534)

a. __Feautrized image pyramid:__ 각각의 scale들마다 feature를 뽑아서 속도가 느림.
b. __Single feature map:__ 모든 층이 rich semantic하지 않음
c. __Pyramidal feature hierarchy:__ 모든 층이 rich semantic하지 않음 (또, low level에 대한 feature x)   
ex) [SDD](https://ownogatari.xyz/posts/sdd/#2-main): backbone 갖다 붙힘 
d.  __Feature Pyramid Network__: 모든 층이 rich semantic함. (low level feature + high level feature)

# 2. main

Q) 어떻게 하면 모든 층이 rich semantic 할 수 있을까?
A) lower 층의 feature map이 rich semantic 하면 됨.
방법) lower 층 feature map +=  higher 층 feature map

## 2.1 architecture


Bottom-up pathway + Top-down pathway and lateral connections.

- Bottom-up pathway: 올라가면서 여러가지 층 만들어주는 것.
- Top-down pathway and lateral connections: higher 층의 feature map을 더해주는 것. ![[Pasted image 20231226072231.png]]


### 2.1.1 Bottom-up pathway

![image](https://github.com/ownvoy/ownogatari/assets/96481582/ee64f1a9-4f11-4b39-ae56-e3bf0f783e43)
conv를 통과하고 나온 output: \\(\{C_{2}, C_{3}, C_{4}, C_{5}\}\\)
각각의 input에 대한 stride \\(\{4,8,16,32\}\\)

### 2.1.2 Top-down pathway and lateral connections
![image](https://github.com/ownvoy/ownogatari/assets/96481582/a5a6942f-039e-43f1-a696-e8f4e7489e4d)

Top-down pathway: upsampling해서 사이즈를 키움. (방법: nearest neighbor upsampling)
later connection: \\(1 \times 1\\) conv 돌려서, channel 맞춘다음, Top-down에서 내려오는거랑 더해줌.

__\\(\Rightarrow\\)각 층의 feature map은 lower layer의 localized 정보와 higher layer의 semantic을 둘 다 가질 수 있음.__

각 층의 final feature map(\\(\{P_{2}, P_{3}, P_{4}, P_{5}\}\\))은 \\(3 \times 3\\)의 convolution을 거친 후 완성.

## 2.2 application

지금까지 과정은 하나의 backbone을 만든 것.
\\(\Rightarrow\\) FPN을 RPN과 head를 연결 해주는 과정이 필요.

### 2.2.1 FPN for RPN
![RPN](https://github.com/ownvoy/ownogatari/assets/96481582/415aaa54-dfe9-4264-bc08-1d85ad0f4070)

- 기존의 RPN: 하나의 feature map에 대해 \\(3\times3\\)의 convolution과 \\(1\times1\\)의 convolution을 돌렸음. ([사진](https://herbwood.tistory.com/10))
- 수정된 RPN: 여러개의 feature map에 대해 \\(3\times3\\)의 convolution과 \\(1\times1\\)의 convolution을 각각 돌리기.

- 기존의 RPN: 하나의 feature map에 대해 여러 scale의 anchor size
- 수정된 RPN: 하나의 feature map에 대해 하나의 scale의 anchor size (\\(\{P_2, P_3, P_4, P_5, P_6\}\\)에 대해 \\(\{32^2, 64^2, 128^2, 256^2, 512^2\}\\) anchor  size 설정)


### 2.2.2 FPN for RoI pooling
- 기존의 RoI pooling : 하나의 feature map에 대해 적용했었음
- 수정된 RoI pooling: 여러 개의 feature map들 중 어느 feature map을 쓸 것인지 정해야함.

$$k=[k_{0}+l o g_{2}(\sqrt{w h}/224)]$$
- RoI의 weight와 height에 따라 \\(k\\) level의 feature map을 정함.


# 3. experiments
![image](https://github.com/ownvoy/ownogatari/assets/96481582/5412c9a6-1bd0-42d2-9628-8d1ae56b1a95)
성능이 압도적으로 좋은지는 잘 모르겠음. 

iterative regression, hard negative mining, context modeling, stronger data augmentation 같은 기법들 안 쓰고 순수하게 이겼다고는 언급함.

![image](https://github.com/ownvoy/ownogatari/assets/96481582/0e21e7cd-9b9e-4269-9fd5-d0857eae0aee)
RPN 부분에 대한 실험

![image](https://github.com/ownvoy/ownogatari/assets/96481582/40a6aa4d-786a-485a-ab33-78ce98ddbefe)
RoI pooling에 대한 실험(only finest level과 비교하면 진짜 별 차이가 없음.)
