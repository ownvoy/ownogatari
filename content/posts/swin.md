---
title: "Swin Transformer"
date: 2024-01-29T23:20:05+09:00
draft: false
katex: true
---
 
# 1. overview

## 1.1 objective

- Transformer에 inductive bias를 주입하자.
- Transformer의 계산 효율성을 높여보자.

## 1.2 background

### 1.2.1 ViT
기존 ViT의 문제점으로 2가지가 있다.
1. Fixed scale
- 기본적으로 16크기의 Patch를 쓴다.
- CV같은 다양한 scale로 보는 것이 중요하다.(ex: FPN)
2. complexity
	- Image Segmentation 이나, Depth Estimation 같은 경우, 높은 해상도를 필요로 한다.
	- ViT의 attention 경우 \\(O(N^2)\\)의 시간복잡도를 가진다.

Swin Transformer의 경우 Patch Merging을 통해 (1)의 문제를 Window Attention을 통해 (2)의 문제를 해결한다.

# 2. main

## 2.1 architecture
### 2.1.1 Window Attention
![image](https://github.com/ownvoy/ownogatari/assets/96481582/4c744e0b-1cec-4736-9527-6a9ec644f8a4)
- 그림과 같이 각각의 색깔 cell들을 window라고 한다.
- 그림에서는 Window 하나 당 2 * 2의 patch를 가지지만, 보통 7 * 7의 patch를 가진다.
- __가장 큰 특징으로는 window 안에서만 attention이 일어난다.__ \\(\Rightarrow\\) 기존의 attention은 patch마다 attention을 수행했기에 \\(O((HW)^2)\\)의 시간복잡도를 가지는데, Window attention의 경우,\\(O(M^2)\\)로 시간 복잡도가 줄어든다.

### 2.1.2 Patch Merging
- Window attention을 하게 되면, 국소적으로만 attention을 하게 되는 문제점이 있다.
- 이를 해결하고자, patch 사이즈를 (4->8->16)키우게 된다.
- 이는 scale에서도 상당한 이점이 있다.
- 단순히 patch 사이즈를 키우는게 아니라, 차원을 더 깊게 만들어준다.(Hierachical한 구조)
![image](https://github.com/ownvoy/ownogatari/assets/96481582/f8e801c6-f55a-47c1-9465-5221af0a0f67)

### 2.1.3 Patch Merging + Window Attention

![image](https://github.com/ownvoy/ownogatari/assets/96481582/7d9bb5a6-25b0-4ac4-bfbd-9c3ef7f662de)
전체적인 구조는 Patch Merging과 Window attention이 반복되는 구조이다. 여기에 Shifted windows라는 것 추가된다.

### 2.1.4 Shifted Windows
![image](https://github.com/ownvoy/ownogatari/assets/96481582/f0835c9e-2331-436f-a6dd-ea785fe169e2)

그러나, 이러한 형태의 window attention은 문제가 있다. 

![image](https://github.com/ownvoy/ownogatari/assets/96481582/36042a13-29f9-455b-b042-4b3216d83980)

위와 같이, 각각의 patch들은 가까이 있음에도 attention을 안하게 된다.(circle)

이를 해결하기 위해서는 shifted window를 도입하여, 저것들도 attention을 해주면 된다.
![image](https://github.com/ownvoy/ownogatari/assets/96481582/dc2c4a2a-97f7-4d86-98b0-dc18dc2f44b8)

이것이 shifted window인 이유는, 기존의 window를 window size/2 만큼 오른쪽으로 아래쪽으로 움직이기 때문이다.

![image](https://github.com/ownvoy/ownogatari/assets/96481582/1f4829ed-7791-454e-aab2-6285add8162c)

### 2.1.5 Batch cyclic-shift

![image](https://github.com/ownvoy/ownogatari/assets/96481582/dc2c4a2a-97f7-4d86-98b0-dc18dc2f44b8)
Shifted Window에 문제가 있다.
1. 각각의 window의 크기가 다르다.
2. window의 수가 늘어난다. \\(\Rightarrow\\) attention 횟수가 늘어난다.

Cyclic shift를 써서 이 문제를 해결한다.
![image](https://github.com/ownvoy/ownogatari/assets/96481582/cd4dda32-77f1-484d-aaee-3740711bc5e8)
왼쪽과 위쪽의 윈도우들을 오른쪽으로 아래로 민 것이다.
이렇게 했을 때 위의 문제를 해결할 수 있다.
1. 2 * 2로 window size를 다시 맞춰준다.
2. 원래의 인접하지 않는 윈도우는 masking을 해줘서 attention 계산을 해준다. (attention 횟수 유지)
# 3. experiments

## 3.1 model variants

- Swin-T: C=96, layer numbers={2,2,6,2}  
- Swin-S: C=96, layer numbers={2,2,18,2}  
- Swin-B: C=128, layer numbers={2,2,18,2}  
- Swin-L: C=192, layer numbers={2,2,18,2}

각 layer가 짝수 인것은 window msa랑 shifted window msa가 쌍으로 이루어지기 때문이다.

## 3.2 experiments

![image](https://github.com/ownvoy/ownogatari/assets/96481582/3ce56043-c9ab-48de-8d46-dc8dc11224f7)
classification에서 정확도랑 속도가 높은 것을 확인할 수 있다.


![image](https://github.com/ownvoy/ownogatari/assets/96481582/2f4857d6-62ce-4cae-8fee-6a59795088c9)
object detection task에서 swin transformer가 backbone으로 쓰였을 때.

![image](https://github.com/ownvoy/ownogatari/assets/96481582/37533dab-7a67-4039-a4de-f1854521a9fc)
shifted windows를 썼을 때의 정확도가 높은 것을 볼 수 있다.

![image](https://github.com/ownvoy/ownogatari/assets/96481582/7f00a676-32f1-4edb-909a-3e0066fb1136)
cyclic shift를 썼을 때 속도가 빠른 것을 볼 수 있다.


