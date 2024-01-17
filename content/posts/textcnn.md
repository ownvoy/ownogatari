---
title: "TextCNN"
date: 2024-01-17T20:52:14+09:00
draft: false
mathjax: true
---

# 1. overview

## 1.1 objective

- pretrained된 word2vec을 여러가지 task에 적용해볼까?

# 2. main

## 2.1 architecture
word2vec + cnn
### 2.1.1 word2vec
아래와 같이 사전학습된 word2vec을 사용할 것이다.
![image](https://github.com/ownvoy/ownogatari/assets/96481582/5a6fc355-b0aa-4900-a772-ce29db8cd73a)

이 word embedding은 학습을 하면서 고정될 수도 있고, 바뀔 수도 있는데 각각을 CNN-static, CNN-non-static이라고함.

또 CNN-static과 CNN-non-static 두개 다 사용할 수 있는데, 이 경우 multi channel임으로, CNN-multichannel이라고함.
\\(\Rightarrow\\) overfitting 방지(resnet 느낌)

### 2.1.2 CNN

CNN은 filter를 학습한다. 근데 filter의 크기가 \\(h\times k\\)로 다소 rough한 형태. 

또, convolutional layer 1개랑 pooling 1번, FCN 1번의 간단한 형태. 아마 task가 간단해서 모델이 간단해도 잘 돌아가나봄.
![image](https://github.com/ownvoy/ownogatari/assets/96481582/0f767db8-75ea-4102-afd0-6a4b87630136)

이 모델에서, pooling을 해주는 이유는 filter의 사이즈가 다르기 때문임. (그림에서는 \\((2\times k), (3\times k)\\))

# 3. experiments
![image](https://github.com/ownvoy/ownogatari/assets/96481582/179e12e0-b9aa-4103-b4b1-9a768dafa739)
using pretrained model is good!

__word embedding이 feature들을 잘 represent함. transformer에 cnn backbone붙이는 것처럼, word embedding이 두루두루 활용되는가 보다.__

![image](https://github.com/ownvoy/ownogatari/assets/96481582/bd5b6fbc-11a1-4375-b594-879ce89a8ca4)

워드 임베딩을 고정시킨 것(static)과 훈련시킨 것(Non-static)의 차이를 보여주는 table임.

흥미로운 것은 `good`같은 경우 닮은 것이 원래 `great`였는데 SST-2로 train 후 `nice`로 바뀜. 이는 SST-2  test가 very positive, positive을 구별하기 때문으로 보임.

또, `n't`, `!`, `,` 같은 경우 random으로 init된게, 학습 후 representation이 meaningful한 것을 볼 수 있음.




