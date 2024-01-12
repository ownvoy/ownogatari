---
title: "FastText"
date: 2024-01-12T15:48:20+09:00
draft: false
katex: true
categories: "NLP"
---

# 1. overview

## 1.1 objective
단어를 형태소의 측면에서 표현해보자.


## 1.2 background
### 1.2.1 형태소적 접근
- 기존의 단어 표현은 하나의 단어당 하나의 벡터로 이루어졌다.
- 하나의 단어를 여러 형태소의 벡터로 표현해보자.
	- `먹음`에 대해서 `먹기`, `먹이`, `먹는`, `먹다`, `먹었음`, `먹었다` 등 여러 형태가 나타날 수 있다.
	- 그러나, 훈련 데이터에는 모든 형태가 등장하지 않는다.
	- 그렇게 되면, 등장하지 않는 단어에 대해서representation이 떨어질 수 있다.
	- 만약 형태소로 접근한다면, `-기`, `-이`, `-는`의 형태는 다른 단어들에서 볼 수 있다.
	- 즉, `먹었음`이 훈련데이터에 없다해도, `먹-`, `-었-`, `-음`의 합으로 표현할 수 있다.

### 1.2.2 Skip Gram

skipgram: 하나의 단어가 주어지면, 주변의 단어를 맞추는 것.

데이터가 주어졌을 때, 아래 식을 최대화하는 것이 목적.

$$\sum_{t=1}^{T}\sum_{c\in C_{t}}\log p(w_{c} \ | \ w_{t})$$

확률 \\(p\\)는 softmax로 표현 될 수 있다.

$$p(w_{c}\mid w_{t})=\frac{e^{s(w_{t},w_{c})}}{\sum_{j=1}^{W}e^{s(w_{t},j)}}$$

\\(s(w_t,w_c)\\)는 \\(w_t\\)와 \\(w_c\\)의 유사도를 구하는 함수이다.

$$s(w_{t},w_{c})=u_{w_{t}}^{T} v_{w_{c}}$$


# 2. main
general model + subword model

## 2.1 architecture
### 2.1.1 General model
변형 skipgram을 사용한다.

#### Binary log loss with negative sampling
기존의 skipgram은 Softmax log loss를 사용한다. 이렇게 되면, \\(w_t\\)에 대해서 하나의 \\(w_c\\)를 예측하는 것으로 학습이 된다. 

여러 개의 \\(w_c\\)를 예측하기 위해서는 \\(c\\)마다 binary class 예측을 해야한다. 따라서, loss를 binary loss로 바꿔준다.

$$\log\left(1+e^{-s(w_{t},w_{c})}\right)+\sum_{n\in N_{t,c}}\log\left(1+e^{s(w_{t},n)}\right)$$

- 첫번째 term: \\(w_t\\)와 \\(w_c\\)의 유사도를 비슷하게 만들도록 학습
- 두번째 term: \\(w_t\\)와 \\(n\\)의 유사도를 다르게 만들도록 학습

### 2.1.2 Subword model
우선 where이라는 단어가 있을 때, 이를 `<where>`로 바꿔준다. 이 때, `<`와 `>`는 경계를 나타낸다.
\\(\leftarrow\\) 형태소는 하나의 단어를 기준으로 하기 때문에 필요하다.

그 후 character \\(n\\)-gram을 도입한다.

\\(n=3\\)이라하면, `<where>`은 `<wh`, `whe`,`her`, `ere`, `re>`로 나누어진다. 각각을 \\(g_1,g_2,g_3,g_4,g_5\\)라고 하자.

또, `<where>` 역시 special sequence로 \\(g_6\\)라 하자.

이제 주변 단어에 대해서 where 한 단어와 비교하는 것이 아니라, \\(g_{1\sim}g_6\\) 와 비교하는 과정으로 바꿀 것이다.



$$s(w,c)=\sum_{g\in G_{w}}z_{g}^{T}v_{c}$$

이렇게 되면 장점은 나중에 못본 단어`<when>`가 있다고 할 때, `<where>`에서의 `<wh`,`whe` 를 재사용할 수 있게 된다. \\(\Rightarrow\\) out of vocabulary 문제 해결 가능.

# 3. experiments

우리의 모델 sisg는 human-judge와 유사도가 다른 모델들보다 크다.
![image](https://github.com/ownvoy/ownogatari/assets/96481582/d6f70da5-ca71-4156-80f4-eb934ea960df)

word anlalogy task에서도 syntatic에 대한 측면에서 우수한 점수를 보인다. symentic에 대해 점수가 낮은 것은 아무래도 의미 단위 없이 나누었기 때문으로 보임. 

__word analogy task__: A is to B as C is to D. ex) (왕,왕비), (할아버지,?)

![image](https://github.com/ownvoy/ownogatari/assets/96481582/152b129d-bc89-41a5-b03d-b409dcc9df59)
 
 \\(n\\)을 키우면 symentic에 대한 성능이 조금은 나아짐. (세로축: 단어의 char개수, 가로축: n-gram)
![image](https://github.com/ownvoy/ownogatari/assets/96481582/38e82290-195c-4f25-86bd-18dd318119ea)

적은 양의 데이터에서도 우수한 성능을 보인다. 이는 하나의 단어를 쪼개었기때문으로 보임.
![image](https://github.com/ownvoy/ownogatari/assets/96481582/e4dfa006-1800-4f8e-afe3-aa1c8fcc67e6)

