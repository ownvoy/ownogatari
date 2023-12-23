---
title: "Mask R-CNN"
date: 2023-12-23T13:42:52+09:00
draft: false
---



# 1. overview

## 1.1 objective
- Faster R-CNN 돌리는김에 segmentation 해보자.
## 1.2 background

|           | detection                             | sementic segmentation                | instance segementation |
| --------- | ------------------------------------- | ------------------------------------ | ---------------------- |
| 대상      |각각이 어떤 물체인지 | 각각이 어떤 클래스이고, instance를 구별x                   |          각각이 어떤 클래스이고, instance를 구별          |
| 범위      | bounding box마다                      | 전체의 pixel마다                       |            bounding box의 pixel마다            |
| 대표 모델 | `Faster R-CNN`                          | `FCN`                                  |             `Mask R-CNN` , `FCIS`          |
|           | ![image](https://github.com/ownvoy/ownogatari/assets/96481582/5055728a-5f85-4229-89dc-f69a23ab74d6) |![image](https://github.com/ownvoy/ownogatari/assets/96481582/f92c0342-3950-4b65-9a6b-fec20e00efa6) |          ![image](https://github.com/ownvoy/ownogatari/assets/96481582/acd1d993-1a08-48c1-b019-b4927ba94459)          |

- Fater R-CNN: sementic segmentation 못함. (RoIPooling이 한 뭉태기로 하니까, pixel에 대한 각각의 정보 소멸)
- FCN: instance segmentation 못함. (전체 픽셀별로 하니까)

# 2. main

mask R-CNN은 RoI마다 mask(K class에 대해)를 내놓기에, instance segmentation 가능함. 
## 2.1 architecture
기존의 Faster R-CNN에서 2가지가 바뀜.
1. mask branch가 추가됨.
2. RoI pooling 대신 RoI Align을 사용.

### 2.1.1 mask branch
![[Pasted image 20231223130351.png]]
- $14 \times 14$ 에는 $0,1$의 값이 들어 있음. 이 부분이 해당 클래스냐 아니냐
- Mask R-CNN은 mask, class, box prediction이 parallel하게 이루어짐.
- 다른 모델은 보통 mask에 대해 prediction을 내놓음 $\Rightarrow$ segementation과 detection이 합쳐진 구조.
- classification이  calss를 내놓으면, 해당 class를 mask에서 갖고 오는 방식
$$L=L_{c l s}+L_{b o x}+L_{m a s k}$$
### 2.1.2 RoI pooling
![image](https://github.com/ownvoy/ownogatari/assets/96481582/979027ef-175a-499d-8302-cfa001f3fd00)
- Faster R-CNN의 RoI pooling은 2번의 quantization이 일어나기에 픽셀에 대해 정보의 손실이 일어남.
- Faster R-CNN의 목적 자체가 pixel별로 무슨 class를 보는 것에 초점이 x
### RoI Align
![image](https://github.com/ownvoy/ownogatari/assets/96481582/63e96397-59c4-4e5a-8a5d-74b211c63465)
- pixel의 정보를 살리기 위해, quantizaion이 없어야 함.


# 3. experimental results

![image](https://github.com/ownvoy/ownogatari/assets/96481582/8ed2d5ea-dfb7-456a-9a9f-334b467311b4)

- Instance segmentation에 있어서, 기존의 SOTA보다 좋음.

![image](https://github.com/ownvoy/ownogatari/assets/96481582/5400579c-d0dd-4b61-b830-8fec2e131821)
- object detection 또한 기존의 SOTA보다 좋음.

![image](https://github.com/ownvoy/ownogatari/assets/96481582/79165c75-45ee-493b-89cf-e6737aa34801)
- Mask R-CNN은 object가 overlap 됐을 때도 성능이 좋은데, 이는 RoI에 대해서, 하나의 제일 높은 class를 정하는 방식이 아니기 때문인 것 같음.(즉 Mask R-CNN은 segmentation과 classificatioin이 결합된 방식이 아니다.)

