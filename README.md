# Galaxies-Classification-By-Using-Deep-Learning

---

# 1. 서론

CNN(Convolution Neural Network)은 합성곱 계층(convolutional layer)과 풀링 계층(pooling layer), 그리고 완전연결 계층(fully-connected layer)로 구성되어 특히 이미지데이터처럼 가까운 데이터끼리 서로 연관성이 높을 때 좋은 성능을 내는 모델이다. 그 구조의 예는 아래와 같다.

<img src="https://i.imgur.com/Ahe2Aoo.png" width="100%">

MLP에서 사용하는 완전연결 계층은 이미지 데이터를 학습하기에 문제점을 가지고 있다. 데이터의 형상이 무시된다는 것이 그것인데, 가로, 세로, 채널(색상)로 구성된 3차원 데이터인 이미지를 평평한 1차원 데이터로 평탄화해주기 때문이다. 그 다음 모든 뉴런과 연결되어 각각의 weight와 bias를 가지고 연산하기 때문에 형상을 무시하고 모든 입력 데이터를 동등한 뉴런(같은 차원의 뉴런)으로 취급한다.<br><br>
하지만 이미지는 분명 3차원 형상이며, 따라서 공간적 정보가 중요한 요소가 된다. 서로 인접한 픽셀끼리는 유사한 값을 가지고 있다던지, 반대로 서로 거리가 먼 픽셀끼리는 큰 관련성이 없다던지, RGB채널 각각은 서로 밀접한 관련이 있다는 등 수많은 공간적 정보가 존재한다. 따라서 완전연결 계층만으로는 해결할 수 없었던 문제를 해결해주는 방법이 CNN이다.<br><br>
Convolution Layer는 합성곱 계층과 풀링 계층으로 이루어져 있다. 합성곱 계층은 채널별로 filter를 가지고 있는데, 이 filter를 이용해 feature map과 합성곱을 통해 output을 도출하고 backpropagation을 통해 filter의 weight들을 학습시킨다. 이 filter는 n by n의 정사각형 형태이기 때문에 합성곱을 할 때 주위 인접한 데이터들을 기준으로 하기 때문에 공간적 정보를 반영하여 학습을 할 수 있는 것이다. 또한 풀링 계층은 가로, 세로 방향의 공간을 줄이는 연산으로써 feature map의 중요한 정보는 최대한 보존한 채 output을 도출해낼 수 있다.<br><br>
따라서 이번 프로젝트를 통해 CNN을 직접 구상해봄으로써 이미지 데이터를 모델에 학습 및 평가해보고 직면하는 문제점들을 해결해나가는 것을 중점으로 한다.

# 2. 프로젝트

이번에 수행한 프로젝트는 '딥러닝으로 은하 분류하기'이다. 이는 데이터사이언스 데이터와 문제를 해결하기 위한 질문식 가이드를 제공해주는 사이트인 DAFIT에 게시되어있는 프로젝트이다. 프로젝트 관련 설명은 아래 링크를 통해 볼 수 있다.

* 프로젝트 주소<br>
http://www.dafit.me/question/?q=YToxOntzOjEyOiJrZXl3b3JkX3R5cGUiO3M6MzoiYWxsIjt9&bmode=view&idx=2547211&t=board

# 3. 연구

[여기를 참조하세요](프로젝트 수행중)
1. modeling
2. load model
3. keras tuner

# 4. 결과

프로젝트 수행중

# 5. 고찰

프로젝트 수행중
