image_segmentation_project


=============



# 1.Intro

Image classification, Image detection, Image captioning등 다양한 딥러닝 응용분야 중, Image segmentation 문제를 직접 부딪혀 보기로 하였다.



즉 구글링해서 튜토리얼 코드 찾아서 기웃거리면서 슬쩍 구현해보고 넘어가는 것이 아니라, 데이터셋 수집부터 파일입출력, 신경망 모델링, 학습과 Fine-Tuning에 이르는 딥러닝 문제해결의 파이프라인을 직접 경험해보겠다는 것이다.



# 2.FCN

우선은 신경망 기반 Image segmentation의 pioneer work인 FCN을 구현해보기로 하였다.



## 2.1.Paper Reading

### 2.1.1.Summary

이 논문의 Motivation은 기존에 Image classification 문제에만 사용하던 인공신경망을 Image segmentation에 응용해보려고 한 것이다. 
이 논문의 가장 핵심적인 아이디어는 Classification 신경망의 FC(MLP)부분을 1x1 Convolution layer로 대체하여 최종출력이 Class 수 만큼의 Feature Map(Activation Map)으로 나오게 만드는 것이다.



두 번째로 핵심적인 아이디어는 Prediction을 하느라 작아진 Feature Map의 크기를 원본크기만큼 원상복구 할 때, 학습가능한 Transposed_convolution(Deconvolution) layer로 원상복구하는 것이다.



저자는 이 방법 외에도 Overfeat 논문에서 영감을 받아 Shift-and-Stitching을 유사하게 구현한 방법도 실험해보았지만 효과가 별로 없었다고 한다.
세 번째로 핵심적인 아이디어는 Transposed_convolution으로 피쳐맵을 Upsample할 때, Downsample되기 전의 앞단으로부터, Skip connection을 만들어, 보다 고해상도의 정보를 Upsample하는데 활용하였다는 것이다. (개인적인 생각으로는 이 뿐만 아니라, ResNet과 유사하게 Gradient flow를 좋게 만드는 효과도 있을 것 같다. 이 논문은 ResNet이 나오기 전에 쓰여진 논문인데, 아마 Skip connection의 정확한 효과를 모두 이해하고 쓰진 않았을 것 같다. 물론 지금도 우리 모두가 잘 모르긴 마찬가지지만)

### 2.1.2.Implemation

VGG16 기반의 Fine-tune된 신경망을 가져와서 사용한다. 이 논문 자체가 애초에 Classification에 사용하던 CNN의 앞단(특징추출부)을 Segmention에 사용해보기 위한 논문이다. 아무래도 같은 신호(영상)를 처리하던 신경망이니 General한 Feature를 잘 뽑기만 하면 어떤 용도로도 잘 사용할 수 있다고 본 것 같다. 실제로 앞단을 고정해 놓고 1x1 Convolution layer와 Upsample layer만 학습시켜보기도 했는데도 꽤 괜찮은 결과가 나왔다고 한다.



Optimizer는 Momentum Optimizer, learning rate 0.0001에 momentum 0.9, weight decay 5^(-4), 학습시 bias들에 learning rate를 2배로 적용하였다.
1x1 Convolution layer는 0으로 초기화했고,  Upsample layer는 Bilinear Interpolation 필터의 가중치로 초기화했다.



Patchwise training을 구현하기 위해, Loss의 일부를 무시해버리는 방법을 사용해봤지만 별 효과도 없고, 수렴하는데 필요한 이미지만 많아져서 그냥 안 쓴다고 한다.

### 2.1.3.Question and Motivation

Data Augmentation 방법으로 미러링(Mirroring)과 지터링(Jittering)을 사용했다. 여기서 든 의문은 각 축으로 최대 32픽셀씩(Downsample된 크기만큼) Jittering해봤다고 되어있는데 이게 사실 Upsample만 안하면 Shift-and-Stitching이랑 근본적으로 동일한 방법(여러 offset에서의 loss를 고려하므로) 아닌가하는 생각이 든다. 별효과가 없었다고 하긴 하는데... 사실 이 논문에선  Shift-and-Stitching을 적용하지 않았다곤 하지만 사실상 적용한 것과 유사하다고 볼 수 있지 않을까하는 생각이 든다.



또 다른 의문은, 이 논문에서는 Training from scratch(Fine-tune된 앞단을 가져다쓰지 않고 전부 Random Initialize된 상태에서 부터 학습하는 것)가 불가능하다고 하는데, 그 이유로 엄청난 시간 자원을 들고 있다. 그렇다면 시간만 많으면 Training from scratch가 가능하다는 것인가? 아무리 구글링을 해봐도 어떤 FCN코드도 scratch에서부터 training하는 예제를 찾아볼 수 없었다. 이것이 내 프로젝트 첫 구현의 주요 Motivation이 됐다. 남들이 다 해봤던 FCN 구현이지만 내 목표는 Training from scratch로 어느 정도 괜찮은 성능을 내보는 것이다.



## 2.2.Data

Data를 모으고 학습에 이용할 수 있도록 전처리하는 것 부터가 쉽지가 않은 작업이었다. 일단은 PASCAL VOC의 Semantic segmentation 데이터셋을 사용하기로 하였다. 근데 남들이 만들어 놓은 API보는게 여간 짜증나는 일이 아니었다. '내가 그래도 학교에서 배운거라곤 영상처리 하는 것 뿐인데'하는 생각에 데이터셋을 읽어서 학습에 이용하는 모듈을 직접 만들었다.(사실 중간중간에 다른 API를 참고하긴 했다.) 솔직히 좀 허접하지만 Data flow를 알고 쓴다는 느낌이 좋다. Augmentation 하는 것은 뒤로 미뤄두고, 일단은 그냥 일괄적으로 Resizing해서 배치로 때려박았다.



**Resizing할 때 가장 조심할 점은 원본 이미지는 Interpolation Method를 뭘 써도 상관이 없지만 Ground Truth 이미지는 NN으로 Interpolation해야 한다. 그렇지 않으면 Class의 경계선마다 보간되어서 아무 클래스에도 속하지 않는 색깔을 가진 픽셀들이 나오게 된다.**


## 2.3.Model
-이미지



사실 인터넷에 구현되어 올라와있는 구조들을 보면 VGGNet기반의 신경망에서 Transfer시키려다 보니 구조를 변형하는데 한계가 있고, Receptive filed 크기를 맞춰주느라 1x1 Convolution layer의 첫 layer의 필터 크기가 1x1이 아닌 7x7인 등 다소 이해가 가지 않는 구조를 가지고 있다.
난 어차피 Training from scratch가 목적이니 1x1 Convolution layer는 모두 1x1로 필터크기를 통일할 것이고 Batch Normalization도 넣어보고 활성화함수도 살짝 변형해보고 그냥 내 마음대로 해볼거다.

## 2.4.Training
GTX1080에서 배치사이즈 15로 놓고 학습시키다가 컴퓨터 파워가 나갔다. 새 컴퓨터에 환경구성을 다시했다. 소심해져서 배치사이즈 5로 놓고 돌리다가 다시 10으로 늘렸다. 배치놈은 배치크기에 민감하다는 점을 주의하여야 할 것 같다. 체크포인트마다 100장의 Validation set을 불러와 Validation IOU를 계산하여 Training loss, Training batch IOU와 같이 기록하도록 했다.


## 2.5.Fail(...)

Training Loss는 계속 줄어드는데 Validation IOU가 20% 언저리에서 원체 늘지를 않는다. 그냥 Training data에만 계속 오버피팅되고 있다. Tensorflow KR에 도움도 구해보고 개인적으로 이런저런 자료를 찾아봤지만 아무래도 데이터가 적은 것이 원인인 것 같다. PASCAL VOC의 클래스는 20개, Training data는 1800개 정도이다. 클래스 당 90개면 적긴 한 것 같다. 일반적으로 Semantic segmentation이 일반 Image classifacation보다 데이터가 더 많이 필요할 것 같기도 하고.
확실히 단순히 시간만 많이 투자해서 에폭 많이 돌린다고 Training from scratch가 되는 것 같진 않다. 논문은 Fine-tune된 앞단을 가져다 썼기 때문에  이 정도 데이터만으로도 어느정도 결과가 나왔던 것으로 보인다.


나는 2가지 방법으로 문제를 해결해 보려고 한다. 



먼저 엄청 적은 데이터로도 Training from scratch에 성공했다는 논문이 있다. 벤지오 아저씨 제자가 쓴 논문이다. DenseNet을 FCN에 적용한 것으로 보인다. 내가 보기엔 꽤 획기적인데 국내엔 별로 알려지지 않은 것 같다. 한가해지는대로 최대한 빨리 내가 한국 최초로 리뷰 및 구현을 해봐야겠다. ㅋㅋㅋ(The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for  Semantic Segmentation(<https://arxiv.org/abs/1611.09326>))



그리고 또 하나의 방법은 그냥 더 많은 데이터를 때려박아보는 방법이다. 이래도 안되면 FCN은 오직 Fine-tune된 앞단을 가져다 썼을 때만 학습이 가능한 신경망이라고 봐야하는 것 아닌가하는 생각이 들고, 이 가설이 나름대로 훗날의 연구거리가 될 수도 있을 것 같다.(왜 안 되는가)



A Review on Deep Learning Techniques Applied to Semantic Segmentation(<https://arxiv.org/abs/1704.06857>) 이 논문은 다양한 Segmentation용 Dataset들 뿐만 아니라 2017년 기준 최신 Semantic Segmentation 기법들에 대해 전반적으로 Overview하고 있는 논문이다. 그나마 샘플이 많은 데이터셋 중 Cityscapes는 뭔가 주행화면이라서 마음에 안 들고 SYNTHIA는 가상데이터라서 Domain Adaptation이 필요하다. MS-COCO에서 일부 레이블만 사용하는 것이 제일 괜찮아보인다.




## 2.6.MS-COCO

많은 시행착오 끝에 MS-COCO Dataset을 다룰 수 있게 되었다. 분명 마이크로소프트에서 만든 데이터셋일텐데 왜 윈도우랑 호환이 잘 안되게 만들어놨을까?
MS-COCO 설치과정은 별도로 업로드를 해두어야겠다.



MS-COCO로부터 PASCAL VOC와 겹치는 클래스들만 따로 추출해서 90000장 정도의 새 데이터를 확보했다. 이걸로 다시 학습을 시켜본다.



### 2.6.1.Fail

Validation IOU가 20% 언저리에서 30% 언저리까지 올랐을 뿐 여전히 형편없었다. 뭔가 다른 원인이 있는 것 같다. 그래서 그냥 VGG16에서 가중치를 가져와서 학습시켜보고 원인이 뭔지 다시 찾아보기로 한다.

### 2.6.2.Fail

또 실패했다. 여전히 30% 언저리에서 오르지 않는다. 디버깅 결과 1x1 Convolution layer의 가중치들이 전부 0에서 학습이 전혀 되지 않고 있었고 그 결과 마지막 Pooling layer(pool5)에서 나온 가중치는 전혀 사용하지 않고 그냥 pool4로부터 스킵연결해서 가져온 텐서로만 Upsample, Prediction하고 있었다.(...) 이런데도 IOU가 30%나 나오는 게 대단하다고 해야할지... 거기다 MS-COCO엔 사람 Class가 특히나 많은데 사람에 한정해서는 Validation IOU가 거의 50%대에 육박했다. 내 사진을 찍어서 입력으로 넣어보거나 인터넷에 떠돌아다니는 사람사진을 넣어봐도 의외로 결과가 괜찮다. 대단하지만 실패는 실패다.

### 2.6.3.Two Hypothesizing
원인을 생각해보니 두 가지 부분에서 논문의 구현을 따르지 않아 문제가 발생한 것 같다. 



첫 번째 의심가는 부분은, 논문은 FCN-32s부터 학습시키고 이후 뒤에 레이어를 붙여 FCN-16s, FCN-8s순으로 학습시켰는데 나는 그냥 바로 FCN-8s를 학습시키려고 했다.



두 번째 의심가는 부분은, 내 코드를 찬찬히 들여다 보니 나는 다른 1x1 Convolution Layer 가중치는 0으로 초기화했는데 스킵연결해온 pool4 뒤에 붙는 1x1 Convolution Layer의 가중치는 0으로 초기화하지 않았다. 그래서 0으로 초기화하지 않은 스킵연결 쪽 1x1 Convolution Layer만 가중치 갱신이 된다.  



두 가지 실험을 해보기로 한다. 하나는 FCN-32s를 일단 학습시키고 16s, 8s를 차례대로 전이학습시키며 잘 되는지 보는 실험. 또 하나는 현재 구조(FCN-8s)를 유지하되 모든 1x1 Convolution layer의 가중치를 0으로 초기화해보는 방법이다. 두 실험을 모두 수행해봐야 무엇이 구현에 있어 큰 문제가 되는지 알 수 있을 것이다. 일단 이 문제를 해결하고 다시 Training from scratch로 돌아가기로 한다. 

### 2.6.4.Fail
모든 1x1 Convolution layer의 가중치를 0으로 초기화해보는 방법을 실험해 본 결과 모든 1x1 Convolution layer의 가중치가 0에서 전혀 갱신되지 않는다!
아무래도 가중치를 0으로 초기화하면 안 되는 것 같다. 아무래도 내가 논문을 오독한 것 같다. Scoring layer를 1x1 Convolution Layer로 이해하고 있었는데 그게 아닌 것 같다. 일단 이 부분을 다시 확인해봐야 할 것 같다. 그리고 일단 모든 1x1 Convolution Layer를 Random initialize한 결과를 봐야겠다. (아마 잘 될듯하다. 물론 이게 목적은 아니지만 일단 잘 된 결과가 어떤지 보고 )
