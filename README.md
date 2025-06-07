# SigLIP Custom Implementation

이번 프로젝트는 구글의 SIGLIP MODEL을 직접 scratch부터 구현하여 학습 후 Inference하는 과정입니다.

## Model Architecture

- SIGLIP의 아키텍쳐는 huggingface에 올라와있는 모델을 출력해서 본떠 만들었습니다[google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224).
- 가중치를 다운받거나 모델을 불러온 게 아니라 `SiglipTextModel`과`SiglipVisionModel`을 각각 py파일로 만든 뒤 main.py에서 합쳐서 만들어보았습니다.



## Dataset

- **ImageNet-100** 을 사용했습니다(100클래스)
- 이 데이터셋도 huggingface에서 다운받아 로컬환경으로 불러와서 사용했습니다.
- 데이터를 로컬 환경으로 불러오는 파일이 ImageNet_data_download.py입니다

## Training

- Optimizer: Adam
- Batch Size: 16
- Epochs: 100
- Loss: Sigmoid Contrastive loss를 직접 만들었습니다. main.py의 Sigmoid_loss()에 구현되어있습니다


### Loss Curve

- 업로드 할 예정

### CLIP과 비교해본 결과

- 업로드 할 예정
  
