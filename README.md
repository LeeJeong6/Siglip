# SigLIP Custom Implementation

이번 프로젝트는 구글의 SIGLIP MODEL을 직접 scratch부터 구현하여 학습 후 Inference하는 과정입니다.

## Model Architecture

- SIGLIP의 아키텍쳐는 huggingface에 올라와있는 모델을 출력해서 본떠 만들었습니다[google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224).
- 가중치를 다운받거나 모델을 불러온 게 아니라 `SiglipTextModel`과`SiglipVisionModel`을 각각 py파일로 만든 뒤 main.py에서 합쳐서 만들어보았습니다.



## Dataset

- Train에는 **MSCOCO** , **Flicker8k**을 사용했습니다 (13만장,8300장)
- Inference에는 **ImageNet-100** 을 사용했습니다(100클래스)

- Dataset_util에 있는 파일이름으로 직접 확인 가능합니다. 
- dataset다운받는 코드와 압축 해제 등이 다 있습니다.


- ImageNet 데이터셋은 huggingface에서 다운받아 로컬환경으로 불러와서 사용했습니다.
- 데이터를 로컬 환경으로 불러오는 파일이 ImageNet_data_download.py입니다

## Training

- Optimizer: Adam
- Batch Size: 32
- Epochs: 10
- Loss: Sigmoid Contrastive loss를 직접 만들었습니다. main.py의 Sigmoid_loss(),Sigmoid_loss_GPT()에 구현되어있습니다

- GPU의 한계로 배치 사이즈를 키울 수 없었습니

### Limitation
- batchsize가 32를 초과할 때, OUT OF MEMORY문제로 진행할 수 없었습니다.1epoch도 끝나기 전에 이미 loss가 수렴해버리는 문제가 발생했고 이를 해결할 수 없었습니다.
- hugging face의 동일한 siglip-base모델과 조금 다른데 text encoder와 vision encoder 둘 다 pooling attetnion층이 있는 것이 차이점입니다
