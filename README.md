# Samsung collegiate programming challenges

**VQA Project – BEiT‑3 기반 멀티모달 질문답변 모델**  
**데모용 베이스라인 코드 – 시각–언어 모델 finetuning**

---

본 저장소는 BEiT‑3 모델을 활용한 시각 질문답변(Visual Question Answering) 문제를 풀기 위한 프로젝트입니다.  
원본 Jupyter 노트북에서 개발된 코드를 모듈화하여 `src` 패키지에 정리하였으며, 학습과 추론을 별도의 스크립트로 분리했습니다.  
**YAML 설정 파일 기반**으로 데이터 경로와 하이퍼파라미터를 관리할 수 있어 실험 반복이 편리합니다.

---

## 🎯 Project Goals

* **멀티모달 입력 처리** – 이미지와 질문(텍스트)를 함께 처리하는 VQA 모델을 구현합니다.
* **BEiT‑3 모델 미세조정** – 사전학습된 비전–언어 모델을 특정 VQA 데이터셋에 맞게 fine‑tune 합니다.
* **모듈화된 코드 구조** – `src/` 패키지에 데이터셋, 모델 래퍼, 손실 함수, 학습 루프 등을 모듈화하여 유지 보수성을 높였습니다.
* **YAML 기반 실험 관리** – `configs/train.yaml`과 `configs/submit.yaml`을 통해 학습 및 추론 설정을 중앙집중적으로 관리합니다.

---

## 📁 Project Structure

```
vqa_project/
├── src/
│   ├── __init__.py          # 패키지 초기화 및 주요 클래스/함수 노출
│   ├── dataset.py           # VQA 데이터셋 로딩 및 전처리
│   ├── losses.py            # 손실 함수 (예: cross entropy)
│   ├── model.py             # BEiT‑3 모델 래퍼 (HuggingFace 기반)
│   ├── trainer.py           # 학습 및 추론 루틴
│   └── utils.py             # 공용 유틸리티 (시드 고정 등)
│
├── train.py                 # 학습 실행 스크립트
├── inference.py             # 추론 및 제출 파일 생성 스크립트
│
├── configs/
│   ├── train.yaml           # 학습용 설정
│   └── submit.yaml          # 추론/제출용 설정
│
├── assets/
│   ├── model.pt             # 미세조정된 모델 가중치 저장 위치 (placeholder)
│   └── tokenizer/           # 문장 토크나이저 파일 폴더 (placeholder)
│
├── data/
│   ├── train.csv            # 학습 데이터 (placeholder)
│   └── test.csv             # 테스트 데이터 (placeholder)
│
├── requirements.txt         # 필요 패키지 목록
├── .gitignore               # Git 무시 파일 패턴
└── .gitattributes           # Git 속성 (예: LFS 설정)
```

---

## 🛠 Environment Setup

Python 3.9 이상에서의 실행을 권장합니다. 다음 명령어로 필요한 패키지를 설치하세요:

```bash
pip install -r requirements.txt
```

`requirements.txt`에는 PyTorch, HuggingFace Transformers, pandas 등 주요 라이브러리가 기재되어 있습니다. GPU를 사용할 경우 CUDA 호환 버전의 PyTorch를 설치해야 합니다.

---

## 🚀 Usage

### Training

BEiT‑3 모델을 미세조정하려면 다음과 같이 실행합니다:

```bash
python train.py --config configs/train.yaml
```

`train.py`는 YAML 설정을 읽어 랜덤 시드를 고정하고, `src/dataset.py`를 사용해 CSV 데이터를 로딩 및 JSON 포맷으로 변환합니다. 그 후 `src/trainer.py`의 학습 루틴을 호출하여 모델을 fine‑tune 합니다. 기본적으로 단순한 PyTorch 학습 루프를 사용하지만, 필요하다면 HuggingFace의 `Trainer` 또는 Microsoft가 제공하는 BEiT‑3 finetuning 스크립트를 `subprocess`로 호출하도록 수정할 수 있습니다.

### Inference

학습된 모델로 테스트 데이터에 대한 예측을 생성하려면 아래 명령을 사용하세요:

```bash
python inference.py --config configs/submit.yaml
```

이 스크립트는 `assets/model.pt`와 `assets/tokenizer/`에 저장된 미세조정된 모델과 토크나이저를 로드합니다. 그런 다음 테스트 CSV를 전처리하여 모델에 입력하고, `src/trainer.py`에 정의된 추론 루틴을 통해 답안을 생성합니다. 다수의 체크포인트를 앙상블하는 기능도 YAML 파일에서 설정할 수 있습니다.

---

## 📜 Notes

* `assets/model.pt`와 `assets/tokenizer/`는 빈 디렉터리로 제공됩니다. 실제 학습 전에 BEiT‑3 사전학습 모델과 문장 토크나이저를 다운로드하여 이 위치에 저장해야 합니다.
* `data/train.csv`와 `data/test.csv`는 예시용 빈 파일입니다. 실제 실험에서는 Visual7W와 같은 원본 데이터셋을 CSV 형식으로 전처리하여 교체해야 합니다.
* `configs/train.yaml`과 `configs/submit.yaml`에는 데이터 경로, 하이퍼파라미터, 출력 경로 등이 정의되어 있습니다. 실험에 따라 적절히 수정하세요.
* `.gitignore`에는 큰 데이터와 체크포인트를 Git에서 제외하도록 설정되어 있습니다. 필요하다면 `.gitattributes`를 수정해 Git LFS로 관리할 파일을 지정할 수 있습니다.

이 프로젝트는 VQA 문제에 대한 시작점으로 제공됩니다. 실제 대회나 연구에 참여할 때는 더 정교한 모델 구조와 학습 전략을 적용해 성능을 향상시킬 수 있습니다.
