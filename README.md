# 텐서 곱을 활용한 신경망 경량화  
아래의 논문 및 특허출원에 기반한 Python 기반 구현물 입니다. 
- 김태현, 최윤식, "심층 합성곱 신경망 압축을 위한 텐서 곱셈 기반 필터 선택", 2021년도 대한전자공학회 하계학술대회 논문집
- "Method and Apparatus for Compression of Convolutional Neural Network Using N-Mode Tensor Product Operation", KR 10-2021-011675, 출원일: 2021 

# Usuage
- Compress a trained model: ``python main.py --load "target_CNN_dir" --save_name "pruned_CNN_dir" --comp_ratio "pruning rate [0.0, 0.99]"``
- PyTorch: >= 1.0.0 version; https://pytorch.org/
- Tensorly: >= 0.4.5 version; for tensor production; ``pip install -U tensorly``
- EfficientDet: https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch

# Acknowledgement
본 구현물은 2021년 정부(과학기술정보통신부)의 재원으로 정보통신기획평가원의 지원을 받아 수행된 연구임(No.2021-0-00022, Edge 컴퓨팅 환경을 위한 AI 모델 최저화 및 경량화 기술 개발)
