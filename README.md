# Machine Learning Project

## Requirements
python >= 3.7 or 3.8 my version is 3.8.5 우분투 환경 20.04

```bash
pip install -r requirement.txt
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
https://drive.google.com/file/d/1AtyvY0qFM5FWPoZnml4jyNnVprb9PVjE/view?usp=sharing

위 링크에서 파일 다운로드해서 3DDFA폴더에 압축풀기

```bash
cd 3DDFA
bash build.sh
cd ../
```
https://drive.google.com/file/d/171V6_Hbdipu9K76cavytuGABBVP8R7br/view?usp=sharing

위 링크에서 shape_predictor_68_face_landmarks.dat 다운로드 하여 data/dlib 위치에 넣어줌

## Demo
```bash
python demo.py --config configs/demo_mpiigaze_lenet_land.yaml
```
