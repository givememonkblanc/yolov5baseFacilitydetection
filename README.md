# background requirement 
- 해당 코드 실행 환경은 RAM 16.0GB, Intel Core i7-11700, NVIDIA GeForce RTX 2060, windows10 환경에서 진행함.
- 따라서 CUDA 10.1버전에서 진행되었지만, 심사용 컴퓨터 실행 환경에 맞춰 CUDA 11.0버전에 맞춰 아래 코드를 세팅함. 

#그래픽 드라이버는 450.36.06버전 이상이여야함. 

### STEP 01 CUDA 11.0, cuDNN 8.0.5 install
- 터미널 실행

CUDA 11.0
```bash
#cuda 설치
$ wget https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda_11.0.3_450.51.06_linux.run
$ sudo sh cuda_11.0.3_450.51.06_linux.run
```
```bash
#cuda 환경변수 설정
$ vim ~/.bashrc
$ export PATH=/usr/local/cuda-11.0/bin:$PATH
$ export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH
```

cuDNN 8.0 5   

https://developer.nvidia.com/rdp/cudnn-download 에서 8.0.5 설치

-> Download cuDNN v8.0.5 (November 9th, 2020), for CUDA 11.0 클릭 후 설치


```bash
#cuDNN 설치 후 필요한 파일 CUDA 폴더 경로로 옮김
$ tar -xzvf cudnn-11.0-linux-x64-v8.0.2.39.tgz
$ sudo cp cuda/include/cudnn*.h /usr/local/cuda-11.0/include
$ sudo cp cuda/lib64/libcudnn* /usr/local/cuda-11.0/lib64
$ sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda-11.0/lib64/libcudnn*
```


### STEP 02 anaconda install
#아나콘다 프롬프트 실행을 위해 아나콘다 설치
 #실행 환경에 아나콘다 프롬프트가 있을 경우 생략
https://www.anaconda.com/products/individual


### STEP 03 anaconda 가상환경 생성
```bash
#gpu3090의 이름으로 python3.8 버전의 가상환경 생성
$ conda create -n gpu3090 python=3.8 
#gpu3090의 가상환경으로 이동
$ conda activate gpu3090
```


### STEP 04 가상환경에 Torch and python requirement install
```bash
#CUDA버전과 맞는 PyTorch 설치
$ pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
#필요한 기타 패키지 설치
$ pip install -r requirements.txt
```


# yolov5 모델을 사용해 학습
```bash
#학습코드
$ python train.py --img 608 --batch 8 --epochs 50 --data data/coco128.yaml --cfg models/yolov5m.yaml --weights weights/yolov5m.pt

```


# 모델 inference
```bash
#inference 결과값 출력을 위한 코드(conf가 0.3 이상인 것만 출력)
$ python detect.py --source ./test_input/ --weights runs/exp()/weights/best.pt --conf 0.3 --save-txt --nosave
```
```bash
#결과값을 test_output에 저장하기 위한 코드
$ python .py
```
```bash
#평가점수 산출을 위한 코드
$ python test.py --weights runs/exp()/weights/best.pt --data data/test.yaml --img 604 --batch 8
```

# 성능평가 위해 사용한 테스트셋 inference
- 평가시 inference 과정이 원활하지 않을 경우를 대비한 참여팀의 성능평가에 사용한 테스트 실행
```bash
#평가점수 산출을 위한 코드
$ python test.py --weights runs/exp()/weights/best.pt --data data/test.yaml --img 604 --batch 8

#위 코드를 실행한 결과가 이미 ourtest_summary폴더에 oursummary.txt로 저장되어있음
```
