# -*- coding: utf-8 -*-
# @Author   : jeffcheng
# @Time     : 2021/9/1 - 15:13
# @Reference: 단일 오디오 추론 스크립트, demo.py와 traintest.py에 기반함
import os
import sys
import csv
import argparse
import datetime  # 현재 시간 확인을 위한 라이브러리 추가
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt  # 스펙트로그램 이미지 저장을 위해 추가

# 오디오 처리 백엔드 설정
torchaudio.set_audio_backend("soundfile")  # torchaudio에서 사용하는 백엔드를 soundfile로 설정
# 기본 경로 설정
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
from src.models import ASTModel  # AST 모델을 src 디렉토리에서 가져오기

# 사전 학습된 모델이 저장된 폴더 경로 설정
os.environ['TORCH_HOME'] = '/home/lsh/share/ast/pretrained_models'

# 오디오 파일로부터 스펙트로그램 특성(feature) 추출 함수 정의
def make_features(wav_name, mel_bins, target_length=1024):
    # 오디오 파일 불러오기
    waveform, sr = torchaudio.load(wav_name)

    # 칼디(kaldi) 라이브러리를 이용한 필터 뱅크(fbank) 스펙트로그램 생성
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0,
        frame_shift=10)

    # 현재 프레임 수 계산
    n_frames = fbank.shape[0]

    # target_length에 맞춰 패딩 또는 자르기
    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))  # 아래쪽으로 p 프레임 패딩
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]  # target_length만큼 자르기

    # 정규화 수행
    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank

# 스펙트로그램을 이미지로 저장하는 함수 정의
def save_spectrogram_image(fbank, save_dir):
    # 디렉터리가 없으면 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 현재 시간 기준으로 파일 이름 생성
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(save_dir, f"spectrogram_{timestamp}.png")

    # 스펙트로그램 이미지 저장
    plt.imshow(fbank.T, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.savefig(image_path)
    plt.close()
    print(f"[*INFO] 스펙트로그램 이미지가 저장되었습니다: {image_path}")

# 레이블 CSV 파일로부터 레이블 정보를 불러오는 함수 정의
def load_label(label_csv):
    with open(label_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        lines = list(reader)
    labels = []
    ids = []  # 각 레이블에 고유 ID 할당
    for i1 in range(1, len(lines)):
        id = lines[i1][1]
        label = lines[i1][2]
        ids.append(id)
        labels.append(label)
    return labels

# 메인 함수 시작
if __name__ == '__main__':

    # 입력 인자 설정
    parser = argparse.ArgumentParser(description='사용 예시:'
                                                 'python inference --audio_path /home/lsh/share/ast/egs/audioset/data/LDoXsip0BEQ_000177.flac '
                                                 '--model_path /home/lsh/share/ast/pretrained_models/audioset_10_10_0.4593.pth')

    parser.add_argument("--model_path", type=str, required=True,
                        help="테스트할 학습된 모델 경로")
    parser.add_argument('--audio_path',
                        help='예측할 오디오 경로 (샘플링 레이트: 16k)',
                        type=str, required=True)

    args = parser.parse_args()

    # 레이블 CSV 파일 경로 설정
    label_csv = '/home/lsh/share/ast/egs/audioset/data/class_labels_indices.csv'

    # 1. 예측을 위한 특성(feature) 생성
    audio_path = args.audio_path
    feats = make_features(audio_path, mel_bins=128)  # 출력 형식: (1024, 128)

    # 스펙트로그램 이미지 저장 경로 설정
    save_dir = '/home/lsh/share/ast/egs/audioset/result_image'
    save_spectrogram_image(feats, save_dir)  # 스펙트로그램 이미지 저장 함수 호출

    # 각 입력 스펙트로그램이 가지는 프레임 수 설정
    input_tdim = feats.shape[0]

    # 2. 학습된 모델과 가중치 불러오기
    checkpoint_path = args.model_path
    ast_mdl = ASTModel(label_dim=527, input_tdim=input_tdim, imagenet_pretrain=False, audioset_pretrain=False)
    print(f'[*INFO] 체크포인트 불러오기: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cuda')  # 체크포인트 불러오기
    audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])  # 모델 병렬처리
    audio_model.load_state_dict(checkpoint)  # 모델 가중치 로드

    # 모델을 CUDA 장치로 이동
    audio_model = audio_model.to(torch.device("cuda:0"))

    # 3. 특성 데이터를 모델에 입력
    feats_data = feats.expand(1, input_tdim, 128)  # 배치 차원 추가하여 reshape

    audio_model.eval()  # 평가 모드 설정
    with torch.no_grad():  # 추론 시에는 그래디언트 계산 비활성화
        output = audio_model.forward(feats_data)  # 모델을 통해 예측 수행
        output = torch.sigmoid(output)  # 시그모이드 활성화 함수 적용
    result_output = output.data.cpu().numpy()[0]  # 결과를 CPU로 이동 후 numpy 배열로 변환

    # 4. 예측 확률을 레이블과 매핑
    labels = load_label(label_csv)

    # 예측 결과를 내림차순으로 정렬하여 상위 인덱스 추출
    sorted_indexes = np.argsort(result_output)[::-1]

    # 예측 결과 상위 확률 출력
    print('[*INFO] 예측 결과:')
    for k in range(10):  # 상위 10개 결과 출력
        print('{}: {:.4f}'.format(np.array(labels)[sorted_indexes[k]],
                                  result_output[sorted_indexes[k]]))
