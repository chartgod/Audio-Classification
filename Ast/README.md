# 1. 모델 다운로드
bash download_models.sh

# 2. Ensemble 모델 결과 확인
python ensemble.py

# 3. 가중치 파일 생성
python gen_weight_file.py --data_path <데이터 경로>

# 4. 단일 오디오 추론
# 오디오 파일 경로와 모델 파일 경로를 입력하세요
python inference.py --audio_path /home/lsh/share/ast/egs/audioset/data/LDoXsip0BEQ_000177.flac --model_path /home/lsh/share/ast/pretrained_models/audioset_10_10_0.4593.pth

# 5. 훈련 및 테스트 실행
bash run.sh
