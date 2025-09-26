# 추가 패키지 설치 필요:
# pip install librosa soundfile

from transformers import pipeline
import os

# ffmpeg 경로 추가
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg-8.0-essentials_build\bin"

# Whisper 모델 로딩
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# 로컬 mp3 파일 처리 (30초 이상이어도 동작)
result = asr("./risk-136788.mp3", return_timestamps=True)

# 전체 텍스트만 출력
print(result["text"])

#  Music I'm not sure if you can see this, but I'm not sure. You
