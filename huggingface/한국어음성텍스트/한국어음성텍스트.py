# 추가 패키지 설치 필요:
# pip install librosa soundfile

from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")
# 로컬 wav/mp3 파일 경로(예: sample_ko.wav)
print(asr("./risk-136788.mp3")["text"])
