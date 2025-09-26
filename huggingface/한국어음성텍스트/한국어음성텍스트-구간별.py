from transformers import pipeline
import os

os.environ["PATH"] += os.pathsep + r"C:\ffmpeg-8.0-essentials_build\bin"

asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")

result = asr("./risk-136788.mp3", return_timestamps=True)

# 전체 문장
print("전체 텍스트:", result["text"])

# 구간별 자막
print("\n[구간별 자막]")
for chunk in result.get("chunks", []):
    print(chunk["timestamp"], chunk["text"])


# 전체 텍스트:  Music I'm not sure if you can see this, but I'm not sure. You
#
# [구간별 자막]
# (0.0, 14.56)  Music
# (0.0, 7.0)  I'm not sure if you can see this, but I'm not sure.
# (0.0, 2.0)  You
