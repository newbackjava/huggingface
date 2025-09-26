from transformers import pipeline

# 모델: ko→en
ko2en = pipeline("translation", model="Helsinki-NLP/opus-mt-ko-en")  # 또는 tc-big-ko-en

# 파일 경로
input_file = "input.txt"    # 원본
output_file = "output_en.txt"  # 번역 결과 저장

# 읽기
with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

# 번역 (가장 간단)
translated = ko2en(text)[0]["translation_text"]

# 터미널 출력
print("===== 🙂 번역 결과 (KO→EN) =====")
print(translated)

# 저장
with open(output_file, "w", encoding="utf-8") as f:
    f.write(translated)
