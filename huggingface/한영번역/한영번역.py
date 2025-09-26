from transformers import pipeline

ko2en = pipeline("translation", model="Helsinki-NLP/opus-mt-ko-en")
en2ko = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ko")

print(ko2en("허깅페이스는 최신 AI 모델을 쉽게 사용할 수 있게 해줍니다."))
print(en2ko("Hugging Face makes state-of-the-art AI easy to use."))
