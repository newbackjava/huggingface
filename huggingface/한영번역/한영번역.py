# pip install sacremoses
# pip install sentencepiece

from transformers import pipeline

ko2en = pipeline("translation", model="Helsinki-NLP/opus-mt-ko-en")
en2ko = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-ko")

print(ko2en("허깅페이스는 최신 AI 모델을 쉽게 사용할 수 있게 해줍니다."))
print(en2ko("Hugging Face makes state-of-the-art AI easy to use."))

#
# C:\Users\Administrator\PycharmProjects\PythonProject\.venv\Scripts\python.exe C:\Users\Administrator\PycharmProjects\PythonProject\huggingface\한영번역\한영번역.py
# [{'translation_text': 'Huggingspace makes the latest AI model easy to use.'}]
# [{'translation_text': 'women09) Now Cheerleader는 onemg row-heart plague를 요청 한 곳을 조사했습니다.'}]
