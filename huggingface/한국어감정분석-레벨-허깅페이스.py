from transformers import pipeline

# 다국어 1~5점 평점형 감정모델(별도 미세조정 없이 한글도 처리 가능)
clf = pipeline("sentiment-analysis",
               model="nlptown/bert-base-multilingual-uncased-sentiment")

texts = [
    "이 제품 정말 대만족이에요. 성능이 훌륭합니다.",
    "배송이 너무 느리고 포장이 엉망이라 화가 납니다.",
    "그냥 무난무난해요. 딱히 좋지도 나쁘지도 않네요."
]
for t in texts:
    print(t, "->", clf(t)[0])
