from transformers import pipeline

zero = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "이번 분기 매출 보고서를 금요일 이사회 전에 마무리해야 합니다."
labels = ["인사", "회계/재무", "영업/마케팅", "법무", "개발"]
res = zero(
    text,
    candidate_labels=labels,
    hypothesis_template="이 문장은 {}와 관련이 있다."
)
print(res)  # labels, scores 정렬된 결과
