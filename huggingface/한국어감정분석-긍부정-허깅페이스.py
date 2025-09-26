from transformers import pipeline

MODEL_NAME = "daekeun-ml/koelectra-small-v3-nsmc"
clf = pipeline("sentiment-analysis", model=MODEL_NAME, tokenizer=MODEL_NAME)

tests = ["이 영화 정말 최고야!", "서비스가 너무 실망스러웠어."]
for t in tests:
    out = clf(t)[0]                 # 예: {'label': 'positive', 'score': 0.998...} 혹은 LABEL_0/1
    label = out["label"].lower()
    # 라벨 정규화(모델마다 'LABEL_0'/'LABEL_1' 또는 'positive'/'negative' 등 출력이 다를 수 있음)
    if label in ["label_0", "0", "negative", "neg"]:
        label_kor = "부정"
    elif label in ["label_1", "1", "positive", "pos"]:
        label_kor = "긍정"
    else:
        label_kor = label
    print(f"{t} -> {label_kor} ({out['score']:.4f})")
