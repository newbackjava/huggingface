from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
import numpy as np
from evaluate import load as load_metric

# 1) 데이터 준비
# ★ 커스텀 스크립트 허용
ds = load_dataset("nsmc", trust_remote_code=True)
print(ds, ds["train"][0])

# 2) 토크나이저/모델 불러오기 (KLUE/BERT 권장)
model_id = "klue/bert-base"
tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

def tokenize(batch):
    return tok(batch["document"], truncation=True, max_length=256)

# document 텍스트는 제거 (메모리 절약)
ds_tok = ds.map(tokenize, batched=True, remove_columns=["document"])

# 3) 분류 모델 로드 (이진 레이블)
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

# 4) 평가지표(정확도)
acc = load_metric("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return acc.compute(predictions=preds, references=labels)

# (데모용) 작은 서브셋으로 빠르게 돌려보기 — 실제는 전체 데이터 사용 권장
train_ds = ds_tok["train"].shuffle(seed=42).select(range(10000))
eval_ds  = ds_tok["test"].shuffle(seed=42).select(range(5000))

# 5) 학습 세팅
args = TrainingArguments(
    output_dir="./ko-sent-out",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=2,                 # 가볍게 1~2epoch로 시작
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# 6) 학습
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tok,
    compute_metrics=compute_metrics,
)
trainer.train()

# 7) 추론 파이프라인으로 사용
from transformers import pipeline
clf_ko = pipeline("text-classification", model=trainer.model, tokenizer=tok, device_map="auto")

tests = [
    "스토리가 탄탄하고 배우들 연기가 훌륭합니다. 강력 추천합니다.",
    "시간 낭비였어요. 전개가 지루하고 완성도가 떨어집니다.",
    "연출은 좋았는데 결말이 아쉬웠습니다."
]
for t in tests:
    y = clf_ko(t)[0]  # {'label': 'LABEL_0 or LABEL_1', 'score': ...}
    label = "부정" if y["label"].endswith("0") else "긍정"
    print(t, "=>", label, f"(score={y['score']:.3f})")
