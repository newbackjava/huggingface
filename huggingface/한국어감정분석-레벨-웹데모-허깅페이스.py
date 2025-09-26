import gradio as gr
from transformers import pipeline

# 빠른 시작: 다국어 모델 (혹은 위에서 학습한 trainer.model/토크나이저로 교체 가능)
clf = pipeline("sentiment-analysis",
               model="nlptown/bert-base-multilingual-uncased-sentiment")

def predict(text: str):
    if not text.strip():
        return "문장을 입력해주세요."
    out = clf(text)[0]  # {'label': 'LABEL_1~5', 'score': ...}
    n = int(out["label"].split("_")[-1])
    if n <= 2: senti = "부정"
    elif n == 3: senti = "중립"
    else: senti = "긍정"
    return f"{senti} (별점:{n}/5, score={out['score']:.4f})"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=4, placeholder="한국어 문장을 입력하세요."),
    outputs="text",
    title="한국어 감정분석 데모",
    description="간단 한국어 감정분석: 다국어 모델 기반(별점→긍/중립/부정 매핑)."
)
demo.launch()
