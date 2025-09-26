import gradio as gr
from transformers import pipeline

en2ko = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ko")
ko2en = pipeline("translation", model="Helsinki-NLP/opus-mt-ko-en")

def translate(text, direction):
    if not text.strip():
        return "텍스트를 입력하세요."
    if direction == "영어 → 한국어":
        return en2ko(text)[0]["translation_text"]
    else:
        return ko2en(text)[0]["translation_text"]

demo = gr.Interface(
    fn=translate,
    inputs=[gr.Textbox(lines=4, label="텍스트 입력"), gr.Radio(["영어 → 한국어", "한국어 → 영어"], value="영어 → 한국어")],
    outputs="text",
    title="한↔영 번역 데모",
    description="OPUS-MT 기반 초간단 번역기"
)
demo.launch()
