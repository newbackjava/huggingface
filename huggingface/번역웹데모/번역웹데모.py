# filename: translate_web_demo_stable.py
import gradio as gr
import re, torch
from typing import List, Tuple, Optional, Union

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

# -------------------------
# 모델 후보 (우선순위)
# -------------------------
NLLB = "facebook/nllb-200-distilled-600M"   # ko<->en 안정, 언어코드 필요
M2M  = "facebook/m2m100_418M"               # 경량/안정
MAR_EN2KO = "Helsinki-NLP/opus-mt-en-ko"    # 환경에 따라 404 날 수 있음 → 최후 폴백
MAR_KO2EN = "Helsinki-NLP/opus-mt-ko-en"

DEVICE = 0 if torch.cuda.is_available() else -1
MAX_SEG_TOKENS = 380
OVERLAP_TOKENS  = 40
MAX_NEW_TOKENS  = 320
BATCH_SIZE      = 12

# 파이프/모델 캐시
CACHE = {
    "NLLB": {"tok": None, "pipe": None},     # translation pipeline
    "M2M":  {"tok": None, "model": None},    # generate 직접
    "MAR":  {"en2ko": None, "ko2en": None},  # translation pipeline 2개
}

# -------------------------
# 유틸: 문장/토큰 분할
# -------------------------
def split_lines(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]

def is_numbered_line(s: str) -> bool:
    return bool(re.match(r"^\s*\d+(\.|)\s", s))

def split_sentences_en(line: str) -> List[str]:
    """
    룩비하인드 없이 영문 문장 분할:
    - 패턴을 캡처 그룹으로 분할한 뒤(구두점 포함) 재조립
    - 예: "He said, \"Go now!\" She left." -> ["He said, \"Go now!\"", "She left."]
    """
    if is_numbered_line(line):
        return [line]
    # 구두점(.,!,?) 뒤에 따옴표/괄호가 있을 수 있음 → 캡처해서 함께 보존
    parts = re.split(r"([.!?]['\")\]]*)\s+", line)
    # parts = [chunk0, delim0, chunk1, delim1, ...] 형태가 됨
    sents = []
    for i in range(0, len(parts), 2):
        chunk = parts[i]
        delim = parts[i+1] if i + 1 < len(parts) else ""
        if chunk is None:
            continue
        sent = (chunk + delim).strip()
        if sent:
            sents.append(sent)
    return sents if sents else [line]

def split_sentences_ko(line: str) -> List[str]:
    if is_numbered_line(line):
        return [line]
    # 한국어는 고정폭 룩비하인드(OK) 또는 캡처-재조합 중 택1
    # 여기서는 호환성을 위해 '캡처-재조합' 방식 사용
    parts = re.split(r"([\.!?])\s+", line)
    sents = []
    for i in range(0, len(parts), 2):
        chunk = parts[i]
        delim = parts[i+1] if i + 1 < len(parts) else ""
        if chunk is None:
            continue
        sent = (chunk + delim).strip()
        if sent:
            sents.append(sent)
    return sents if sents else [line]

def slice_ids_with_overlap(ids: List[int], max_tokens: int, overlap: int) -> List[List[int]]:
    if len(ids) <= max_tokens:
        return [ids[:]]
    chunks, start = [], 0
    step = max_tokens - overlap if max_tokens > overlap else max_tokens
    while start < len(ids):
        end = min(start + max_tokens, len(ids))
        chunks.append(ids[start:end])
        if end == len(ids): break
        start = max(0, end - overlap)
    return chunks

def segment_text(tokenizer, text: str, src_lang: str) -> Tuple[List[str], List[int]]:
    """문장→토큰 슬라이싱으로 세그먼트화, line break 인덱스 반환"""
    lines = split_lines(text)
    segments, marks = [], []
    splitter = split_sentences_en if src_lang == "en" else split_sentences_ko
    for line in lines:
        before = len(segments)
        for sent in splitter(line):
            ids = tokenizer(sent, add_special_tokens=False)["input_ids"]
            for chunk in slice_ids_with_overlap(ids, MAX_SEG_TOKENS, OVERLAP_TOKENS):
                segments.append(tokenizer.decode(chunk, skip_special_tokens=True).strip())
        after = len(segments) - 1
        if after >= before: marks.append(after)
    return segments, marks

def rebuild_text(segs: List[str], marks: List[int]) -> str:
    out, buf = [], []
    for i, s in enumerate(segs):
        clean = re.sub(r"\s+", " ", s).strip()
        if clean: buf.append(clean)
        if i in marks:
            out.append(" ".join(buf).strip()); buf = []
    if buf: out.append(" ".join(buf).strip())
    return "\n".join(out).strip()

def batched(xs: List[str], n: int) -> List[List[str]]:
    return [xs[i:i+n] for i in range(0, len(xs), n)]

# -------------------------
# 백엔드 로더
# -------------------------
def load_nllb():
    if CACHE["NLLB"]["pipe"] is not None:
        return CACHE["NLLB"]["pipe"], CACHE["NLLB"]["tok"]
    tok = AutoTokenizer.from_pretrained(NLLB)
    pipe_nllb = pipeline(
        "translation",
        model=NLLB, tokenizer=tok, device=DEVICE,
        # src_lang/tgt_lang은 호출 시 지정
    )
    CACHE["NLLB"]["tok"]  = tok
    CACHE["NLLB"]["pipe"] = pipe_nllb
    return pipe_nllb, tok

def load_m2m():
    if CACHE["M2M"]["model"] is not None:
        return CACHE["M2M"]["model"], CACHE["M2M"]["tok"]
    tok = AutoTokenizer.from_pretrained(M2M)
    model = AutoModelForSeq2SeqLM.from_pretrained(M2M).to("cuda" if DEVICE == 0 else "cpu")
    CACHE["M2M"]["tok"] = tok
    CACHE["M2M"]["model"] = model
    return model, tok

def load_marian(direction: str):
    if direction == "en2ko":
        if CACHE["MAR"]["en2ko"] is None:
            CACHE["MAR"]["en2ko"] = pipeline("translation", model=MAR_EN2KO, device=DEVICE)
        return CACHE["MAR"]["en2ko"]
    else:
        if CACHE["MAR"]["ko2en"] is None:
            CACHE["MAR"]["ko2en"] = pipeline("translation", model=MAR_KO2EN, device=DEVICE)
        return CACHE["MAR"]["ko2en"]

# -------------------------
# 번역 함수 (백엔드 자동 폴백)
# -------------------------
def translate_backend(text: str, direction: str) -> str:
    """
    direction: "영어 → 한국어" or "한국어 → 영어"
    1) NLLB 시도 → 2) M2M → 3) Marian
    """
    src_lang = "en" if direction.startswith("영어") else "ko"
    tgt_lang = "ko" if src_lang == "en" else "en"

    # 1) NLLB
    try:
        pipe_nllb, tok_nllb = load_nllb()
        segments, marks = segment_text(tok_nllb, text, src_lang)
        out_all = []
        for batch in batched(segments, BATCH_SIZE):
            kwargs = dict(truncation=False, max_new_tokens=MAX_NEW_TOKENS)
            if src_lang == "en":
                res = pipe_nllb(batch, src_lang="eng_Latn", tgt_lang="kor_Hang", **kwargs)
            else:
                res = pipe_nllb(batch, src_lang="kor_Hang", tgt_lang="eng_Latn", **kwargs)
            out_all.extend([r["translation_text"] for r in res])
        return rebuild_text(out_all, marks)
    except Exception as e_nllb:
        nllb_err = str(e_nllb)

    # 2) M2M
    try:
        model, tok = load_m2m()
        tok.src_lang = "en" if src_lang == "en" else "ko"
        segments, marks = segment_text(tok, text, src_lang)
        outs = []
        for batch in batched(segments, BATCH_SIZE):
            enc = tok(batch, return_tensors="pt", padding=True, truncation=False).to(model.device)
            bos = tok.get_lang_id("ko" if tgt_lang == "ko" else "en")
            gen = model.generate(
                **enc,
                forced_bos_token_id=bos,
                num_beams=5,
                max_new_tokens=MAX_NEW_TOKENS
            )
            outs.extend(tok.batch_decode(gen, skip_special_tokens=True))
        return rebuild_text(outs, marks)
    except Exception as e_m2m:
        m2m_err = str(e_m2m)

    # 3) Marian (최후 폴백)
    try:
        if src_lang == "en":
            pipe_mar = load_marian("en2ko")
            tok_mar  = AutoTokenizer.from_pretrained(MAR_EN2KO)
        else:
            pipe_mar = load_marian("ko2en")
            tok_mar  = AutoTokenizer.from_pretrained(MAR_KO2EN)
        segments, marks = segment_text(tok_mar, text, src_lang)
        outs = []
        for batch in batched(segments, BATCH_SIZE):
            res = pipe_mar(batch, truncation=False, max_new_tokens=MAX_NEW_TOKENS)
            outs.extend([r["translation_text"] for r in res])
        return rebuild_text(outs, marks)
    except Exception as e_mar:
        err = f"NLLB error: {nllb_err}\nM2M error: {m2m_err}\nMarian error: {str(e_mar)}"
        return f"[모든 백엔드 로딩 실패]\n{err}\n\n➡ 네트워크(방화벽/프록시) 또는 모델 접근 권한/미러 설정을 확인하세요."

# -------------------------
# Gradio UI
# -------------------------
def translate(text, direction):
    text = (text or "").strip()
    if not text:
        return "텍스트를 입력하세요."
    return translate_backend(text, direction)

with gr.Blocks(title="한↔영 번역 데모 (안정판)") as demo:
    gr.Markdown("## 한↔영 번역 데모 (NLLB 기본, M2M/Marian 폴백)\n긴 입력도 안전하게 처리합니다.")
    with gr.Row():
        inp = gr.Textbox(lines=6, label="텍스트 입력")
        opts = gr.Radio(["영어 → 한국어", "한국어 → 영어"], value="영어 → 한국어", label="방향")
    out = gr.Textbox(lines=6, label="번역 결과")
    btn = gr.Button("번역")
    btn.click(fn=translate, inputs=[inp, opts], outputs=out)

demo.queue().launch()
