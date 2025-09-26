from transformers import AutoTokenizer, pipeline
from transformers.utils import logging as hf_logging
import torch, re
from typing import List, Tuple, Optional, Union

# ===== 설정 =====
MODEL_NAME = "facebook/nllb-200-distilled-600M"
INPUT_PATH  = "input_en.txt"   # 영어 원문
OUTPUT_PATH = "output_ko.txt"  # 한국어 번역 결과

MAX_INPUT_TOKENS_SEGMENT = 380
OVERLAP_TOKENS_SEGMENT   = 40
BATCH_SIZE = 16
MAX_NEW_TOKENS = 320

PRINT_MODE: Union[Optional[int], str] = "full"  # "full" 또는 정수(N줄 미리보기)
hf_logging.set_verbosity_error()
# ==============

def split_text_to_lines(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]

def is_numbered_line(s: str) -> bool:
    return bool(re.match(r"^\s*\d+(\.|)\s", s))

def split_line_to_sentences_en(line: str) -> List[str]:
    if is_numbered_line(line):
        return [line]
    parts = re.split(r"(?<=[\.!?]['\"\)\]]*)\s+", line)
    parts = [p.strip() for p in parts if p and p.strip()]
    return parts if parts else [line]

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

def ensure_sentence_segments(tok, sentence: str, max_tokens: int, overlap: int) -> List[str]:
    ids = tok(sentence, add_special_tokens=False)["input_ids"]
    id_chunks = slice_ids_with_overlap(ids, max_tokens, overlap)
    return [tok.decode(c, skip_special_tokens=True).strip() for c in id_chunks]

def load_and_segment_text(tok, text: str, max_seg: int, overlap: int) -> Tuple[List[str], List[int]]:
    lines = split_text_to_lines(text)
    segments, line_break_marks = [], []
    for line in lines:
        before = len(segments)
        for sent in split_line_to_sentences_en(line):
            segments.extend(ensure_sentence_segments(tok, sent, max_seg, overlap))
        after = len(segments) - 1
        if after >= before:
            line_break_marks.append(after)
    return segments, line_break_marks

def batched(xs: List[str], n: int) -> List[List[str]]:
    return [xs[i:i+n] for i in range(0, len(xs), n)]

def translate_segments(pipe, segments: List[str]) -> List[str]:
    out = []
    for batch in batched(segments, BATCH_SIZE):
        # NLLB는 src_lang/tgt_lang 필요
        res = pipe(batch, truncation=False, max_new_tokens=MAX_NEW_TOKENS)
        out.extend([r["translation_text"] for r in res])
    return out

def rebuild_text_from_segments(segs: List[str], marks: List[int]) -> str:
    lines_out, buf = [], []
    for i, seg in enumerate(segs):
        clean = re.sub(r"\s+", " ", seg).strip()
        if clean: buf.append(clean)
        if i in marks:
            lines_out.append(" ".join(buf).strip()); buf=[]
    if buf: lines_out.append(" ".join(buf).strip())
    return "\n".join(lines_out).strip()

def print_result(text: str, mode: Union[Optional[int], str] = "full"):
    print("\n===== EN → KO (NLLB-200) =====")
    if mode == "full":
        print(text); print("================================\n"); return
    if isinstance(mode, int):
        lines = text.splitlines(); n = max(1, mode)
        print("\n".join(lines[:n]))
        if len(lines) > n: print(f"\n[... first {n} lines shown ...]")
        print("================================\n"); return
    print(text); print("================================\n")

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = 0 if torch.cuda.is_available() else -1
    # pipeline에 언어 코드 명시
    pipe = pipeline(
        "translation",
        model=MODEL_NAME,
        tokenizer=tok,
        src_lang="eng_Latn",
        tgt_lang="kor_Hang",
        device=device,
    )

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        raw = f.read()

    segments, marks = load_and_segment_text(tok, raw, MAX_INPUT_TOKENS_SEGMENT, OVERLAP_TOKENS_SEGMENT)
    translated_segments = translate_segments(pipe, segments)
    final_text = rebuild_text_from_segments(translated_segments, marks)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(final_text)

    print_result(final_text, PRINT_MODE)
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Segments translated: {len(segments)}")

if __name__ == "__main__":
    main()
