# filename: en2ko_no_drop_print_py39.py
from transformers import pipeline, AutoTokenizer
from transformers.utils import logging as hf_logging
import torch, re
from typing import List, Tuple, Optional, Union

# ===================== 설정 =====================
# 1) 먼저 tc-big 계열을 시도하고, 실패하면 기본 en-ko로 폴백
PREFERRED_MODELS = [
    "Helsinki-NLP/opus-mt-tc-big-en-ko",  # 사용 가능하면 이걸 우선
    "Helsinki-NLP/opus-mt-en-ko",         # 폴백: 가장 널리 쓰이는 기본 en→ko
]

INPUT_PATH  = "input_en.txt"      # 원문(영어)
OUTPUT_PATH = "output_ko.txt"     # 번역 결과(한국어)

# 세그먼트(번역 단위) 최대 토큰 수: 512보다 충분히 낮게(경고 방지)
MAX_INPUT_TOKENS_SEGMENT = 380     # 권장: 360~420
OVERLAP_TOKENS_SEGMENT   = 40      # 인접 세그먼트 간 문맥 연결용(60~80까지 가도 좋음)

# 파이프라인 배치 크기(세그먼트 리스트를 나눠 보내는 크기)
BATCH_SIZE = 16

# 출력 길이 상한(선택): 한국어가 길어지면 400~600으로 확대 가능
MAX_NEW_TOKENS = 300

# 경고 로그 숨김(옵션)
hf_logging.set_verbosity_error()

# 터미널 프린트 모드: "full"이면 전체 출력, 정수면 처음 N줄만 미리보기
# 예: PRINT_MODE = "full"  또는 PRINT_MODE = 50
PRINT_MODE: Union[Optional[int], str] = "full"
# ===============================================


def is_numbered_line(s: str) -> bool:
    # "1. " "2. " 혹은 "1 " 등의 번호 목록 탐지(간단 휴리스틱)
    return bool(re.match(r"^\s*\d+(\.|)\s", s))


def split_text_to_lines(text: str) -> List[str]:
    # 비어있는 줄 제거 + 양끝 공백 제거
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def split_line_to_sentences_en(line: str) -> List[str]:
    """
    영어 줄을 문장들로 분해. 번호 목록은 가급적 한 문장으로 유지.
    그 외에는 종결부호 기준 분리(큰따옴표/작은따옴표/괄호 마무리 포함).
    """
    if is_numbered_line(line):
        return [line]

    # 종결부호 패턴: ., ?, ! 뒤에 따옴표/괄호가 올 수 있음 -> 공백/줄끝에서 분할
    # 예) He said, "Go now!" She left. -> 두 문장 분리
    parts = re.split(r"(?<=[\.!?]['\"\)\]]*)\s+", line)
    parts = [p.strip() for p in parts if p and p.strip()]
    return parts if parts else [line]


def tokenize_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"])


def slice_ids_with_overlap(ids: List[int], max_tokens: int, overlap: int) -> List[List[int]]:
    """
    토큰 ID 시퀀스를 max_tokens 이하로 겹치며 분할.
    어떤 경우에도 '버림' 없음.
    """
    if len(ids) <= max_tokens:
        return [ids[:]]

    chunks = []
    start = 0
    step = max_tokens - overlap if max_tokens > overlap else max_tokens
    while start < len(ids):
        end = min(start + max_tokens, len(ids))
        chunks.append(ids[start:end])
        if end == len(ids):
            break
        start = end - overlap  # 겹침 유지
        if start < 0:
            start = 0
        if start >= len(ids):
            break
    return chunks


def ensure_sentence_segments(tokenizer, sentence: str,
                             max_tokens: int, overlap: int) -> List[str]:
    """
    한 '문장'이 너무 길면 토큰 기준으로 안전하게 여러 세그먼트로 쪼갬.
    어떤 길이의 문장도 '제외되지 않음'을 보장.
    """
    ids = tokenizer(sentence, add_special_tokens=False)["input_ids"]
    id_chunks = slice_ids_with_overlap(ids, max_tokens, overlap)
    # 토큰 조각을 다시 텍스트로 디코딩
    return [tokenizer.decode(c, skip_special_tokens=True).strip() for c in id_chunks]


def load_and_segment_text(tokenizer, text: str,
                          max_tokens_segment: int,
                          overlap_tokens_segment: int) -> Tuple[List[str], List[int]]:
    """
    입력 텍스트를 '번역 세그먼트' 리스트로 변환.
    반환:
      - segments: 번역 단위 텍스트 조각(절대 누락 없음)
      - line_break_marks: 어느 인덱스 뒤에 줄바꿈을 넣을지 표시하는 '세그먼트 인덱스' 목록
                          (i번째 세그먼트 뒤에 줄바꿈)
    """
    lines = split_text_to_lines(text)
    segments: List[str] = []
    line_break_marks: List[int] = []  # 세그먼트 인덱스

    for line in lines:
        before_line_start = len(segments)
        sentences = split_line_to_sentences_en(line)
        for sent in sentences:
            # 한 문장이 너무 길면 토큰 슬라이스로 반드시 쪼갠다(누락 없음)
            segs = ensure_sentence_segments(
                tokenizer, sent, max_tokens_segment, overlap_tokens_segment
            )
            segments.extend(segs)
        # 이 줄이 끝난 위치(마지막 세그먼트 인덱스)에 줄바꿈 표시
        after_line_end = len(segments) - 1
        if after_line_end >= before_line_start:
            line_break_marks.append(after_line_end)

    return segments, line_break_marks


def batched(iterable: List[str], n: int) -> List[List[str]]:
    return [iterable[i:i+n] for i in range(0, len(iterable), n)]


def translate_segments(pipe, segments: List[str]) -> List[str]:
    """
    세그먼트 리스트를 배치로 번역.
    세그먼트와 1:1로 결과가 매핑되도록 리스트 입력 사용.
    """
    results: List[str] = []
    for batch in batched(segments, BATCH_SIZE):
        outs = pipe(
            batch,
            truncation=False,          # 세그먼트 길이는 이미 안전 보장
            max_new_tokens=MAX_NEW_TOKENS
        )
        # pipeline(list) -> list of dict
        for o in outs:
            results.append(o["translation_text"])
    return results


def rebuild_text_from_segments(translated_segments: List[str],
                               line_break_marks: List[int]) -> str:
    """
    번역된 세그먼트들을 원래 줄 경계에 맞춰 복원.
    같은 줄의 세그먼트는 공백으로 연결, 줄 사이에는 개행.
    """
    lines_out: List[str] = []
    buf: List[str] = []
    for i, seg in enumerate(translated_segments):
        clean = re.sub(r"\s+", " ", seg).strip()
        if clean:
            buf.append(clean)
        if i in line_break_marks:
            lines_out.append(" ".join(buf).strip())
            buf = []
    if buf:
        lines_out.append(" ".join(buf).strip())
    return "\n".join(lines_out).strip()


def print_result_to_terminal(final_text: str,
                             mode: Union[Optional[int], str] = "full"):
    """
    최종 결과를 터미널에 출력.
    - mode == "full": 전체 출력
    - mode 가 int: 처음 N줄만 미리보기
    """
    print("\n===== EN → KO Translation Result =====")
    if mode == "full":
        print(final_text)
        print("======================================\n")
        return

    if isinstance(mode, int):
        lines = final_text.splitlines()
        n = max(1, mode)
        preview = "\n".join(lines[:n])
        print(preview)
        if len(lines) > n:
            print("\n[... truncated preview. Showing first {} lines ...]".format(n))
        print("======================================\n")
        return

    # 예외적 입력은 full로 처리
    print(final_text)
    print("======================================\n")


def build_pipe_with_fallback():
    last_err = None
    for name in PREFERRED_MODELS:
        try:
            tokenizer = AutoTokenizer.from_pretrained(name)
            device = 0 if torch.cuda.is_available() else -1
            pipe = pipeline("translation", model=name, tokenizer=tokenizer, device=device)
            return pipe, tokenizer, name
        except Exception as e:
            last_err = e
            continue
    # 전부 실패하면 에러 리레이즈
    raise RuntimeError(f"Failed to load any EN→KO model. Last error: {last_err}")


def main():
    # 모델/토크나이저/파이프라인 (폴백 포함)
    pipe, tokenizer, model_name = build_pipe_with_fallback()
    print(f"Loaded EN→KO model: {model_name}")

    # 입력 읽기
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        raw = f.read()

    # 세그먼트 생성(초장문 문장도 토큰 슬라이스로 절대 누락 X)
    segments, lb_marks = load_and_segment_text(
        tokenizer, raw, MAX_INPUT_TOKENS_SEGMENT, OVERLAP_TOKENS_SEGMENT
    )

    # (선택) 디버그: 각 세그먼트 토큰 길이 확인
    # for idx, s in enumerate(segments, 1):
    #     print(f"[DEBUG] seg#{idx} tokens={tokenize_len(tokenizer, s)}")

    # 번역
    translated_segments = translate_segments(pipe, segments)

    # 재조립
    final_text = rebuild_text_from_segments(translated_segments, lb_marks)

    # 파일 저장
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(final_text)

    # ✅ 터미널 출력(설정에 따라 full 또는 미리보기)
    print_result_to_terminal(final_text, PRINT_MODE)

    # 상태 메시지
    print("Saved to: {}".format(OUTPUT_PATH))
    print("Segments translated: {} (batch size = {})".format(len(segments), BATCH_SIZE))
    print("Done.")


if __name__ == "__main__":
    main()
