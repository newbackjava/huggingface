# -*- coding: utf-8 -*-
# Python 3.8 compatible version (trafilatura headers fix)

import re
import requests
import trafilatura
from trafilatura.settings import use_config
from bs4 import BeautifulSoup
from newspaper import Article, Config
from transformers import pipeline
from typing import Optional, List, Tuple, Dict, Any

HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
}

# --- trafilatura 전용 설정 (User-Agent 주입) ---
TRAFI_CONFIG = use_config()
TRAFI_CONFIG.set(
    "DEFAULT",
    "USER_AGENTS",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

def to_amp(url: str) -> Optional[str]:
    # 조선닷컴은 보통 /amp/ 변형 제공
    if not url.startswith("https://www.chosun.com/"):
        return None
    if url.endswith("/"):
        return url + "amp/"
    return url + "/amp/"

def to_mobile(url: str) -> Optional[str]:
    if not url.startswith("https://www.chosun.com/"):
        return None
    return url.replace("https://www.chosun.com", "https://m.chosun.com", 1)

def to_print(url: str) -> Optional[str]:
    # print 전용 경로가 고정은 아니므로 AMP 유사 쿼리로 재시도
    return url + ("&" if "?" in url else "?") + "outputType=amp"

def http_get(url: str) -> Optional[str]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        if r.status_code == 200 and r.text:
            return r.text
        return None
    except Exception:
        return None

def extract_with_trafilatura(url: str) -> Optional[str]:
    # ✅ fetch_url은 headers 파라미터를 받지 않으므로 config로 UA 설정
    downloaded = trafilatura.fetch_url(url, config=TRAFI_CONFIG)
    if not downloaded:
        return None
    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        favor_recall=True,
        url=url,
        target_language="ko",
        with_metadata=False,
        config=TRAFI_CONFIG,  # ✅ extract에도 동일 config 전달
    )
    if text and text.strip():
        return text.strip()
    return None

def extract_with_bs4(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
    candidates = [
        {"name": "div", "attrs": {"class": re.compile(r"(article-body|article-content|story-news|news-body)")}},
        {"name": "section", "attrs": {"class": re.compile(r"(article|content)")}},
        {"name": "div", "attrs": {"id": re.compile(r"(content|article)")}},
    ]
    texts: List[str] = []
    for sel in candidates:
        for node in soup.find_all(sel["name"], attrs=sel["attrs"]):
            for s in node(["script", "style", "noscript"]):
                s.decompose()
            t = node.get_text(separator="\n", strip=True)
            if t and len(t) > 100:
                texts.append(t)
    if texts:
        return max(texts, key=len)
    return None

def extract_with_newspaper(url: str) -> Optional[str]:
    cfg = Config()
    cfg.browser_user_agent = HEADERS["User-Agent"]
    cfg.request_timeout = 12
    article = Article(url, language="ko", memoize_articles=False, config=cfg)
    article.download()
    article.parse()
    t = (article.text or "").strip()
    return t or None

def smart_extract(url: str) -> Tuple[str, str]:
    """
    여러 전략을 순차 시도해 본문을 최대 확보.
    반환: (사용전략, 본문텍스트) / 실패 시 ("", "")
    """
    # 1) trafilatura 원본
    txt = extract_with_trafilatura(url)
    if txt:
        return ("trafilatura", txt)

    # 2) AMP 변형
    amp = to_amp(url)
    if amp:
        txt = extract_with_trafilatura(amp)
        if txt:
            return ("trafilatura-amp", txt)
        html = http_get(amp)
        if html:
            txt = extract_with_bs4(html)
            if txt:
                return ("bs4-amp", txt)

    # 3) 모바일 변형
    mob = to_mobile(url)
    if mob:
        txt = extract_with_trafilatura(mob)
        if txt:
            return ("trafilatura-mobile", txt)
        html = http_get(mob)
        if html:
            txt = extract_with_bs4(html)
            if txt:
                return ("bs4-mobile", txt)

    # 4) 프린트/AMP 쿼리
    prt = to_print(url)
    if prt and prt != amp:
        txt = extract_with_trafilatura(prt)
        if txt:
            return ("trafilatura-print", txt)
        html = http_get(prt)
        if html:
            txt = extract_with_bs4(html)
            if txt:
                return ("bs4-print", txt)

    # 5) newspaper3k
    try:
        txt = extract_with_newspaper(url)
        if txt:
            return ("newspaper3k", txt)
    except Exception:
        pass

    # 6) 원본 HTML → bs4
    html = http_get(url)
    if html:
        txt = extract_with_bs4(html)
        if txt:
            return ("bs4-original", txt)

    return ("", "")

def classify(text: str, labels: List[str]) -> Dict[str, Any]:
    clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    res = clf(
        text,
        candidate_labels=labels,
        hypothesis_template="This sentence is about {}.",
        truncation=True,
    )
    return res

if __name__ == "__main__":
    # 테스트: 실제 기사 URL로 교체하세요
    # url = "https://www.chosun.com/politics/politics_general/2025/09/26/7KEXY5JRS5DCTJ6J64UZGFQLCM/"
    url = "https://view.asiae.co.kr/article/2025092608572502066?utm_source=newsstand.naver.com&utm_medium=referral&utm_campaign=top1"
    strategy, text = smart_extract(url)
    if not text:
        raise SystemExit("본문 추출 실패: 모든 전략이 실패했습니다.")

    print("[추출전략]", strategy)
    preview = text[:300].replace("\n", " ")
    print("[미리보기]", preview, "...\n")

    labels = ["정치", "경제", "사회", "국제", "스포츠", "문화", "사설", "리뷰"]
    res = classify(text, labels)
    print("예측된 분류:", res["labels"][0], "/ score=", "{:.4f}".format(res["scores"][0]))
