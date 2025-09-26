# -*- coding: utf-8 -*-
# batch_news_zero_shot_ko.py  (Python 3.8)

import csv
import time
import re
import requests
import trafilatura
from trafilatura.settings import use_config
from bs4 import BeautifulSoup
from newspaper import Article, Config
from transformers import pipeline
from typing import Optional, List, Tuple, Dict, Any

# ---------------------------
# 설정
# ---------------------------
HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
}
TRAFI_CONFIG = use_config()
TRAFI_CONFIG.set("DEFAULT", "USER_AGENTS", HEADERS["User-Agent"])

LABELS: List[str] = ["정치", "경제", "사회", "국제", "스포츠", "문화", "사설", "리뷰"]
TOP_K: int = 5
OUTPUT_CSV: str = "news_zero_shot_results.csv"
SLEEP_BETWEEN_REQ: float = 1.0  # 요청 간격(초): 차단 예방용

# 테스트용 URL들
URLS: List[str] = [
    # 조선/아시아경제/한겨레/네이버 등 섞어서 테스트
    "https://www.chosun.com/politics/politics_general/2025/09/26/7KEXY5JRS5DCTJ6J64UZGFQLCM/",
    "https://view.asiae.co.kr/article/2025092608572502066",
    # "https://n.news.naver.com/mnews/article/001/0011234567",
    "https://www.chosun.com/politics/politics_general/2025/09/26/7KEXY5JRS5DCTJ6J64UZGFQLCM/",
    "https://www.chosun.com/international/international_general/2025/09/26/XMDHEX3SNFCEDHHRI36V2CNUD4/",
    "https://www.chosun.com/international/international_general/2025/09/26/J6JZWKT7I5HNNF6Q3RCQCSFORU/",
    "https://www.chosun.com/international/china/2025/09/26/JVJD3AXYWVFDDBLO5DJK4LWFZ4/",
    "https://www.chosun.com/international/international_general/2025/09/26/UOV45W4IMRCIHMKCECLORHBEIQ/",
    "https://www.chosun.com/international/international_general/2025/09/26/JFV75OSBJJFRXOJWPM2HDQEATA/",
    "https://www.chosun.com/international/international_general/2025/09/26/3V3AG4CFMNGLHHVL55B5EOUPQA/",
    "https://www.chosun.com/international/international_general/2025/09/26/XMDHEX3SNFCEDHHRI36V2CNUD4",
]


# ---------------------------
# 공통 유틸
# ---------------------------
def http_get(url: str) -> Optional[str]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        if r.status_code == 200 and r.text:
            return r.text
        return None
    except Exception:
        return None

def to_amp(url: str) -> Optional[str]:
    if url.startswith("https://www.chosun.com/"):
        return url + ("amp/" if url.endswith("/") else "/amp/")
    return None

def to_mobile(url: str) -> Optional[str]:
    if url.startswith("https://www.chosun.com/"):
        return url.replace("https://www.chosun.com", "https://m.chosun.com", 1)
    return None

def to_print(url: str) -> Optional[str]:
    return url + ("&" if "?" in url else "?") + "outputType=amp"

# ---------------------------
# 본문/제목 추출
# ---------------------------
TRAFI_CONFIG = use_config()
TRAFI_CONFIG.set("DEFAULT", "USER_AGENTS", HEADERS["User-Agent"])

def extract_with_trafilatura(url: str) -> Tuple[Optional[str], Optional[str]]:
    downloaded = trafilatura.fetch_url(url, config=TRAFI_CONFIG)
    if not downloaded:
        return None, None

    # 1) 본문 (config 사용 가능)
    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        favor_recall=True,
        url=url,
        target_language="ko",
        with_metadata=False,      # 본문만 추출
        config=TRAFI_CONFIG,
    )

    if text:
        text = text.strip()

    # 2) 제목 (⚠ config 인자 없이 호출)
    title_str = None
    try:
        md = trafilatura.extract_metadata(downloaded)  # ← config 인자 제거
        if md and md.title:
            title_str = md.title.strip()
    except Exception:
        title_str = None

    if text:
        return text, title_str
    return None, None

def extract_title_from_html(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.text:
        return soup.title.text.strip()
    # og:title 등 보조 시도
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return og["content"].strip()
    tw = soup.find("meta", attrs={"name": "twitter:title"})
    if tw and tw.get("content"):
        return tw["content"].strip()
    return None

def extract_with_bs4(html: str) -> Tuple[Optional[str], Optional[str]]:
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
    text_out = max(texts, key=len) if texts else None
    title_out = extract_title_from_html(html)
    return text_out, title_out

def extract_with_newspaper(url: str) -> Tuple[Optional[str], Optional[str]]:
    cfg = Config()
    cfg.browser_user_agent = HEADERS["User-Agent"]
    cfg.request_timeout = 12
    article = Article(url, language="ko", memoize_articles=False, config=cfg)
    article.download()
    article.parse()
    text = (article.text or "").strip()
    title = (article.title or "").strip()
    return (text or None), (title or None)

def smart_extract(url: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    반환: (사용전략, 본문, 제목) / 실패 시 ("", None, None)
    """
    # 1) trafilatura 원본
    txt, ttl = extract_with_trafilatura(url)
    if txt:
        return "trafilatura", txt, ttl

    # 2) AMP
    amp = to_amp(url)
    if amp:
        txt, ttl = extract_with_trafilatura(amp)
        if txt:
            return "trafilatura-amp", txt, ttl
        html = http_get(amp)
        if html:
            txt, ttl = extract_with_bs4(html)
            if txt:
                return "bs4-amp", txt, ttl

    # 3) 모바일
    mob = to_mobile(url)
    if mob:
        txt, ttl = extract_with_trafilatura(mob)
        if txt:
            return "trafilatura-mobile", txt, ttl
        html = http_get(mob)
        if html:
            txt, ttl = extract_with_bs4(html)
            if txt:
                return "bs4-mobile", txt, ttl

    # 4) 프린트/AMP 쿼리
    prt = to_print(url)
    if prt and prt != amp:
        txt, ttl = extract_with_trafilatura(prt)
        if txt:
            return "trafilatura-print", txt, ttl
        html = http_get(prt)
        if html:
            txt, ttl = extract_with_bs4(html)
            if txt:
                return "bs4-print", txt, ttl

    # 5) newspaper3k
    try:
        txt, ttl = extract_with_newspaper(url)
        if txt:
            return "newspaper3k", txt, ttl
    except Exception:
        pass

    # 6) 원본 HTML → bs4
    html = http_get(url)
    if html:
        txt, ttl = extract_with_bs4(html)
        if txt:
            return "bs4-original", txt, ttl

    return "", None, None

# ---------------------------
# 분류
# ---------------------------
def build_classifier():
    clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return clf

def classify_text(clf, text: str, labels: List[str], top_k: int = TOP_K) -> Dict[str, Any]:
    res = clf(
        text,
        candidate_labels=labels,
        hypothesis_template="This sentence is about {}.",
        truncation=True,
    )
    labels_sorted = res["labels"][:top_k]
    scores_sorted = [f"{s:.4f}" for s in res["scores"][:top_k]]
    out = {
        "top1_label": labels_sorted[0] if labels_sorted else "",
        "top1_score": scores_sorted[0] if scores_sorted else "",
        "topk_labels": "|".join(labels_sorted),
        "topk_scores": "|".join(scores_sorted),
        "raw": res,
    }
    return out

# ---------------------------
# 배치 실행
# ---------------------------
def run_batch(urls: List[str]) -> List[Dict[str, Any]]:
    clf = build_classifier()
    rows: List[Dict[str, Any]] = []

    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] {url}")
        time.sleep(SLEEP_BETWEEN_REQ)

        try:
            strategy, text, title = smart_extract(url)
            if not text:
                rows.append({
                    "url": url, "title": title or "",
                    "strategy": strategy,
                    "predicted_label_top1": "", "score_top1": "",
                    "top_k_labels": "", "top_k_scores": "",
                    "error": "본문 추출 실패"
                })
                print("  └─ 실패: 본문 추출 실패")
                continue

            result = classify_text(clf, text, LABELS, TOP_K)
            rows.append({
                "url": url, "title": (title or ""),
                "strategy": strategy,
                "predicted_label_top1": result["top1_label"],
                "score_top1": result["top1_score"],
                "top_k_labels": result["topk_labels"],
                "top_k_scores": result["topk_scores"],
                "error": ""
            })
            print(f"  └─ 예측: {result['top1_label']} (score={result['top1_score']})  / 전략={strategy}")

        except Exception as e:
            rows.append({
                "url": url, "title": "",
                "strategy": "",
                "predicted_label_top1": "", "score_top1": "",
                "top_k_labels": "", "top_k_scores": "",
                "error": f"처리 오류: {e}"
            })
            print(f"  └─ 오류: {e}")

    return rows

def save_csv(rows: List[Dict[str, Any]], path: str):
    cols = ["url", "title", "strategy",
            "predicted_label_top1", "score_top1",
            "top_k_labels", "top_k_scores", "error"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nCSV 저장 완료 → {path}")

if __name__ == "__main__":
    results = run_batch(URLS)
    save_csv(results, OUTPUT_CSV)

# C:\Users\Administrator\PycharmProjects\PythonProject\.venv\Scripts\python.exe C:\Users\Administrator\PycharmProjects\PythonProject\huggingface\한국어라벨주제분류\한국어라벨주제분류3-기사여러개.py
# [1/10] https://www.chosun.com/politics/politics_general/2025/09/26/7KEXY5JRS5DCTJ6J64UZGFQLCM/
#   └─ 예측: 정치 (score=0.1531)  / 전략=trafilatura-print
# [2/10] https://view.asiae.co.kr/article/2025092608572502066
#   └─ 예측: 국제 (score=0.1545)  / 전략=trafilatura
# [3/10] https://www.chosun.com/politics/politics_general/2025/09/26/7KEXY5JRS5DCTJ6J64UZGFQLCM/
#   └─ 예측: 정치 (score=0.1622)  / 전략=trafilatura-mobile
# [4/10] https://www.chosun.com/international/international_general/2025/09/26/XMDHEX3SNFCEDHHRI36V2CNUD4/
#   └─ 예측: 정치 (score=0.1622)  / 전략=trafilatura-mobile
# [5/10] https://www.chosun.com/international/international_general/2025/09/26/J6JZWKT7I5HNNF6Q3RCQCSFORU/
#   └─ 예측: 정치 (score=0.1622)  / 전략=trafilatura-mobile
# [6/10] https://www.chosun.com/international/china/2025/09/26/JVJD3AXYWVFDDBLO5DJK4LWFZ4/
#   └─ 예측: 정치 (score=0.1622)  / 전략=trafilatura-mobile
# [7/10] https://www.chosun.com/international/international_general/2025/09/26/UOV45W4IMRCIHMKCECLORHBEIQ/
#   └─ 예측: 정치 (score=0.1622)  / 전략=trafilatura-mobile
# [8/10] https://www.chosun.com/international/international_general/2025/09/26/JFV75OSBJJFRXOJWPM2HDQEATA/
#   └─ 예측: 경제 (score=0.1485)  / 전략=trafilatura-print
# [9/10] https://www.chosun.com/international/international_general/2025/09/26/3V3AG4CFMNGLHHVL55B5EOUPQA/
#   └─ 예측: 정치 (score=0.1622)  / 전략=trafilatura-mobile
# [10/10] https://www.chosun.com/international/international_general/2025/09/26/XMDHEX3SNFCEDHHRI36V2CNUD4
#   └─ 예측: 정치 (score=0.1622)  / 전략=trafilatura-mobile
#
# CSV 저장 완료 → news_zero_shot_results.csv