from transformers import pipeline
from newspaper import Article

# 1. Zero-shot 분류 파이프라인 (영문 MNLI 기반)
zero = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# 2. 기사 URL
url = "https://n.news.naver.com/mnews/article/001/0011234567"


# 3. 본문 크롤링 (예외 처리 포함)
article = Article(url, language="ko")
article.download()
article.parse()
text = (article.text or "").strip()
if not text:
    raise ValueError("본문이 비어 있습니다. 로그인/페이월/동적로딩 페이지일 수 있습니다.")

# 4. 라벨 후보
labels = [
    "영화", "드라마", "음악", "문학", "게임",
    "패션", "미술", "무용", "웹툰", "예능", "정치", "경제", "사회", "스포츠"
]

# 5. 제로샷 분류 실행
# - 영어 템플릿 추천: MNLI 모델 특성상 일관성 ↑
# - 한국어 템플릿도 가능하지만 점수 안정성은 영어가 대체로 좋음
res = zero(
    text,
    candidate_labels=labels,
    hypothesis_template="This sentence is about {}.",
    truncation=True
)

# 6. 결과 출력
print("기사 제목:", article.title)
print("예측된 분류:", res["labels"][0])
print("상세 결과:", res)

#
# 기사 제목: '국군간호사관학교 성희롱 단톡방 사건 은폐 관련'
# 예측된 분류: 음악
# 상세 결과: {'sequence': '“매달 통장에 2000만원이 따박따박”…‘억대 연봉’ 강남 임대업자, 알고 보니 14살?\n\n사업장 대표로 등록된 미성년자의 월 평균 소득이 300만원을 넘어선 것으로 파악됐다. 상속이나 증여를 통해 미성년자가 고소득 사업장 대표로 등재되는 사례가 잇따르고 있는 가운데, 불법은 아니지만 편법 증여 등의 가능', 'labels': ['음악', '미술', '예능', '문학', '게임', '드라마', '패션', '스포츠', '사회', '경제', '무용', '정치', '영화', '웹툰'], 'scores': [0.2907235324382782, 0.17780892550945282, 0.13082343339920044, 0.10832099616527557, 0.07539954036474228, 0.04843072220683098, 0.024780303239822388, 0.0246605034917593, 0.024450985714793205, 0.020774610340595245, 0.019940655678510666, 0.019600218161940575, 0.01846097595989704, 0.015824565663933754]}
