from sentence_transformers import SentenceTransformer, util
import torch

# 1) 모델 로드 (GPU 있으면 자동 사용)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device=device)

# 2) 비교할 10개 문장
queries = [
    "AI 모델 배포 방법을 알려줘",
    "머신러닝 서비스를 운영 환경에 배포하는 방법은?",
    "점심 메뉴 추천해줘",
    "쿠버네티스에서 모델 서빙을 어떻게 구성하나요?",
    "도커 이미지를 사용해서 예측 API를 만들고 싶어",
    "MLOps 파이프라인의 핵심 단계는 뭐야?",
    "오늘 서울 날씨 어때?",
    "CI/CD로 모델을 자동 배포하고 싶어",
    "가벼운 샐러드나 파스타 어때?",
    "실시간 추론과 배치 추론 차이를 설명해줘",
]

# 3) 임베딩 계산 (정규화 권장)
with torch.no_grad():
    emb = model.encode(queries, convert_to_tensor=True, normalize_embeddings=True)

# 4) 10x10 코사인 유사도 행렬
sim = util.cos_sim(emb, emb)
print("=== Cosine similarity matrix (10x10) ===")
print(sim)

# 5) 각 문장별 Top-3 유사 문장 (자기 자신 제외)
print("\n=== Per-query Top-3 similar sentences (excluding self) ===")
for i, q in enumerate(queries):
    hits = util.semantic_search(emb[i].unsqueeze(0), emb, top_k=4)[0]  # self 포함 최대 4개
    top = [h for h in hits if h["corpus_id"] != i][:3]
    print(f"\n[Query #{i:02d}] {q}")
    for rank, h in enumerate(top, 1):
        print(f"  {rank}. (#{h['corpus_id']:02d}) {queries[h['corpus_id']]}  score={h['score']:.4f}")

# 6) 전체에서 유사한 문장쌍 Top-10 (자기자신/중복 자동 제외)
# ❗ 'corpus' 인자 제거
pairs = util.paraphrase_mining_embeddings(emb)  # [(score, i, j), ...] 내림차순

# (선택) 임계값 필터링: 너무 낮은 점수 제거하고 싶다면 주석 해제
# MIN_SCORE = 0.3
# pairs = [p for p in pairs if p[0] >= MIN_SCORE]

print("\n=== Global Top-10 most similar sentence pairs (paraphrase mining) ===")
for k, (score, i, j) in enumerate(pairs[:10], 1):
    print(f"{k:2d}. ({i:02d}, {j:02d}) score={score:.4f}")
    print(f"    - {queries[i]}")
    print(f"    - {queries[j]}")

# C:\Users\Administrator\PycharmProjects\PythonProject\.venv\Scripts\python.exe C:\Users\Administrator\PycharmProjects\PythonProject\huggingface\문장임베딩의미유사도\문장임베딩의미유사도-문장10.py
# === Cosine similarity matrix (10x10) ===
# tensor([[ 1.0000,  0.3258,  0.6915,  0.4719,  0.3875,  0.3130,  0.2662,  0.3774,
#           0.0071,  0.7861],
#         [ 0.3258,  1.0000,  0.0979,  0.3629,  0.3292,  0.5255,  0.0720,  0.3853,
#          -0.0243,  0.1769],
#         [ 0.6915,  0.0979,  1.0000,  0.2807,  0.0862,  0.1429,  0.4012,  0.1350,
#           0.3199,  0.8013],
#         [ 0.4719,  0.3629,  0.2807,  1.0000,  0.4818,  0.4007,  0.0506,  0.4212,
#           0.0981,  0.3545],
#         [ 0.3875,  0.3292,  0.0862,  0.4818,  1.0000,  0.2891,  0.0606,  0.3120,
#           0.0079,  0.2173],
#         [ 0.3130,  0.5255,  0.1429,  0.4007,  0.2891,  1.0000,  0.0753,  0.3108,
#          -0.0471,  0.3030],
#         [ 0.2662,  0.0720,  0.4012,  0.0506,  0.0606,  0.0753,  1.0000,  0.0209,
#           0.1368,  0.3872],
#         [ 0.3774,  0.3853,  0.1350,  0.4212,  0.3120,  0.3108,  0.0209,  1.0000,
#          -0.0444,  0.2289],
#         [ 0.0071, -0.0243,  0.3199,  0.0981,  0.0079, -0.0471,  0.1368, -0.0444,
#           1.0000,  0.1245],
#         [ 0.7861,  0.1769,  0.8013,  0.3545,  0.2173,  0.3030,  0.3872,  0.2289,
#           0.1245,  1.0000]])

# === Per-query Top-3 similar sentences (excluding self) ===
#
# [Query #00] AI 모델 배포 방법을 알려줘
#   1. (#09) 실시간 추론과 배치 추론 차이를 설명해줘  score=0.7861
#   2. (#02) 점심 메뉴 추천해줘  score=0.6915
#   3. (#03) 쿠버네티스에서 모델 서빙을 어떻게 구성하나요?  score=0.4719
#
# [Query #01] 머신러닝 서비스를 운영 환경에 배포하는 방법은?
#   1. (#05) MLOps 파이프라인의 핵심 단계는 뭐야?  score=0.5255
#   2. (#07) CI/CD로 모델을 자동 배포하고 싶어  score=0.3853
#   3. (#03) 쿠버네티스에서 모델 서빙을 어떻게 구성하나요?  score=0.3629
#
# [Query #02] 점심 메뉴 추천해줘
#   1. (#09) 실시간 추론과 배치 추론 차이를 설명해줘  score=0.8013
#   2. (#00) AI 모델 배포 방법을 알려줘  score=0.6915
#   3. (#06) 오늘 서울 날씨 어때?  score=0.4012
#
# [Query #03] 쿠버네티스에서 모델 서빙을 어떻게 구성하나요?
#   1. (#04) 도커 이미지를 사용해서 예측 API를 만들고 싶어  score=0.4818
#   2. (#00) AI 모델 배포 방법을 알려줘  score=0.4719
#   3. (#07) CI/CD로 모델을 자동 배포하고 싶어  score=0.4212
#
# [Query #04] 도커 이미지를 사용해서 예측 API를 만들고 싶어
#   1. (#03) 쿠버네티스에서 모델 서빙을 어떻게 구성하나요?  score=0.4818
#   2. (#00) AI 모델 배포 방법을 알려줘  score=0.3875
#   3. (#01) 머신러닝 서비스를 운영 환경에 배포하는 방법은?  score=0.3292
#
# [Query #05] MLOps 파이프라인의 핵심 단계는 뭐야?
#   1. (#01) 머신러닝 서비스를 운영 환경에 배포하는 방법은?  score=0.5255
#   2. (#03) 쿠버네티스에서 모델 서빙을 어떻게 구성하나요?  score=0.4007
#   3. (#00) AI 모델 배포 방법을 알려줘  score=0.3130
#
# [Query #06] 오늘 서울 날씨 어때?
#   1. (#02) 점심 메뉴 추천해줘  score=0.4012
#   2. (#09) 실시간 추론과 배치 추론 차이를 설명해줘  score=0.3872
#   3. (#00) AI 모델 배포 방법을 알려줘  score=0.2662
#
# [Query #07] CI/CD로 모델을 자동 배포하고 싶어
#   1. (#03) 쿠버네티스에서 모델 서빙을 어떻게 구성하나요?  score=0.4212
#   2. (#01) 머신러닝 서비스를 운영 환경에 배포하는 방법은?  score=0.3853
#   3. (#00) AI 모델 배포 방법을 알려줘  score=0.3774
#
# [Query #08] 가벼운 샐러드나 파스타 어때?
#   1. (#02) 점심 메뉴 추천해줘  score=0.3199
#   2. (#06) 오늘 서울 날씨 어때?  score=0.1368
#   3. (#09) 실시간 추론과 배치 추론 차이를 설명해줘  score=0.1245
#
# [Query #09] 실시간 추론과 배치 추론 차이를 설명해줘
#   1. (#02) 점심 메뉴 추천해줘  score=0.8013
#   2. (#00) AI 모델 배포 방법을 알려줘  score=0.7861
#   3. (#06) 오늘 서울 날씨 어때?  score=0.3872
#
# === Global Top-10 most similar sentence pairs (paraphrase mining) ===
#  1. (02, 09) score=0.8013
#     - 점심 메뉴 추천해줘
#     - 실시간 추론과 배치 추론 차이를 설명해줘
#  2. (00, 09) score=0.7861
#     - AI 모델 배포 방법을 알려줘
#     - 실시간 추론과 배치 추론 차이를 설명해줘
#  3. (00, 02) score=0.6915
#     - AI 모델 배포 방법을 알려줘
#     - 점심 메뉴 추천해줘
#  4. (01, 05) score=0.5255
#     - 머신러닝 서비스를 운영 환경에 배포하는 방법은?
#     - MLOps 파이프라인의 핵심 단계는 뭐야?
#  5. (03, 04) score=0.4818
#     - 쿠버네티스에서 모델 서빙을 어떻게 구성하나요?
#     - 도커 이미지를 사용해서 예측 API를 만들고 싶어
#  6. (00, 03) score=0.4719
#     - AI 모델 배포 방법을 알려줘
#     - 쿠버네티스에서 모델 서빙을 어떻게 구성하나요?
#  7. (03, 07) score=0.4212
#     - 쿠버네티스에서 모델 서빙을 어떻게 구성하나요?
#     - CI/CD로 모델을 자동 배포하고 싶어
#  8. (02, 06) score=0.4012
#     - 점심 메뉴 추천해줘
#     - 오늘 서울 날씨 어때?
#  9. (03, 05) score=0.4007
#     - 쿠버네티스에서 모델 서빙을 어떻게 구성하나요?
#     - MLOps 파이프라인의 핵심 단계는 뭐야?
# 10. (00, 04) score=0.3875
#     - AI 모델 배포 방법을 알려줘
#     - 도커 이미지를 사용해서 예측 API를 만들고 싶어
#
# 종료 코드 0(으)로 완료된 프로세스
