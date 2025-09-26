from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

queries = [
    "AI 모델 배포 방법을 알려줘",
    "머신러닝 서비스를 운영 환경에 배포하는 방법은?",
    "점심 메뉴 추천해줘",
]
emb = model.encode(queries, convert_to_tensor=True)
sim = util.cos_sim(emb, emb)  # 3x3 코사인 유사도 행렬
print(sim)

# 가장 유사한 문장 쌍 찾기
pair = util.semantic_search(emb[0].unsqueeze(0), emb, top_k=3)[0]
for hit in pair:
    print(queries[hit['corpus_id']], hit['score'])
