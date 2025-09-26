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

# C:\Users\Administrator\PycharmProjects\PythonProject\.venv\Scripts\python.exe C:\Users\Administrator\PycharmProjects\PythonProject\huggingface\문장임베딩의미유사도\문장임베딩의미유사도.py
# tensor([[1.0000, 0.3258, 0.6915],
#         [0.3258, 1.0000, 0.0979],
#         [0.6915, 0.0979, 1.0000]])
# AI 모델 배포 방법을 알려줘 0.9999998807907104
# 점심 메뉴 추천해줘 0.6915063858032227
# 머신러닝 서비스를 운영 환경에 배포하는 방법은? 0.3258000314235687