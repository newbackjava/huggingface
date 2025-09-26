from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

clf = pipeline("image-classification", model="google/vit-base-patch16-224")

# 샘플: 웹 이미지
# url = "https://huggingface.co/front/thumbnails/models.png"
# url = "https://ssl.pstatic.net/melona/libs/1546/1546409/4de5a58c57f7988d4df6_20250925152541812.jpg"
# url = "https://imgnews.pstatic.net/image/009/2025/09/26/0005565375_001_20250926111214626.jpg?type=w860"
url = "https://cafe24img.poxo.com/andar01/web/andar/img/detail/product/anpsp15/images/250422/01.jpg"
img = Image.open(BytesIO(requests.get(url, timeout=10).content)).convert("RGB")

print(clf(img)[:3])  # 상위 3개 라벨
