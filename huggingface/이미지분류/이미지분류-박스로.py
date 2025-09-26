from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import os

# 1) 대상 이미지 URL
url = "https://cafe24img.poxo.com/andar01/web/andar/img/detail/product/anpsp15/images/250422/01.jpg"

# 2) 이미지 로드
resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
resp.raise_for_status()
img = Image.open(BytesIO(resp.content)).convert("RGB")

# 3) 객체 감지 모델 (DETR)
detector = pipeline("object-detection", model="facebook/detr-resnet-50")

# 4) 추론
results = detector(img)

# 5) 신뢰도 임계값 필터
boxes = [r for r in results if r.get("score", 0) >= 0.7]
if not boxes:
    boxes = [r for r in results if r.get("score", 0) >= 0.5]

# 6) 박스/라벨 그리기
draw = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype(r"C:\Windows\Fonts\malgun.ttf", 18)  # 윈도우 한글 폰트
except:
    font = ImageFont.load_default()

for r in boxes:
    box = r["box"]
    label = r["label"]
    score = r["score"]
    x1, y1, x2, y2 = box["xmin"], box["ymin"], box["xmax"], box["ymax"]

    # 사각형
    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)

    # 라벨 + 점수
    text = f"{label} {score:.2f}"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([x1, y1 - th - 6, x1 + tw + 8, y1], fill=(0, 255, 0))
    draw.text((x1 + 4, y1 - th - 4), text, fill=(0, 0, 0), font=font)

# 7) 저장
os.makedirs("out", exist_ok=True)
out_path = os.path.join("out", "detected_boxes.jpg")
img.save(out_path)
print(f"[완료] 결과 저장: {out_path}")

# 8) 화면 표시
img.show()
