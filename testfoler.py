import os
import cv2
import easyocr
import re

# detector_model_path = r'C:/Users/USER/EasyOCR/model/craft_mlt_25k.pth'
recognizer_model_path = r'C:/Users/USER/EasyOCR/model/custom.pth'
char_set = "1234567890나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주아바사자하허호배"

# EasyOCR 리더 초기화
reader = easyocr.Reader(['ko'], recognizer=recognizer_model_path)
# reader = easyocr.Reader(['ko'], gpu=False)

# 이미지 폴더 경로
image_folder = r"C:\\Users\\USER\\NumberData\\training"  # 이미지 폴더 경로로 변경하세요
images = os.listdir(image_folder)

# 정규 표현식 패턴
pattern = r'(\d{2}[나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주아바사자하허호배]\d{4}|\d{3}[나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주아바사자하허호배]\d{4})'

# 최대 10개의 이미지 처리
for img_name in images: # [:10]:
    img_path = os.path.join(image_folder, img_name)
    
    # 이미지 읽기
    img = cv2.imread(img_path)

    if img is not None:
        # OCR 수행
        result = reader.readtext(img, allowlist=char_set)
        # result = reader.readtext(img)
        
        # 결과 출력
        print(f"파일명: {img_name}")
        for (bbox, text, confidence) in result:
            # 정규 표현식으로 필터링
            # if re.match(pattern, text):
            print(f"탐지된 텍스트: {text}, 신뢰도: {confidence:.2f}")
        
        
    else:
        print(f"{img_name}를 읽을 수 없습니다.")
