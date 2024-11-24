from difflib import SequenceMatcher
import easyocr
from ultralytics import YOLO
import cv2

# YOLO 모델 로드 및 EasyOCR 초기화
model = YOLO("C:\\Users\\USER\\EasyOCR\\yolo_model\\yolov8-custom_number_plate_toy2\\weights\\best.pt")
reader = easyocr.Reader(['ko'], recognizer=r'C:/Users/USER/EasyOCR/model/custom.pth')

# 허용할 문자 (숫자만 허용)
import re
allowed_characters = re.compile(r'\d+')

# 실제 텍스트(ground truth)와 예측된 텍스트 비교하여 정확도 계산
def calculate_accuracy(true_text, predicted_text):
    matcher = SequenceMatcher(None, true_text, predicted_text)
    return matcher.ratio()

# 데이터셋 테스트 함수
def evaluate_performance(image_paths, true_texts):
    total_accuracy = 0
    total_full_matches = 0
    total_images = len(image_paths)

    for img_path, true_text in zip(image_paths, true_texts):
        # 이미지 로드 및 YOLO 추론
        frame = cv2.imread(img_path)
        results = model(frame)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()

                if conf > 0.5:
                    x1, y1, x2, y2 = xyxy
                    # OCR로 번호판 텍스트 인식
                    plate_texts = reader.readtext(frame[y1:y2, x1:x2])

                    if plate_texts:
                        # 숫자만 필터링
                        predicted_text = "".join([c[1] for c in plate_texts if allowed_characters.match(c[1])])
                        if predicted_text:
                            # 문자 단위 정확도 계산
                            accuracy = calculate_accuracy(true_text, predicted_text)
                            total_accuracy += accuracy

                            # 전체 번호판이 완전히 일치하는 경우
                            if true_text == predicted_text:
                                total_full_matches += 1
                            print(f"실제: {true_text}, 예측: {predicted_text}, 정확도: {accuracy * 100:.2f}%")
                        else:
                            print(f"인식된 숫자가 없습니다. 실제: {true_text}")
                    else:
                        print(f"번호판 인식 결과가 없습니다. 실제: {true_text}")

    # 평균 문자 단위 정확도
    # avg_accuracy = total_accuracy / total_images
    # 번호판 단위 정확도
    full_plate_accuracy = total_full_matches / total_images

    # print(f"\n전체 문자 단위 평균 인식률: {avg_accuracy * 100:.2f}%")
    print(f"전체 번호판 단위 일치율: {full_plate_accuracy * 100:.2f}%")

# 테스트용 이미지와 실제 번호판 텍스트
image_paths = ["numberpleat\\test\\num1-1.jpg", "numberpleat\\test\\num1-2.jpg", "numberpleat\\test\\num1-3.jpg","numberpleat\\test\\num1-4.jpg","numberpleat\\test\\num1-5.jpg","numberpleat\\test\\num1-6.jpg"
,"numberpleat\\test\\num1-7.jpg","numberpleat\\test\\num1-8.jpg","numberpleat\\test\\num1-9.jpg","numberpleat\\test\\num2-1.jpg", "numberpleat\\test\\num2-2.jpg", "numberpleat\\test\\num2-3.jpg","numberpleat\\test\\num2-4.jpg","numberpleat\\test\\num2-5.jpg","numberpleat\\test\\num2-6.jpg"
,"numberpleat\\test\\num2-7.jpg","numberpleat\\test\\num2-8.jpg","numberpleat\\test\\num3-1.jpg", "numberpleat\\test\\num3-2.jpg", "numberpleat\\test\\num3-3.jpg","numberpleat\\test\\num3-4.jpg","numberpleat\\test\\num3-5.jpg","numberpleat\\test\\num3-6.jpg"
,"numberpleat\\test\\num3-7.jpg","numberpleat\\test\\num3-8.jpg","numberpleat\\test\\num3-9.jpg","numberpleat\\test\\num3-10.jpg","numberpleat\\test\\num3-11.jpg","numberpleat\\test\\num3-12.jpg","numberpleat\\test\\num3-13.jpg","numberpleat\\test\\num3-14.jpg","numberpleat\\test\\num3-15.jpg"
,"numberpleat\\test\\num4-1.jpg", "numberpleat\\test\\num4-2.jpg", "numberpleat\\test\\num4-3.jpg","numberpleat\\test\\num4-4.jpg","numberpleat\\test\\num4-5.jpg","numberpleat\\test\\num4-6.jpg"
,"numberpleat\\test\\num4-7.jpg","numberpleat\\test\\num4-8.jpg","numberpleat\\test\\num4-9.jpg","numberpleat\\test\\num5-1.jpg", "numberpleat\\test\\num5-2.jpg", "numberpleat\\test\\num5-3.jpg","numberpleat\\test\\num5-4.jpg","numberpleat\\test\\num5-5.jpg","numberpleat\\test\\num5-6.jpg"
,"numberpleat\\test\\num5-7.jpg","numberpleat\\test\\num5-8.jpg","numberpleat\\test\\num5-9.jpg"]
# ,"numberpleat\\test\\num6-1.jpg", "numberpleat\\test\\num6-2.jpg", "numberpleat\\test\\num6-3.jpg","numberpleat\\test\\num6-4.jpg","numberpleat\\test\\num6-5.jpg","numberpleat\\test\\num6-6.jpg"
# ,"numberpleat\\test\\num6-7.jpg","numberpleat\\test\\num6-8.jpg"]
true_texts = ["8137","8137","8137","8137","8137","8137","8137","8137","8137"
              ,"7649","7649","7649","7649","7649","7649","7649","7649"
              ,"2593","2593","2593","2593","2593","2593","2593","2593","2593","2593","2593","2593","2593","2593","2593"
              , "2267", "2267", "2267", "2267", "2267", "2267", "2267", "2267", "2267"
              ,"0214","0214","0214","0214","0214","0214","0214","0214","0214"]
            #   ,"8411","8411","8411","8411","8411","8411","8411","8411"]  # 실제 번호판 텍스트

# 성능 평가
evaluate_performance(image_paths, true_texts)
