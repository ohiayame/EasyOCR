from difflib import SequenceMatcher
import easyocr
import cv2
import re

# OCR 초기화
# recognizer_model_path = r'model/4_Num과적합/custom.pth'
# "C:\\Users\\USER\\Parking_control_system\\Parking_control_system_EasyOCR\\model\\4_Num과적합\cusom.pth"
reader = easyocr.Reader(['ko'], gpu=False, 
    model_storage_directory=r'C:\\Users\\USER\\Parking_control_system\\Parking_control_system_EasyOCR\\model\\1_onlyNum',
    user_network_directory='user_network',
    recog_network='custom')

# reader = easyocr.Reader(['ko'])
# 허용할 문자: 숫자만
only_digits_pattern = re.compile(r'\d+')

# 정확도 계산 함수
def calculate_accuracy(true_text, predicted_text):
    matcher = SequenceMatcher(None, true_text, predicted_text)
    return matcher.ratio()

# 데이터셋 테스트 함수
def evaluate_performance(image_paths, true_texts):
    total_accuracy = 0
    total_full_matches = 0
    total_images = len(image_paths)

    for img_path, true_text in zip(image_paths, true_texts):
        # 이미지 로드
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"이미지를 로드할 수 없습니다: {img_path}")
            continue

        # EasyOCR 수행
        try:
            result = reader.readtext(frame)
        except RuntimeError as e:
            print(f"RuntimeError: {e}")
            continue

        
        # OCR 결과 처리
        detected_texts = []
        for bbox, string, confidence in result:
            # 숫자만 남기기
            filtered_text = "".join(only_digits_pattern.findall(string))
            if filtered_text:
                detected_texts.append(filtered_text)

        if detected_texts:
            # 가장 긴 텍스트를 선택
            predicted_text = max(detected_texts, key=len)
            accuracy = calculate_accuracy(true_text, predicted_text)
            total_accuracy += accuracy

            if true_text == predicted_text:
                total_full_matches += 1

            print(f"이미지: {img_path}, 실제: {true_text}, 예측: {predicted_text}, 정확도: {accuracy * 100:.2f}%")
        else:
            print(f"이미지: {img_path}, 실제: {true_text}, 예측: 없음")

    # 평균 문자 단위 정확도
    avg_accuracy = total_accuracy / total_images if total_images > 0 else 0
    # 번호판 단위 정확도
    full_plate_accuracy = total_full_matches / total_images if total_images > 0 else 0

    print(f"\n전체 문자 단위 평균 인식률: {avg_accuracy * 100:.2f}%")
    print(f"전체 번호판 단위 일치율: {full_plate_accuracy * 100:.2f}%")

# 파일에서 이미지 경로와 레이블 읽기
filename = "gt.txt"
image_paths = []
true_texts = []

with open(filename, "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split("\t")
        image_paths.append(parts[0])
        true_texts.append(parts[1])

# 성능 평가
evaluate_performance(image_paths, true_texts)