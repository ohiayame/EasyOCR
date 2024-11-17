import cv2
import easyocr

# 이미지 읽기
img = cv2.imread(r"C:\\Users\\USER\\NumberData\\training\\4776_03272.png")

# 이미지 shape 출력
print("Initial Image shape:", img.shape)

# 흑백 이미지가 아닐 경우 3채널로 변환
if img.ndim == 2 or img.shape[2] == 1:  # 흑백 이미지인 경우
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# 최종 이미지 shape 확인
print("Final Image shape:", img.shape)
recognizer_model_path = r'C:/Users/USER/EasyOCR/model/custom.pth'
# EasyOCR 리더 초기화
# reader = easyocr.Reader(['ko'])  # 언어 설정
reader = easyocr.Reader(['ko'], recognizer=recognizer_model_path)
# OCR 수행
try:
    result = reader.readtext(img)
except RuntimeError as e:
    print("RuntimeError:", e)
    result = []  # 에러 발생 시 빈 리스트로 초기화

# 결과 출력
if result:
    for (bbox, string, confidence) in result:
        print("Detected text:", string)
else:
    print("No text detected.")
print("Image dtype:", img.dtype)