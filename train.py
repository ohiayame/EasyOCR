import tkinter as tk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import easyocr
import torch

# YOLOv8 모델 로드
model = YOLO("C:\\Users\\USER\\EasyOCR\\yolo_model\\yolov8-custom_number_plate_toy2\\weights\\best.pt")

# CUDA가 가능하면 FP16 적용, CPU 모드에서는 사용하지 않음
if torch.cuda.is_available():
    model.model.half()  # GPU에서만 FP16 적용

cap = cv2.VideoCapture(0)
recognizer_model_path = r'C:/Users/USER/EasyOCR/model/custom.pth'

# easyocr Reader는 한번만 초기화
# reader = easyocr.Reader(['ko'], recognizer=recognizer_model_path)
reader = easyocr.Reader(['ko'])
# 특정 구역
zone_x1, zone_y1 = 100, 100
zone_x2, zone_y2 = 400, 400
frame_count = 0 

# Tkinter 윈도우 생성
root = tk.Tk()
label = tk.Label(root)
label.pack()

def show_frame():
    global frame_count
    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.resize(frame, (640, 480))  # 해상도 조정
    
    # 구역 표시
    cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (255, 0, 0), 2)
    frame_count += 1
    # if frame_count % 5 == 0:
    # 모델 추론
    results = model(frame)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].cpu().numpy()

            if conf > 0.5:
                x1, y1, x2, y2 = xyxy
                if (x1 >= zone_x1 and x2 <= zone_x2 and y1 >= zone_y1 and y2 <= zone_y2):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # 번호판 텍스트 인식
                    plate_texts = reader.readtext(frame[y1:y2, x1:x2])
                    
                    if plate_texts:
                        plate_text_build = "".join([c[1] for c in plate_texts if c[1].strip()])

                        # 인식된 텍스트가 있을 때만 표시
                        if plate_text_build:
                            cv2.putText(frame, plate_text_build.strip(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                            print(f"인식된 텍스트: {plate_text_build.strip()}")
                        else:
                            print("텍스트 인식 실패 또는 빈 텍스트")
                    else:
                        print("번호판 인식 결과가 없습니다.")

    # OpenCV 이미지를 Pillow 이미지로 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    # Tkinter Label에 이미지 표시
    label.imgtk = imgtk
    label.configure(image=imgtk)

    # 1ms마다 업데이트
    root.after(1, show_frame)

# 실시간 영상 표시 시작
show_frame()

# Tkinter 이벤트 루프 시작
root.mainloop()

# 비디오 캡처 및 윈도우 해제
cap.release()
cv2.destroyAllWindows()
