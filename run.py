import os
import cv2
from easyocr import Reader

# GPU 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def get_files(path):
    file_list = []

    files = [f for f in os.listdir(path) if not f.startswith('.')]  # skip hidden file
    files.sort()
    abspath = os.path.abspath(path)
    for file in files:
        file_path = os.path.join(abspath, file)
        file_list.append(file_path)

    return file_list, len(file_list)


if __name__ == '__main__':

    # # Using default model
    # reader = Reader(['ko'], gpu=True)

    # Using custom model
    reader = Reader(['ko'], gpu=False,
                    model_storage_directory='model',
                    user_network_directory='user_network',
                    recog_network='custom')

    files, count = get_files("C:/Users/USER/EasyOCR/demo_image")

    for idx, file in enumerate(files):
        filename = os.path.basename(file)

        # 이미지 읽기
        img = cv2.imread(file, cv2.IMREAD_COLOR)
        print("Image shape:", img.shape)
        if img is None:
            print(f"Could not read image: {file}")
            continue
        
        # 이미지가 그레이스케일인 경우 RGB로 변환
        if img.shape[2] == 1:  # 흑백 이미지인 경우
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) 
        print("Final Image shape:", img.shape)     
            
        if img.ndim == 2:  # 2차원 배열인 경우 (즉, 그레이스케일)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # BGR로 변환
        elif img.shape[2] == 3:  # 3채널 (RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환
        else:
            print(f"Unexpected number of channels in {filename}: {img.shape[2]}")        
        try:
            result = reader.readtext(img)
        except RuntimeError as e:
            print("RuntimeError:", e)

        # ./easyocr/utils.py 733 lines
        # result[0]: bbox
        # result[1]: string
        # result[2]: confidence
        for (bbox, string, confidence) in result:
            print("filename: '%s', confidence: %.4f, string: '%s'" % (filename, confidence, string))
            # print('bbox: ', bbox)