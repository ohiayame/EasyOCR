import os

# 폴더 경로 설정
folder_path = r"C:\\Users\\USER\\NumberData\\validation"

# 폴더 내 파일을 순차적으로 접근
for index, filename in enumerate(os.listdir(folder_path)):
    # 파일이 png 형식인지 확인
    if filename.endswith('.png'):
        # _ 기준으로 앞부분과 뒷부분을 분리
        name_parts = filename.split('_')
        if len(name_parts) == 2:
            front_part = name_parts[0]  # _의 앞부분 (변경하지 않을 부분)
            number_part = name_parts[1].split('.')[0]  # _의 뒷부분 (수정할 숫자 부분)

            # 새로 정리된 숫자 부분 (예: 0001, 0002 등)
            new_number = str(index + 1).zfill(5)  # 4자리로 맞춤

            # 새로운 파일명 생성
            new_filename = f"{front_part}_{new_number}.png"
            if new_filename != filename:
                # 파일 경로 변경
                os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))

print("파일 이름 정리가 완료되었습니다.")
