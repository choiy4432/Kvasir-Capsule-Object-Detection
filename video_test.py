import cv2

# =====================
# 1. 모델 로드
# =====================
model = YOLO('/content/colab/AICV_study_yolov11/data/runs/detect/train2/weights/best.pt')

input_video = "/content/colab/AICV_study_yolov11/data/video2.mp4"
output_video = "/content/colab/AICV_study_yolov11/data/result_with_counts3.mp4"

# =====================
# 2. 비디오 정보 읽기
# =====================
cap = cv2.VideoCapture(input_video)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

cap.release()

# =====================
# 3. VideoWriter 설정
# =====================
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# =====================
# 4. YOLO 추론 + 저장
# =====================
for r in model(input_video, stream=True, conf=0.25):
    img = r.plot()  # BGR 이미지

    class_counts = {}

    if r.boxes is not None and r.boxes.cls is not None:
        classes = r.boxes.cls.cpu().numpy().astype(int)

        for cls_id in classes:
            class_name = model.names[cls_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

    # 클래스별 count 화면에 표시
    y_offset = 30
    for cls_name, count in class_counts.items():
        cv2.putText(
            img,
            f"{cls_name}: {count}",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )
        y_offset += 30

    out.write(img)

# =====================
# 5. 마무리
# =====================
out.release()

print(f"✅ 저장 완료: {output_video}")
