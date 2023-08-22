from ultralytics import YOLO
import cv2
import math

def video_detection(path_x):
    video_capture = path_x
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    model = YOLO("YOLO-Weights/nepaliitem.pt")
    classNames = ['dhaka topi', 'doko', 'gagri', 'karuwa', 'khukuri']

    while True:
        success, img = cap.read()
        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f'{class_name} {conf:.2f}'

                if class_name == 'dhaka topi':
                    color = (0, 204, 255)
                elif class_name == "doko":
                    color = (222, 82, 175)
                elif class_name == "khukuri":
                    color = (0, 149, 255)
                elif class_name == "karuwa":
                    color = (3, 252, 3)
                else:
                    color = (85, 45, 255)

                if conf > 0.5:

                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 6)


                    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)[0]
                    cv2.rectangle(img, (x1, y1), (x1 + t_size[0], y1 - t_size[1] - 20), color, -1)


                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255], 3)

        yield img

cv2.destroyAllWindows()
