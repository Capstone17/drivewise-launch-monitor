import cv2
from ultralytics import YOLO

def main():
    model = YOLO('golf_ball_detector.onnx')
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print('Could not open webcam')
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow('Webcam Golf Ball Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
