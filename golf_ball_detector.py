import cv2
from ultralytics import YOLO

def main():
    model = YOLO('golf_ball_detector.onnx')


    # Scan for a working camera index
    def find_working_camera(max_index=5):
        for idx in range(max_index):
            for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]:
                if backend is not None:
                    cap = cv2.VideoCapture(idx, backend)
                else:
                    cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    # Try to grab a frame
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    for _ in range(5):
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.mean() > 5:
                            print(f"Camera found at index {idx} with backend {backend if backend is not None else 'default'}.")
                            return cap
                    cap.release()
        print("No working camera found. Tried all indices and backends.")
        return None

    cap = find_working_camera(5)
    if cap is None or not cap.isOpened():
        print('Could not open any webcam. Try closing other apps that use the camera, check permissions, or try a different camera.')
        return


    # Test: Try to grab a frame and check if it's black, fallback to lower resolution if needed
    ret, frame = cap.read()
    if not ret or frame is None:
        print('Failed to grab initial frame from webcam.')
        cap.release()
        cv2.destroyAllWindows()
        return
    if frame.mean() < 5:
        print('First frame is all black, trying lower resolution (640x480)...')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None and frame.mean() > 5:
                print('Successfully got a non-black frame at 640x480.')
                break
        else:
            print('Still getting black frames. Please check your camera index, close other apps, or try a different webcam.')
            cap.release()
            cv2.destroyAllWindows()
            return

    # Show the first frame for confirmation
    cv2.imshow('Webcam Test Frame', frame)
    print(f'Frame shape: {frame.shape}, mean pixel value: {frame.mean():.2f}')
    cv2.waitKey(1000)  # Show for 1 second
    cv2.destroyWindow('Webcam Test Frame')

    # Main detection loop
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print('Failed to grab frame from webcam.')
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
