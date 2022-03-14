import cv2
import time
import detect_lane_pipeline

video_path = "./videos/solidWhiteRight.mp4"

video = cv2.VideoCapture(video_path)
while video.isOpened():
    ret, frame = video.read()

    start_time = time.time()

    detected = detect_lane_pipeline.detect(frame)

    end_time = time.time()

    fps = 1/(end_time - start_time)
    fps = int(fps)
    fps = str(fps)
    text = "FPS: " + fps

    cv2.putText(detected, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("pythonDetected", detected)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

video.release()
cv2.destroyAllWindows()