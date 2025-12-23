import cv2
import os
import csv
from ultralytics import YOLO

INPUT_FOLDER = "../data/raw_videos/raw_videos/Backhand Chop"
OUTPUT_FOLDER = "../data/racket_npy/Backhand Chop"
# YOLO weights
MODEL_PATH = "/runs/weights/best.pt"

def process_video(video_path, output_csv_path, model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"can't open video: {video_path}")
        return

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame", "x1", "y1", "x2", "y2", "cx", "cy"])

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=0.2, verbose=False)

            
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    writer.writerow([frame_idx, int(x1), int(y1), int(x2), int(y2), cx, cy])
            else:
                writer.writerow([frame_idx, 0, 0, 0, 0, 0, 0])

            frame_idx += 1

    cap.release()
    print(f"Processing completed: {output_csv_path}")

def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    model = YOLO(MODEL_PATH)

    for file_name in os.listdir(INPUT_FOLDER):
        if file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(INPUT_FOLDER, file_name)
            base_name = os.path.splitext(file_name)[0]
            output_csv_path = os.path.join(OUTPUT_FOLDER, base_name + ".csv")
            process_video(video_path, output_csv_path, model)

if __name__ == "__main__":
    main()
