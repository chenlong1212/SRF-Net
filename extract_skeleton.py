import cv2
import mediapipe as mp
import os
import numpy as np
from tqdm import tqdm

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5
)

def extract_keypoints_from_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    keypoints = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 转换为 RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            frame_keypoints = []
            for lm in results.pose_landmarks.landmark:
                frame_keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
            keypoints.append(frame_keypoints)
        else:
            keypoints.append([0] * (33 * 4))

    cap.release()
    keypoints = np.array(keypoints)
    np.save(output_path, keypoints)

def process_dataset(input_root, output_root):
    video_files = []
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.endswith(".mp4"):
                video_files.append(os.path.join(root, file))

    for video_path in tqdm(video_files, desc="Processing videos", unit="file"):
        rel_path = os.path.relpath(os.path.dirname(video_path), input_root)
        save_dir = os.path.join(output_root, rel_path)
        os.makedirs(save_dir, exist_ok=True)

        output_file = os.path.join(save_dir, os.path.basename(video_path).replace(".mp4", ".npy"))
        extract_keypoints_from_video(video_path, output_file)

if __name__ == "__main__":
    input_root = "data/raw_videos/Backhand Chop"     # 你的视频数据集目录
    output_root = "data/skeleton_npy/Backhand Chop"  # 输出目录
    process_dataset(input_root, output_root)
