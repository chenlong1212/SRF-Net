from ultralytics import YOLO

def run_inference(video_path, output_path, model_path):

    model = YOLO(model_path)


    results = model.predict(
        source=video_path,    
        save=True,             
        save_txt=False,      
        project="runs/inference", 
        name="pingpang_video",    
        exist_ok=True,           
        device=0                   
    )


if __name__ == "__main__":
    video_path = "../data/raw_videos/Forehand Drive_1.mp4"   
    output_path = "runs/inference/pingpang_video"                    
    model_path = "runs/weights/best.pt"           
    run_inference(video_path, output_path, model_path)
