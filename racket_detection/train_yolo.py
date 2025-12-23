from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  

    model.train(
        data="yolo_data/pingpang.yaml", 
        epochs=50,
        imgsz=640,
        batch=8,
        device="0",  
        project="runs/train",
        name="pingpang_yolo",
        exist_ok=True
    )

if __name__ == "__main__":
    main()
