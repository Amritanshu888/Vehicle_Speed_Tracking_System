### Method 1: Copy or Move After Training (Recommended)
import shutil
from ultralytics import YOLO

# 1. Load a pretrained model and train it
model = YOLO("yolov8n.pt")
results = model.train(data="coco8.yaml", epochs=10)

# 2. Locate the best weights from the training run
best_model_path = f"{results.save_dir}/weights/best.pt"

# 3. Copy and rename it directly to your project directory
shutil.copy(best_model_path, "./yolov8.pt")

### Method 2: Set Your Project Directory During Training
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Saves weights to ./my_project/weights_folder/weights/best.pt
model.train(
    data="coco8.yaml",
    epochs=10,
    project="./my_project",
    name="weights_folder",
)

## Method 3: Download Pretrained Weights Directly
from ultralytics import YOLO

# Automatically downloads 'yolov8n.pt' to your current directory
model = YOLO("yolov8n.pt")
