import torch
import torchvision.models as models
from ultralytics import YOLO

def load_yolo_model(model_name="yolov8n"):

    try:
        model = YOLO(model_name)
        print(f"Successfully loaded YOLO model: {model_name}")
        return model
    except Exception as e:
        print(f"Error loading YOLO model {model_name}: {e}")
        return None

def load_pose_model(pretrained=True):

    model = models.resnet50(pretrained=pretrained)
    model.fc = torch.nn.Linear(model.fc.in_features, 26) 
    return model

def load_tracknet_model(model_path=None):

    if model_path:
        try:
            model = torch.load(model_path)
            print(f"Loaded custom TrackNet model from {model_path}")
        except Exception as e:
            print(f"Error loading TrackNet model: {e}")
            model = None
    return model

def load_models(yolo_path="yolov8n.pt", tracknet_path=None, pose_pretrained=True):

    models = {
        "yolo_model": load_yolo_model(yolo_path),
        "tracknet_model": load_tracknet_model(tracknet_path),
        "pose_model": load_pose_model(pretrained=pose_pretrained),
    }
    return models