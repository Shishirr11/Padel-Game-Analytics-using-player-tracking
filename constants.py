
# Court dimensions in meters (standard singles badminton court)
DEFAULT_SIDE_LINE = 5.18     # meters
DEFAULT_BASE_LINE = 13.4     # Length of court (front to back)

SERVICE_SIDE_LINE = 3
NET_SIDE_LINE = 10


FRAME_RATE = 25  
TRACKING_CONFIDENCE = 0.5
POSE_CONFIDENCE = 0.5


HEATMAP_WIDTH = 1920
HEATMAP_HEIGHT = 1080


YOLO_MODEL_PATH = "yolov8n.pt"  
BALL_TRACKING_MODEL = "tracknet_model_best.pt"     


BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10