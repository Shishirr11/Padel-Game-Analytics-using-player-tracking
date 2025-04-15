import cv2
import torch
import numpy as np
from ultralytics import YOLO
from tracknet_architecture import BallTrackerNet
from projection import CourtProjection
from utils import draw_bounding_boxes, assign_ids

class Tracker:
    def __init__(self, court_corners, court_dimensions):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.player_model = YOLO("yolov8n.pt")
        self.player_model.fuse()

        self.ball_model = BallTrackerNet(out_channels=256).to(self.device)
        self.ball_model.load_state_dict(torch.load("tracknet_model_best.pt", map_location=self.device))
        self.ball_model.eval()

        self.projection = CourtProjection(court_corners, court_dimensions)

        self.previous_players = []
        self.prev_ball_coords = None

        self.player_history = {}  # {player_id: [coords]}
        self.ball_history = []    # [coords]
        self.smoothing_window = 5

    def preprocess_frames_for_ball(self, frames):
        processed = []
        for frame in frames:
            resized = cv2.resize(frame, (640, 360))
            normalized = resized / 255.0
            tensor = torch.tensor(normalized, dtype=torch.float32).permute(2, 0, 1)
            processed.append(tensor)
        return torch.cat(processed, dim=0).unsqueeze(0).to(self.device)

    def track_ball(self, frame_buffer):
        if len(frame_buffer) < 3:
            return None
        input_tensor = self.preprocess_frames_for_ball(frame_buffer)
        with torch.no_grad():
            heatmap = self.ball_model(input_tensor).squeeze().cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        if heatmap.max() < 0.5:
            return None
        max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        x = max_idx[1] * (frame_buffer[-1].shape[1] / heatmap.shape[1])
        y = max_idx[0] * (frame_buffer[-1].shape[0] / heatmap.shape[0])
        return (x, y)

    def track_players_and_ball(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        tracking_results = []
        frame_buffer = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            frame_buffer.append(frame)
            if len(frame_buffer) > 3:
                frame_buffer.pop(0)

            # --- Player Detection ---
            player_results = self.player_model(frame)[0]
            player_boxes = [
                box.xyxy[0].cpu().numpy().tolist()
                for box in player_results.boxes
                if int(box.cls) == 0
            ]

            # Assign consistent IDs
            tracked_players = assign_ids(player_boxes, self.previous_players)
            self.previous_players = tracked_players

            # --- Ball Tracking ---
            ball_coords = self.track_ball(frame_buffer)

            # --- Projection ---
            player_centers = [(int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)) for _, b in tracked_players]
            court_player_coords = self.projection.project_to_court(player_centers)
            court_ball_coords = self.projection.project_to_court([ball_coords]) if ball_coords else None

            # --- Smoothing: Players ---
            smoothed_players = []
            for (pid, _), coord in zip(tracked_players, court_player_coords):
                hist = self.player_history.get(pid, [])
                hist.append(coord)
                if len(hist) > self.smoothing_window:
                    hist.pop(0)
                self.player_history[pid] = hist
                avg_coord = tuple(np.mean(hist, axis=0))
                smoothed_players.append({"id": pid, "coords": avg_coord})

            # --- Smoothing: Ball ---
            if court_ball_coords:
                self.ball_history.append(court_ball_coords[0])
                if len(self.ball_history) > self.smoothing_window:
                    self.ball_history.pop(0)
                smoothed_ball = tuple(np.mean(self.ball_history, axis=0))
            else:
                smoothed_ball = None

            # --- Save frame data ---
            frame_data = {
                "frame": frame_idx,
                "players": smoothed_players,
                "ball": smoothed_ball
            }
            tracking_results.append(frame_data)

            # --- Draw output frame ---
            vis_boxes = [box for _, box in tracked_players]
            annotated = draw_bounding_boxes(frame.copy(), vis_boxes, ball_coords)
            out.write(annotated)

        cap.release()
        out.release()
        return tracking_results