import os
import cv2
from tracking import Tracker
from analytics import Analytics
from constants import DEFAULT_SIDE_LINE, DEFAULT_BASE_LINE

def select_points_from_frame(frame):
    selected_points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Point selected: ({x}, {y})")
            selected_points.append((x, y))
            cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Select Court Points", param)

    temp_frame = frame.copy()
    cv2.imshow("Select Court Points", temp_frame)
    cv2.setMouseCallback("Select Court Points", click_event, temp_frame)

    print("Click on the four corners of the court in the following order:")
    print("Top-left, Top-right, Bottom-right, Bottom-left")
    while len(selected_points) < 4:
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    return selected_points

def estimate_court_dimensions(corners):
    # Estimate dimensions from pixel distances (if calibration is known, update here)
    pixel_width = ((corners[0][0] - corners[1][0]) ** 2 + (corners[0][1] - corners[1][1]) ** 2) ** 0.5
    pixel_height = ((corners[0][0] - corners[3][0]) ** 2 + (corners[0][1] - corners[3][1]) ** 2) ** 0.5

    print(f"Detected court size in pixels: width={pixel_width:.2f}, height={pixel_height:.2f}")

    # Currently return default values (you can adapt with actual calibration)
    return (DEFAULT_SIDE_LINE, DEFAULT_BASE_LINE)

def main():
    input_video = "data/rally.mp4"
    output_video = "output/tracked_rally.mp4"
    os.makedirs("output", exist_ok=True)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {input_video}")

    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Could not read a frame from the video.")

    print("Select court corners manually...")
    court_corners = select_points_from_frame(frame)
    print(f"Court corners selected: {court_corners}")

    # Estimate or use default court dimensions
    court_dimensions = estimate_court_dimensions(court_corners)

    print("Initializing tracker...")
    tracker = Tracker(court_corners, court_dimensions)

    print("Tracking players and ball...")
    tracking_data = tracker.track_players_and_ball(input_video, output_video)

    print("Analyzing tracking data...")
    analytics = Analytics(court_corners, court_dimensions)
    for frame_data in tracking_data:
        analytics.add_tracking_data(
            frame_number=frame_data["frame"],
            players=frame_data["players"],
            ball=frame_data["ball"]
        )

    summary = analytics.generate_summary()
    print("Analytics Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()