import numpy as np
import cv2

class CourtProjection:
    def __init__(self, court_corners, court_dimensions):
        """
        court_corners: 4 points in image space (selected by user)
        court_dimensions: (width=SIDE_LINE, height=BASE_LINE) in meters
        """
        self.court_corners = np.array(court_corners, dtype=np.float32)

        self.court_dimensions = np.array([
            [0, 0],                              # top-left
            [court_dimensions[0], 0],           # top-right
            [court_dimensions[0], court_dimensions[1]],  # bottom-right
            [0, court_dimensions[1]]            # bottom-left
        ], dtype=np.float32)

        self.homography_matrix, _ = cv2.findHomography(self.court_corners, self.court_dimensions)

    def project_to_court(self, image_coords):
        if image_coords is None or len(image_coords) == 0:
            return []

        pts = np.array(image_coords, dtype=np.float32).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(pts, self.homography_matrix)
        return projected.reshape(-1, 2).tolist()

    def project_to_image(self, court_coords):
        if court_coords is None or len(court_coords) == 0:
            return []

        pts = np.array(court_coords, dtype=np.float32).reshape(-1, 1, 2)
        image_coords = cv2.perspectiveTransform(pts, np.linalg.inv(self.homography_matrix))
        return image_coords.reshape(-1, 2).tolist()