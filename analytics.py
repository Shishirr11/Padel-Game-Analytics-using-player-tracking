from utils import calculate_distance, calculate_velocity

class Analytics:
    def __init__(self, court_corners, court_dimensions):
        self.tracking_data = []
        self.frame_rate = 30  # Default, can be updated if known
        self.player_distances = {}  # {player_id: total_distance}
        self.ball_distance = 0
        self.prev_player_positions = {}
        self.prev_ball_position = None

    def add_tracking_data(self, frame_number, players, ball):
        self.tracking_data.append({
            "frame": frame_number,
            "players": players,
            "ball": ball
        })

        # Track player distances
        for player in players:
            pid = player["id"]
            current_pos = player["coords"]

            if pid in self.prev_player_positions:
                dist = calculate_distance(self.prev_player_positions[pid], current_pos)
                self.player_distances[pid] = self.player_distances.get(pid, 0) + dist

            self.prev_player_positions[pid] = current_pos

        # Track ball distance
        if ball is not None:
            if self.prev_ball_position is not None:
                self.ball_distance += calculate_distance(self.prev_ball_position, ball)
            self.prev_ball_position = ball

    def generate_summary(self):
        summary = {}

        # Player stats
        for pid, total_distance in self.player_distances.items():
            avg_speed = total_distance * self.frame_rate / len(self.tracking_data)
            summary[f"Player {pid} - Distance (m)"] = round(total_distance, 2)
            summary[f"Player {pid} - Avg Speed (m/s)"] = round(avg_speed, 2)

        # Ball stats
        summary["Ball - Distance Travelled (m)"] = round(self.ball_distance, 2)
        ball_speed = self.ball_distance * self.frame_rate / len(self.tracking_data)
        summary["Ball - Avg Speed (m/s)"] = round(ball_speed, 2)

        return summary