import numpy as np
import cv2

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxBArea = (box2[2] - box2[0]) * (box2[3] - box2[1])
    unionArea = float(boxAArea + boxBArea - interArea)

    if unionArea == 0:
        return 0.0
    return interArea / unionArea

def assign_ids(current_boxes, prev_tracked, iou_threshold=0.3):
    assigned = []
    unmatched_ids = set(pid for pid, _ in prev_tracked)
    used_current = set()

    for pid, prev_box in prev_tracked:
        best_iou = 0
        best_idx = -1
        for i, curr_box in enumerate(current_boxes):
            if i in used_current:
                continue
            iou = compute_iou(prev_box, curr_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou >= iou_threshold and best_idx != -1:
            assigned.append((pid, current_boxes[best_idx]))
            used_current.add(best_idx)
            unmatched_ids.discard(pid)

    next_id = max([pid for pid, _ in prev_tracked], default=0) + 1
    for i, box in enumerate(current_boxes):
        if i not in used_current:
            assigned.append((next_id, box))
            next_id += 1

    return assigned

def draw_bounding_boxes(frame, player_boxes, ball_coords=None):
    for box in player_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        center_x = int((x1 + x2) / 2)
        cv2.putText(frame, "Player", (center_x, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if ball_coords:
        bx, by = map(int, ball_coords)
        cv2.circle(frame, (bx, by), 5, (0, 0, 255), -1)
        cv2.putText(frame, "Ball", (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame