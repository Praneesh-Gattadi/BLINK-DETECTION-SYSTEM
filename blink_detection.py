import cv2
import mediapipe as mp
import time
import math
import numpy as np
from collections import deque
from statistics import median

# ---------------- CONFIG ----------------
EAR_SMOOTHING_LEN = 5       # smoother EAR
CALIBRATION_TIME = 2.0      # seconds to auto-calibrate open-eye EAR
MIN_CALIB_SAMPLES = 10      # minimum samples to accept calibration

DEFAULT_EAR_CLOSE_FRAC = 0.65   # close = open_ear * frac
DEFAULT_EAR_OPEN_FRAC  = 0.85   # open  = open_ear * frac

MIN_BLINK_DUR = 0.06        # seconds
MAX_BLINK_DUR = 0.5         # seconds
MIN_INTER_BLINK = 0.15      # refractory period after a counted blink

STABILITY_MOVE_THRESH = 0.08   # fraction of face width
STABLE_REQUIRED_TIME = 0.18    # seconds required stable before counting

# Mediapipe indices for eye EAR (Face Mesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# ----------------------------------------

# Initialize Mediapipe solution references (we create instances later in the `with` block)
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

def calculate_EAR_from_tuples(eye_points, landmarks):
    # landmarks is list of (x_px, y_px)
    # This function computes the Eye Aspect Ratio (EAR) for a given eye using six landmark points:
    # eye_points: indices for the eye landmarks (6 points). landmarks: list of (x, y) tuples in pixel coords.
    # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    # Verticals are averaged and normalized by horizontal distance.
    try:
        p1 = landmarks[eye_points[0]]
        p2 = landmarks[eye_points[1]]
        p3 = landmarks[eye_points[2]]
        p4 = landmarks[eye_points[3]]
        p5 = landmarks[eye_points[4]]
        p6 = landmarks[eye_points[5]]
    except IndexError:
        # If any landmark is missing (index out of range) return 0.0 to indicate invalid EAR
        return 0.0
    vertical1 = math.dist(p2, p6)   # vertical distance pair 1
    vertical2 = math.dist(p3, p5)   # vertical distance pair 2
    horizontal = math.dist(p1, p4)  # horizontal distance
    if horizontal == 0:
        # Avoid division by zero if horizontal distance is zero
        return 0.0
    return (vertical1 + vertical2) / (2.0 * horizontal)

def compute_eye_center(eye_points, landmarks):
    # Compute the geometric center (average x, average y) of the provided eye landmark indices.
    # This helps with simple yaw checks and drawing.
    xs = [landmarks[i][0] for i in eye_points if i < len(landmarks)]
    ys = [landmarks[i][1] for i in eye_points if i < len(landmarks)]
    if not xs or not ys:
        return None
    return (sum(xs) / len(xs), sum(ys) / len(ys))

# State variables used during runtime
blink_count = 0
ear_history = deque(maxlen=EAR_SMOOTHING_LEN)  # keeps last N EAR values for smoothing

closed_state = False          # True when we are currently tracking a closed-eye period
closed_start_time = None      # timestamp when eyes were detected closed
closed_max_motion = 0.0       # track maximum head motion while eyes were closed

prev_face_center = None       # previous frame face center (cx, cy)
prev_face_width = None        # previous frame face width (in pixels)
stable_since = None           # timestamp when face became "stable"

last_blink_time = 0.0         # timestamp of last counted blink (for refractory)

# Dynamic thresholds (will be set by calibration)
EAR_CLOSE_THRESH = None
EAR_OPEN_THRESH  = None

# Open the default camera (device 0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    raise SystemExit

# Create full-screen window for display
WIN_NAME = "Highly-Accurate Blink Detection"
cv2.namedWindow(WIN_NAME, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Force a refresh so fullscreen shows immediately (show a blank frame momentarily)
blank = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.imshow(WIN_NAME, blank)
cv2.waitKey(1)

# Use Mediapipe face detection + face mesh inside a context manager (will auto-close properly)
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
     mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

    # --- Calibration phase ---
    # We collect EAR samples while the user keeps eyes open & steady for CALIBRATION_TIME seconds.
    # The median of those samples is used to compute open/close thresholds.
    print("Calibration: keep your face centered and eyes open for ~2 seconds...")
    calib_samples = []
    calib_start = time.time()
    while time.time() - calib_start < CALIBRATION_TIME:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        det_results = face_detection.process(rgb)
        if det_results.detections:
            d = det_results.detections[0]
            bbox = d.location_data.relative_bounding_box
            x_px = int(bbox.xmin * w)
            y_px = int(bbox.ymin * h)
            width_px = int(bbox.width * w)
            height_px = int(bbox.height * h)
            cx = x_px + width_px / 2.0
            cy = y_px + height_px / 2.0

            # calculate motion vs previous frame (small helper for stability)
            movement_ratio = 0.0
            if prev_face_center is not None and prev_face_width is not None:
                dx = cx - prev_face_center[0]
                dy = cy - prev_face_center[1]
                move_px = math.hypot(dx, dy)
                movement_ratio = move_px / prev_face_width
                scale_change = abs(width_px - prev_face_width) / prev_face_width
                movement_ratio = max(movement_ratio, scale_change)

            # If face mesh landmarks are available and movement is small, compute EAR samples
            mesh_results = face_mesh.process(rgb)
            if mesh_results.multi_face_landmarks and movement_ratio <= STABILITY_MOVE_THRESH:
                mesh = mesh_results.multi_face_landmarks[0]
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in mesh.landmark]
                left_ear = calculate_EAR_from_tuples(LEFT_EYE, landmarks)
                right_ear = calculate_EAR_from_tuples(RIGHT_EYE, landmarks)
                avg_ear = (left_ear + right_ear) / 2.0
                if avg_ear > 0:
                    # Collect median-friendly samples only when EAR computed successfully
                    calib_samples.append(avg_ear)

            # store previous face center/size for next frame movement calculation
            prev_face_center = (cx, cy)
            prev_face_width = max(1, width_px)

            # draw guidance rectangle & message during calibration
            cv2.rectangle(frame, (x_px, y_px), (x_px + width_px, y_px + height_px), (0, 255, 0), 1)
            cv2.putText(frame, "Calibrating... keep eyes open & steady", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            # Face not found during calibration
            cv2.putText(frame, "Calibration: face not found", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow(WIN_NAME, frame)
        # Allow user to abort calibration with 'q' or Esc
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            print("Calibration aborted by user.")
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit

    # Decide thresholds using collected calibration samples if enough were gathered
    if len(calib_samples) >= MIN_CALIB_SAMPLES:
        open_ear_est = median(calib_samples)  # robust central tendency for open-eye EAR
        EAR_CLOSE_THRESH = open_ear_est * DEFAULT_EAR_CLOSE_FRAC
        EAR_OPEN_THRESH  = open_ear_est * DEFAULT_EAR_OPEN_FRAC
        print(f"Calibration done. open_ear≈{open_ear_est:.3f} -> close={EAR_CLOSE_THRESH:.3f}, open={EAR_OPEN_THRESH:.3f}")
    else:
        # fallback to reasonable defaults if calibration failed or insufficient samples
        EAR_CLOSE_THRESH = 0.20
        EAR_OPEN_THRESH  = 0.27
        print("Calibration insufficient — using default thresholds.")

    # Reset some state for main loop
    ear_history.clear()
    prev_face_center = None
    prev_face_width = None
    stable_since = None
    closed_state = False
    closed_start_time = None
    closed_max_motion = 0.0

    print("▶ Running. Press 'q' or Esc to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face detection for bbox / stability
        det_results = face_detection.process(rgb)

        face_present = False
        current_face_center = None
        current_face_width = None
        stable = False
        movement_ratio = 0.0
        yaw_ok = True  # simple yaw check

        if det_results.detections:
            # Extract bounding box & compute center/size in pixels
            d = det_results.detections[0]
            bbox = d.location_data.relative_bounding_box
            x_px = int(bbox.xmin * w)
            y_px = int(bbox.ymin * h)
            width_px = int(bbox.width * w)
            height_px = int(bbox.height * h)
            cx = x_px + width_px / 2.0
            cy = y_px + height_px / 2.0

            current_face_center = (cx, cy)
            current_face_width = max(1, width_px)
            face_present = True

            # movement vs previous frame (translation + scale change)
            if prev_face_center is not None and prev_face_width is not None:
                dx = cx - prev_face_center[0]
                dy = cy - prev_face_center[1]
                move_px = math.hypot(dx, dy)
                movement_ratio = move_px / prev_face_width
                scale_change = abs(current_face_width - prev_face_width) / prev_face_width
                movement_ratio = max(movement_ratio, scale_change)
            else:
                movement_ratio = 0.0

            # stability logic: require small movement for a short time to consider face "stable"
            if movement_ratio <= STABILITY_MOVE_THRESH:
                if stable_since is None:
                    stable_since = time.time()
                stable = (time.time() - stable_since) >= STABLE_REQUIRED_TIME
            else:
                stable_since = None
                stable = False

            # draw bounding box and stability status on frame
            color = (0, 200, 0) if stable else (0, 165, 255)
            cv2.rectangle(frame, (x_px, y_px), (x_px + width_px, y_px + height_px), color, 2)
            cv2.putText(frame, f"Stable: {stable}", (x_px, y_px + height_px + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            # No face detected — show a large red rectangle and reset stability
            cv2.rectangle(frame, (30, 30), (w - 30, int(h * 0.7)), (0, 0, 255), 2)
            cv2.putText(frame, "No Face Detected", (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            stable_since = None
            stable = False
            movement_ratio = 1.0

        # Landmark-based EAR only when face present
        if face_present:
            mesh_results = face_mesh.process(rgb)
            if mesh_results.multi_face_landmarks:
                mesh = mesh_results.multi_face_landmarks[0]
                # Convert normalized mesh landmarks to pixel coordinates
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in mesh.landmark]

                # Draw eye landmarks (optional visual aid)
                for idx in LEFT_EYE + RIGHT_EYE:
                    if idx < len(landmarks):
                        cv2.circle(frame, landmarks[idx], 2, (200, 100, 0), -1)

                # Compute EAR for left/right and average
                left_ear = calculate_EAR_from_tuples(LEFT_EYE, landmarks)
                right_ear = calculate_EAR_from_tuples(RIGHT_EYE, landmarks)
                avg_ear = (left_ear + right_ear) / 2.0

                # yaw check (simple): if eye centers are shifted too much relative to face center -> skip
                left_center = compute_eye_center(LEFT_EYE, landmarks)
                right_center = compute_eye_center(RIGHT_EYE, landmarks)
                if left_center and right_center and current_face_center:
                    mid_x = (left_center[0] + right_center[0]) / 2.0
                    yaw_ratio = abs((mid_x - current_face_center[0]) / current_face_width)
                    # if yaw_ratio is large, user turned head -> mark yaw_ok False
                    if yaw_ratio > 0.12:
                        yaw_ok = False
                    cv2.putText(frame, f"YawOK: {yaw_ok}", (30, 170),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                # smooth EAR using recent history to reduce frame-to-frame jitter
                ear_history.append(avg_ear)
                smooth_ear = sum(ear_history) / len(ear_history)

                # display left, right and smoothed EAR on the frame for debugging
                cv2.putText(frame, f"L:{left_ear:.2f} R:{right_ear:.2f} S:{smooth_ear:.2f}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Blink logic only if face stable & yaw okay
                if stable and yaw_ok:
                    # Update max motion while closed (so we can reject blinks that happen during large movement)
                    if closed_state:
                        if movement_ratio > closed_max_motion:
                            closed_max_motion = movement_ratio

                    # Condition for both eyes closed. If one eye missing, fallback to avg.
                    both_closed = (left_ear < EAR_CLOSE_THRESH) and (right_ear < EAR_CLOSE_THRESH)
                    at_least_avg_closed = (smooth_ear < EAR_CLOSE_THRESH)

                    # open -> closed transition: mark start time & initial motion
                    if (not closed_state) and (both_closed or at_least_avg_closed):
                        closed_state = True
                        closed_start_time = time.time()
                        closed_max_motion = movement_ratio

                    # closed -> open transition (candidate blink)
                    elif closed_state and (smooth_ear > EAR_OPEN_THRESH):
                        if closed_start_time is not None:
                            duration = time.time() - closed_start_time
                            # check timing, motion during closure and refractory
                            if (MIN_BLINK_DUR <= duration <= MAX_BLINK_DUR) and (closed_max_motion <= STABILITY_MOVE_THRESH):
                                if (time.time() - last_blink_time) >= MIN_INTER_BLINK:
                                    blink_count += 1
                                    last_blink_time = time.time()
                                    print(f"Blink #{blink_count} (dur {duration:.3f}s, max_move {closed_max_motion:.3f})")
                        # reset closure tracking after processing candidate blink
                        closed_state = False
                        closed_start_time = None
                        closed_max_motion = 0.0
                else:
                    # unstable or head turned — reset closed tracking to avoid false positives
                    closed_state = False
                    closed_start_time = None
                    closed_max_motion = 0.0

            else:
                # no mesh landmarks — reset and avoid counting
                closed_state = False
                closed_start_time = None
                closed_max_motion = 0.0
        else:
            # no face present -> reset everything that depends on the face being tracked
            ear_history.clear()
            closed_state = False
            closed_start_time = None
            closed_max_motion = 0.0
            prev_face_center = None
            prev_face_width = None

        # Draw overall blink count & debug info on the frame
        cv2.putText(frame, f"Blink Count: {blink_count}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, f"Move: {movement_ratio:.3f}", (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(WIN_NAME, frame)

        # update previous face center/size for next frame movement calculation
        if face_present and current_face_center is not None:
            prev_face_center = current_face_center
            prev_face_width = current_face_width

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or Esc
            break

cap.release()
cv2.destroyAllWindows()
