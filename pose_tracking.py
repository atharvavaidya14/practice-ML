import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
cap = cv2.VideoCapture("violin.mp4")  # Replace with your video
if not cap.isOpened():
    raise ValueError("Could not open video file.")

# Target resolution (1080x720)
TARGET_WIDTH = 1080
TARGET_HEIGHT = 720

orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
scale = min(TARGET_WIDTH / orig_width, TARGET_HEIGHT / orig_height)
new_width = int(orig_width * scale)
new_height = int(orig_height * scale)
pad_x = (TARGET_WIDTH - new_width) // 2
pad_y = (TARGET_HEIGHT - new_height) // 2
wrist_traj = []
elbow_traj = []
shoulder_traj = []
smooth_window = 5

output_path = "output_resized_trajectory.mp4"
out = cv2.VideoWriter(
    output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (TARGET_WIDTH, TARGET_HEIGHT)
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame (maintain aspect ratio + padding)
    resized_frame = cv2.resize(frame, (new_width, new_height))
    padded_frame = cv2.copyMakeBorder(
        resized_frame,
        pad_y,
        TARGET_HEIGHT - new_height - pad_y,
        pad_x,
        TARGET_WIDTH - new_width - pad_x,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),  # Black padding
    )

    # Process with MediaPipe (on padded frame)
    rgb_frame = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        # Get right wrist_r coordinates (denormalize to padded frame)
        wrist_r = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        elbow_r = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        shoulder_r = results.pose_landmarks.landmark[
            mp_pose.PoseLandmark.RIGHT_SHOULDER
        ]

        cx, cy = int(wrist_r.x * TARGET_WIDTH), int(wrist_r.y * TARGET_HEIGHT)
        wrist_traj.append((cx, cy))
        cv2.circle(padded_frame, (cx, cy), 8, (0, 0, 255), -1)
        cx, cy = int(elbow_r.x * TARGET_WIDTH), int(elbow_r.y * TARGET_HEIGHT)
        elbow_traj.append((cx, cy))
        cv2.circle(padded_frame, (cx, cy), 8, (0, 0, 255), -1)
        cx, cy = int(shoulder_r.x * TARGET_WIDTH), int(shoulder_r.y * TARGET_HEIGHT)
        shoulder_traj.append((cx, cy))
        cv2.circle(padded_frame, (cx, cy), 8, (0, 0, 255), -1)

        # Draw wrist_traj (last 50 points)
        for i in range(1, min(len(wrist_traj), 50)):
            cv2.line(padded_frame, wrist_traj[-i], wrist_traj[-i - 1], (0, 255, 0), 2)
        for i in range(1, min(len(elbow_traj), 50)):
            cv2.line(padded_frame, elbow_traj[-i], elbow_traj[-i - 1], (0, 255, 0), 2)
        for i in range(1, min(len(shoulder_traj), 50)):
            cv2.line(
                padded_frame, shoulder_traj[-i], shoulder_traj[-i - 1], (0, 255, 0), 2
            )

    # Display
    cv2.imshow("Resized (1080x720) + Trajectory (Press q to exit)", padded_frame)
    out.write(padded_frame)  # Optional: Save output

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
