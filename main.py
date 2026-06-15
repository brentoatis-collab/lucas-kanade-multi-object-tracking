import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

video_path = "videos/ucf101_sample.avi"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

ret, old_frame = cap.read()

if not ret:
    print("Could not read first frame.")
    exit()

height, width = old_frame.shape[:2]

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

feature_params = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

p0 = cv2.goodFeaturesToTrack(
    old_gray,
    mask=None,
    **feature_params
)

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        10,
        0.03
    )
)

trajectory_mask = np.zeros_like(old_frame)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    "outputs/tracked_output.mp4",
    fourcc,
    20.0,
    (width, height)
)

object_paths = defaultdict(list)

frame_count = 0
max_frames = 250

while cap.isOpened() and frame_count < max_frames:

    ret, frame = cap.read()

    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(
        old_gray,
        frame_gray,
        p0,
        None,
        **lk_params
    )

    if p1 is None:
        break

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    motion_points = []

    for new, old in zip(good_new, good_old):

        a, b = new.ravel()
        c, d = old.ravel()

        displacement = np.sqrt((a - c) ** 2 + (b - d) ** 2)

        if displacement > 1.5:
            motion_points.append((int(a), int(b)))

            trajectory_mask = cv2.line(
                trajectory_mask,
                (int(a), int(b)),
                (int(c), int(d)),
                (0, 255, 0),
                2
            )

            frame = cv2.circle(
                frame,
                (int(a), int(b)),
                4,
                (0, 0, 255),
                -1
            )

    motion_mask = np.zeros(
        frame_gray.shape,
        dtype=np.uint8
    )

    for x, y in motion_points:
        cv2.circle(
            motion_mask,
            (x, y),
            8,
            255,
            -1
        )

    kernel = np.ones((25, 25), np.uint8)

    motion_mask = cv2.dilate(
        motion_mask,
        kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        motion_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    object_id = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        if area > 250:

            x, y, w, h = cv2.boundingRect(contour)

            cx = x + w // 2
            cy = y + h // 2

            object_paths[object_id].append((cx, cy))

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Object {object_id}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            object_id += 1

    output = cv2.add(frame, trajectory_mask)

    cv2.putText(
        output,
        f"Frame: {frame_count}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    out.write(output)

    cv2.imshow(
        "Multi-Object Tracking with Optical Flow",
        output
    )

    if frame_count == 75:
        cv2.imwrite(
            "outputs/optical_flow_tracking.png",
            output
        )

    key = cv2.waitKey(30)

    if key == 27 or key == ord("q"):
        break

    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    if frame_count % 30 == 0 or len(p0) < 40:
        new_features = cv2.goodFeaturesToTrack(
            old_gray,
            mask=None,
            **feature_params
        )

        if new_features is not None:
            p0 = new_features

    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print("Tracking complete.")
print("Saved video: outputs/tracked_output.mp4")
print("Saved screenshot: outputs/optical_flow_tracking.png")

plt.figure(figsize=(10, 7))

for obj_id, points in object_paths.items():
    if len(points) > 1:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]

        plt.plot(
            xs,
            ys,
            marker="o",
            label=f"Object {obj_id}"
        )

plt.title("Multi-Object Trajectory Visualization")
plt.xlabel("X-axis Position in Frame")
plt.ylabel("Y-axis Position in Frame")
plt.gca().invert_yaxis()
plt.grid(True)
plt.legend()

plt.savefig(
    "outputs/trajectory_visualization.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved trajectory plot: outputs/trajectory_visualization.png")