import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import Tuple


def skip_keypoint(x: float, y: float, conf: float, threshold: float) -> bool:
    """Return True if keypoint is invalid (low confidence or near (0,0))."""
    if conf < threshold:
        return True
    return abs(x) < 1 and abs(y) < 1


def main(
    video_source: str,
    bbox_model_path: str,
    pose_model_path: str,
    output_video: str,
    alpha: float,
    pose_conf_threshold: float,
    dilation_kernel_size: int,
    dilation_iterations: int,
    blur_kernel_size: int,
):
    """Main function to run person anonymization using bounding boxes."""
    # Load YOLO models
    model_box = YOLO(bbox_model_path)
    model_pose = YOLO(pose_model_path)

    # Open video source
    try:
        source_int = int(video_source)
        cap = cv2.VideoCapture(source_int)
    except ValueError:
        cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video source: {video_source}")

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup video writer
    Path(output_video).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height), True)

    background_est = None
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9),
        (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13),
        (13, 15), (12, 14), (14, 16),
    ]
    
    dilation_kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_float = frame.astype(np.float32)

        if background_est is None:
            background_est = frame_float.copy()

        # Step 1: Bounding Box Detection
        results_box = model_box(frame, verbose=False)
        r = results_box[0]

        bbox_mask = np.zeros((height, width), dtype=np.uint8)
        if r.boxes:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            for i, cls in enumerate(classes):
                if int(cls) == 0:  # Person class
                    x1, y1, x2, y2 = map(int, boxes[i])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    bbox_mask[y1:y2, x1:x2] = 1

        # Step 2: Freeze Background Update for Foreground
        bg_update_mask = 1 - bbox_mask
        bg_update_mask_3ch = cv2.merge([bg_update_mask] * 3).astype(np.float32)
        background_est += alpha * (frame_float - background_est) * bg_update_mask_3ch
        current_background = background_est.astype(np.uint8)

        # Step 3: Create a Soft Mask
        bbox_mask_dilated = cv2.dilate(bbox_mask, dilation_kernel, iterations=dilation_iterations)
        mask_float = bbox_mask_dilated.astype(np.float32)
        soft_mask = cv2.GaussianBlur(
            mask_float, (blur_kernel_size, blur_kernel_size), 0
        )
        soft_mask = np.clip(soft_mask, 0, 1)
        soft_mask_3ch = cv2.merge([soft_mask] * 3)

        # Step 4: Alpha-Blend to Anonymize
        out_float = (soft_mask_3ch * current_background.astype(np.float32) +
                     (1 - soft_mask_3ch) * frame_float)
        anonymized_frame = out_float.astype(np.uint8)

        # Step 5: Pose Estimation & Skeleton Drawing
        results_pose = model_pose(frame, verbose=False)
        rp = results_pose[0]
        if rp.keypoints and rp.keypoints.xy is not None:
            keypoints_xy = rp.keypoints.xy.cpu().numpy()
            keypoints_conf = rp.keypoints.conf.cpu().numpy()

            for instance_xy, instance_conf in zip(keypoints_xy, keypoints_conf):
                for k1, k2 in skeleton:
                    if k1 < len(instance_xy) and k2 < len(instance_xy):
                        x1, y1 = instance_xy[k1]
                        x2, y2 = instance_xy[k2]
                        conf1, conf2 = instance_conf[k1], instance_conf[k2]
                        if not skip_keypoint(x1, y1, conf1, pose_conf_threshold) and \
                           not skip_keypoint(x2, y2, conf2, pose_conf_threshold):
                            cv2.line(
                                anonymized_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                            )
                for (x, y), conf in zip(instance_xy, instance_conf):
                    if not skip_keypoint(x, y, conf, pose_conf_threshold):
                        cv2.circle(anonymized_frame, (int(x), int(y)), 4, (0, 255, 0), -1)

        # Step 6: Display and Save
        combined_display = np.hstack((frame, anonymized_frame))
        cv2.imshow("Original (left) vs. Anonymized (right)", combined_display)
        out.write(anonymized_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anonymize persons in a video using bounding boxes.")
    parser.add_argument("--video-source", type=str, required=True, help="Path to video file or '0' for webcam.")
    parser.add_argument("--bbox-model", type=str, default="models/yolo11n.pt", help="Path to YOLO bounding box model.")
    parser.add_argument("--pose-model", type=str, default="models/yolo11n-pose.pt", help="Path to YOLO pose model.")
    parser.add_argument("--output-video", type=str, default="output/anonymized_bbox.mp4", help="Path to save output video.")
    parser.add_argument("--alpha", type=float, default=0.2, help="Background adaptation speed (learning rate).")
    parser.add_argument("--pose-conf-threshold", type=float, default=0.2, help="Confidence threshold for pose keypoints.")
    parser.add_argument("--dilation-kernel-size", type=int, default=21, help="Kernel size for bounding box dilation.")
    parser.add_argument("--dilation-iterations", type=int, default=1, help="Number of iterations for dilation.")
    parser.add_argument("--blur-kernel-size", type=int, default=25, help="Gaussian blur kernel size for feathering edges.")
    args = parser.parse_args()

    # Ensure blur and dilation kernel sizes are odd
    if args.blur_kernel_size % 2 == 0:
        args.blur_kernel_size += 1
    if args.dilation_kernel_size % 2 == 0:
        args.dilation_kernel_size += 1

    main(
        args.video_source,
        args.bbox_model_path,
        args.pose_model_path,
        args.output_video,
        args.alpha,
        args.pose_conf_threshold,
        args.dilation_kernel_size,
        args.dilation_iterations,
        args.blur_kernel_size,
    )