# Person Anonymization with Dynamic Background Reconstruction

## 📖 Overview

This project provides tools for real-time person anonymization in videos using computer vision techniques. It replaces detected persons with a dynamically generated background, effectively redacting them from the scene while preserving context. The system can use either bounding boxes or segmentation masks for person detection and overlays skeletal keypoints to maintain motion information.

This is ideal for applications requiring privacy protection, such as public video surveillance analysis, data collection for autonomous vehicles, or any domain where personal identity must be obscured.

### Key Features

- **Dynamic Background Modeling**: The background is continuously updated, adapting to changes like lighting shifts or minor object movements.
- **Two Anonymization Methods**:
  1.  **Bounding Box-Based**: Fast and efficient, suitable for real-time applications where precision is secondary.
  2.  **Segmentation-Based**: More precise, creating a clean anonymization effect by following the exact contours of a person.
- **Pose Estimation**: Overlays a skeletal structure on the anonymized area to retain crucial motion and activity information without revealing identity.
- **Soft Blending**: Uses Gaussian blurring on mask edges for a seamless, feathered blend between the person and the reconstructed background.

---

## 🖼️ Examples

Here's a comparison of the two anonymization methods. Segmentation provides a tighter, more accurate redaction.

| Bounding Box Anonymization                                          | Segmentation Anonymization                                         |
| ------------------------------------------------------------------- | ------------------------------------------------------------------ |
| <img src="https://i.imgur.com/your_bbox_image_url.gif" width="400"> | <img src="https://i.imgur.com/your_seg_image_url.gif" width="400"> |

_(Suggestion: Run your code on a short clip, record the output, and convert it to a GIF. Upload the GIF to a site like Imgur and replace the placeholder URLs above.)_

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- An NVIDIA GPU is recommended for real-time performance.

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/person-anonymizer.git](https://github.com/your-username/person-anonymizer.git)
cd person-anonymizer
```

### 2. Set Up a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

Install the required Python packages from `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4. Download Models

The YOLO models are required for detection, segmentation, and pose estimation.

- [yolo11n.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt)
- [yolo11n-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.pt)
- [yolo11n-pose.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt)

Place the downloaded `.pt` files into the `models/` directory.

---

## 💻 Usage

The scripts can be run directly from the command line. You can specify the video source, model paths, and other parameters.

### Anonymization using Segmentation Masks

This method is more precise and follows the person's outline.

```bash
python anonymise/anonymize_seg.py \
    --video-source data/your_video.mp4 \
    --seg-model models/yolo11n-seg.pt \
    --pose-model models/yolo11n-pose.pt
```

### Anonymization using Bounding Boxes

This method is faster but less precise, using rectangular boxes.

```bash
python anonymise/anonymize_bbox.py \
    --video-source data/your_video.mp4 \
    --bbox-model models/yolo11n.pt \
    --pose-model models/yolo11n-pose.pt
```

To use a webcam feed, set `--video-source 0`.

### Command-Line Arguments

Both scripts accept several arguments. Here are the most common ones:

- `--video-source`: Path to the input video file or `0` for webcam.
- `--output-video`: Path to save the anonymized output video.
- `--seg-model` / `--bbox-model`: Path to the YOLO detection model.
- `--pose-model`: Path to the YOLO pose estimation model.
- `--alpha`: Learning rate for background adaptation (0.0 to 1.0).
- `--pose-conf-threshold`: Confidence threshold for displaying keypoints.

For a full list of options, run the script with the `-h` or `--help` flag.

---

## 🛠️ How It Works

The anonymization process follows these steps for each frame of the video:

1.  **Person Detection**: A YOLOv8 model detects all persons in the current frame, generating either bounding boxes or segmentation masks.
2.  **Mask Creation**: A binary mask is created from the detected persons. For segmentation, this mask is inflated using a distance transform to ensure full coverage. For bounding boxes, the mask is dilated.
3.  **Background Update**: The dynamic background model is updated. The update is "frozen" in the areas covered by the person mask, meaning only visible background pixels are used for adaptation.
4.  **Soft Blending**: The person mask is blurred to create soft, feathered edges.
5.  **Anonymization**: The original frame is alpha-blended with the reconstructed background using the soft mask. This replaces the person with the background.
6.  **Pose Overlay**: A YOLOv8-Pose model detects human keypoints, and a skeleton is drawn on the anonymized area to preserve motion context.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
