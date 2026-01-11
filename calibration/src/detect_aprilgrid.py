import argparse
from pathlib import Path

import cv2
import numpy as np

from aprilgrid import Detector


def load_grayscale_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to read image at {image_path}")
    return image


def load_color_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image at {image_path}")
    return image


def detect_aprilgrid_corners(image: np.ndarray, tag_family: str = "t36h11"):
    detector = Detector(tag_family)
    detections = detector.detect(image)
    results = []
    for detection in detections:
        corners = np.asarray(detection.corners, dtype=np.float32)
        if corners.ndim == 3 and corners.shape[1] == 1 and corners.shape[2] == 2:
            corners = corners[:, 0, :]
        else:
            corners = corners.reshape(-1, 2)
        results.append({"tag_id": detection.tag_id, "corners": corners})
    return results


def detect_from_path(image_path: Path, tag_family: str = "t36h11"):
    image = load_grayscale_image(image_path)
    return detect_aprilgrid_corners(image, tag_family=tag_family)


def draw_detections(image: np.ndarray, detections):
    for detection in detections:
        corners = detection["corners"]
        tag_id = detection["tag_id"]
        center = np.mean(corners, axis=0).astype(np.int32)
        cv2.putText(
            image,
            str(tag_id),
            (center[0], center[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        for corner in corners.astype(np.int32):
            cv2.circle(image, (corner[0], corner[1]), 4, (0, 255, 0), -1)
    return image


def main():
    parser = argparse.ArgumentParser(
        description="Detect AprilGrid corners in an image using the aprilgrid library."
    )
    parser.add_argument("image", type=Path, help="Path to the image file.")
    parser.add_argument(
        "--tag-family",
        default="t36h11",
        help="AprilTag family name (default: t36h11).",
    )
    args = parser.parse_args()

    detections = detect_from_path(args.image, tag_family=args.tag_family)
    print(f"Detected {len(detections)} tags")
    for detection in detections:
        tag_id = detection["tag_id"]
        corners = detection["corners"]
        print(f"tag_id={tag_id} corners={corners.tolist()}")

    color_image = load_color_image(args.image)
    annotated = draw_detections(color_image, detections)
    cv2.imshow("AprilGrid Detections", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
