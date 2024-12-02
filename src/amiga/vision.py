from typing import Any
import datetime as dt
from typing import Dict, List, Any
import hashlib

import numpy as np
from jaxtyping import Int
from ultralytics import YOLO
import cv2


def load_yolov11_model(mdl_path: str) -> None:
    return YOLO(mdl_path, verbose=False)


def _check_img(img: np.array) -> Int[np.ndarray, "H W C"]:
    assert img.max() >= 10, f"Image should be int [0-255]; current max is {img.max()}"
    assert len(img.shape) == 3, f"Image should be 3D; current shape is {img.shape}"
    
    if img.shape[0] == 3: 
        print(f"Image should be H W C, got C H W: {img.shape}")
        img = img.transpose(1, 2, 0)
    return img


def deterministic_color(name):
    """Generate a deterministic color based on a class name."""
    # Hash the class name and take the first 3 bytes for RGB values
    hash_bytes = hashlib.md5(name.encode('utf-8')).digest()
    return tuple(int(hash_bytes[i]) % 256 for i in range(3))


def overlay_results(image: np.ndarray, results: List[Dict[str, Any]]) -> np.ndarray:
    """Overlay detection results on an image."""

    for result in results:
        # Unpack details
        obj_id = result['id']
        conf = result['conf']
        x, y, w, h = result['xywh']
        class_name = result['class_name']

        # Convert xywh to top-left and bottom-right corners
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)

        # Get color for the class name
        color = deterministic_color(class_name)

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Prepare label text
        label = f"{class_name} ({obj_id:.0f}): {conf:.2f}"

        # Put the label above the box
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image

class KitchenObjectDetector:
    def __init__(self, mdl_path: str, time_buffer_sec: float = 0.5):
        """
        Initialise the KitchenObjectDetector.
        
        Args:
            mdl_path (str): Path to the YOLO model.
            time_buffer (float): Time buffer (in seconds) to prevent object flickering.
        """
        self.mdl = load_yolov11_model(mdl_path)
        self.time_buffer = dt.timedelta(seconds=time_buffer_sec)
        self.tracked_objects: Dict[int, Dict] = {}  # Tracks object_id, bbox, and last_seen timestamp.

    def __call__(self, img: np.ndarray) -> Any:
        img = _check_img(img)

        results = self.mdl.track(img, verbose=False)[0]

        detected_objects = []
        current_time = dt.datetime.now()

        # Update tracked objects
        for obj in results.boxes:
            if not obj.is_track:
                continue
            
            obj_id = obj.id.detach().cpu().numpy()[0]
            xywh = obj.xywh.detach().cpu().numpy()[0]
            cls_id = int(obj.cls.detach().cpu().numpy()[0])
            cls_name = self.mdl.names[cls_id]
            conf = obj.conf.detach().cpu().numpy()[0]

            if obj_id in self.tracked_objects:
                # Update the object if it exists
                self.tracked_objects[obj_id]["xywh"] = xywh
                self.tracked_objects[obj_id]["last_seen"] = current_time
                self.tracked_objects[obj_id]["class_id"] = cls_id
                self.tracked_objects[obj_id]["class_name"] = cls_name
                self.tracked_objects[obj_id]["conf"] = conf
            else:
                # Add new object
                self.tracked_objects[obj_id] = {
                    "xywh": xywh,
                    "last_seen": current_time,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "conf": conf
                }

        # Remove stale objects
        expired_ids = [
            obj_id
            for obj_id, data in self.tracked_objects.items()
            if current_time - data["last_seen"] > self.time_buffer
        ]
        for obj_id in expired_ids:
            del self.tracked_objects[obj_id]
       
        # Add objects from buffer that are still valid
        for obj_id, data in self.tracked_objects.items():
            detected_objects.append({"id": obj_id, "xywh": data["xywh"], "class_id": data["class_id"], "class_name": data["class_name"], "conf": data["conf"]})

        return detected_objects
