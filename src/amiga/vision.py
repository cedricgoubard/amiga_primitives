from typing import Any
import datetime as dt
from typing import Dict

import numpy as np
from jaxtyping import Int
from ultralytics import YOLO


def load_yolov11_model(mdl_path: str) -> None:
    return YOLO(mdl_path, verbose=False)


def _check_img(img: np.array) -> Int[np.ndarray, "H W C"]:
    assert img.max() >= 10, f"Image should be int [0-255]; current max is {img.max()}"
    assert len(img.shape) == 3, f"Image should be 3D; current shape is {img.shape}"
    
    if img.shape[0] == 3: 
        print(f"Image should be H W C, got C H W: {img.shape}")
        img = img.transpose(1, 2, 0)
    return img


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

            if obj_id in self.tracked_objects:
                # Update the object if it exists
                self.tracked_objects[obj_id]["xywh"] = xywh
                self.tracked_objects[obj_id]["last_seen"] = current_time
                self.tracked_objects[obj_id]["class_id"] = cls_id
                self.tracked_objects[obj_id]["class_name"] = cls_name
            else:
                # Add new object
                self.tracked_objects[obj_id] = {
                    "xywh": xywh,
                    "last_seen": current_time,
                    "class_id": cls_id,
                    "class_name": cls_name
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
            detected_objects.append({"id": obj_id, "xywh": data["xywh"], "class_id": data["class_id"], "class_name": data["class_name"]})

        return detected_objects