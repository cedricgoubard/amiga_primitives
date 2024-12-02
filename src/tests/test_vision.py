import unittest
from unittest.mock import MagicMock, patch
import datetime as dt

import numpy as np
import torch

import amiga.vision as av


class TestCheckImage(unittest.TestCase):
    def _generate_random_image(self, shape, low=10, high=256):
        """Helper to generate a random image."""
        return np.random.randint(low, high, size=shape, dtype=np.uint8)

    def test_valid_input(self):
        """Test with a valid HWC image."""
        img = self._generate_random_image((100, 200, 3))
        result = av._check_img(img)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (100, 200, 3))

    def test_transpose_input(self):
        """Test with a CHW image to check transposing."""
        img = self._generate_random_image((3, 100, 200))
        result = av._check_img(img)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (100, 200, 3))  # Transposed to HWC

    def test_low_max_value(self):
        """Test with an image that has max value below 10."""
        img = self._generate_random_image((100, 200, 3), low=0, high=9)
        with self.assertRaises(AssertionError) as context:
            av._check_img(img)
        self.assertIn("Image should be int [0-255]; current max is", str(context.exception))

    def test_invalid_dimensions(self):
        """Test with an image that is not 3D."""
        img = self._generate_random_image((100, 200))
        with self.assertRaises(AssertionError) as context:
            av._check_img(img)
        self.assertIn("Image should be 3D; current shape is", str(context.exception))

    def test_edge_case(self):
        """Test with an image that just meets the threshold."""
        img = self._generate_random_image((50, 50, 3), low=10, high=11)
        result = av._check_img(img)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (50, 50, 3))


class TestKitchenObjectDetector(unittest.TestCase):
    @patch("amiga.vision.load_yolov11_model")
    def setUp(self, mock_load_model):
        """Set up a mock YOLO model and detector instance."""
        self.mock_model = MagicMock()
        mock_load_model.return_value = self.mock_model

        # Mock YOLO model behavior
        self.mock_model.names = {0: "apple", 1: "banana"}
        self.mock_model.track.return_value = [
            MagicMock(
                boxes=[
                    MagicMock(
                        id=torch.from_numpy(np.array([1])), 
                        xywh=torch.from_numpy(np.array([[50, 50, 20, 20]])), 
                        cls=torch.from_numpy(np.array([0])), 
                        conf=torch.from_numpy(np.array([0.88]))
                    ),
                    MagicMock(
                        id=torch.from_numpy(np.array([2])), 
                        xywh=torch.from_numpy(np.array([[100, 100, 30, 30]])), 
                        cls=torch.from_numpy(np.array([1])), 
                        conf=torch.from_numpy(np.array([0.92]))
                    )
                ]
            )
        ]

        self.detector = av.KitchenObjectDetector(mdl_path="mock_model_path", time_buffer_sec=0.5)

    def test_detect_objects(self):
        """Test basic object detection."""
        img = np.random.randint(10, 256, size=(200, 200, 3), dtype=np.uint8)

        # Call the detector
        results = self.detector(img)
        # Verify detected objects
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], 1)
        self.assertEqual(results[0]["xywh"].tolist(), [50, 50, 20, 20])
        self.assertEqual(results[1]["id"], 2)
        self.assertEqual(results[1]["xywh"].tolist(), [100, 100, 30, 30])

    def test_time_buffer_keeps_recent_objects(self):
        """Test that objects remain within the buffer period."""
        img = np.random.randint(10, 256, size=(200, 200, 3), dtype=np.uint8)

        now = dt.datetime.now()
        # Initial detection
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = now
            self.detector(img)

        # Mock an empty model output to simulate a frame without detections
        self.mock_model.track.return_value = [ MagicMock(boxes=[]) ]

        # Call again after less than the time buffer duration
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = now + dt.timedelta(seconds=0.3)
            results = self.detector(img)

        # Verify that previous objects are still retained
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], 1)
        self.assertEqual(results[1]["id"], 2)

    def test_time_buffer_expires_old_objects(self):
        """Test that objects are removed after the buffer expires."""
        img = np.random.randint(10, 256, size=(200, 200, 3), dtype=np.uint8)

        now = dt.datetime.now()
        
        # Initial detection
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = now
            self.detector(img)

        # Mock an empty model output to simulate a frame without detections
        self.mock_model.track.return_value = [ MagicMock(boxes=[]) ]

        # Call again after more than the time buffer duration
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = now + dt.timedelta(seconds=0.6)
            results = self.detector(img)

        # Verify that no objects remain
        self.assertEqual(len(results), 0)

    def test_check_img_called(self):
        """Ensure _check_img is called during detection."""
        with patch("amiga.vision._check_img", wraps=av._check_img) as mock_check_img:
            img = np.random.randint(10, 256, size=(200, 200, 3), dtype=np.uint8)
            self.detector(img)
            mock_check_img.assert_called_once()

    def test_empty_frame(self):
        """Test behavior with no detections."""
        img = np.random.randint(10, 256, size=(200, 200, 3), dtype=np.uint8)

        # Mock an empty detection result
        self.mock_model.track.return_value = [ MagicMock(boxes=[]) ]

        results = self.detector(img)

        # Verify no objects are returned
        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()
