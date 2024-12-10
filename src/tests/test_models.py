import unittest
from unittest.mock import MagicMock, patch
import datetime as dt

import numpy as np
import torch

import amiga.models as mdl


class TestLearningGraspParams(unittest.TestCase):
    def test_compute_l2_dist(self):
        """Test L2 distance computation."""
        a = torch.tensor([1, 2, 3]).unsqueeze(0).float()
        b = torch.tensor([4, 5, 2]).unsqueeze(0).float()
        dist, dx, dy, dz = mdl.compute_l2_distance(a, b)
        self.assertAlmostEqual(dist.detach().cpu().numpy()[0], 4.3589, places=4)
        self.assertAlmostEqual(dx.detach().cpu().numpy()[0], 3, places=4)
        self.assertAlmostEqual(dy.detach().cpu().numpy()[0], 3, places=4)
        self.assertAlmostEqual(dz.detach().cpu().numpy()[0], 1, places=4)