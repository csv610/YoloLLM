"""Unit tests for mask_processing."""

import numpy as np

from mask_processing import Segment


class TestSegment:
    def test_create_segment(self):
        mask = np.ones((10, 10), dtype=np.float32)
        seg = Segment(mask, label="person")
        assert seg.label == "person"
        assert seg.mask.shape == (10, 10)

    def test_create_segment_default_label(self):
        mask = np.zeros((5, 5), dtype=np.float32)
        seg = Segment(mask)
        assert seg.label == "object"

