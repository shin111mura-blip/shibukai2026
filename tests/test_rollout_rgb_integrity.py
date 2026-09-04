import numpy as np


def test_rgb_contract_is_uint8_hwc():
    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    assert rgb.dtype == np.uint8
    assert rgb.shape[-1] == 3
