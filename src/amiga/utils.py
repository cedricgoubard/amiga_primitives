import time


class Rate:
    def __init__(self, rate: float):
        self._interval = 1.0 / rate
        self._next_time = time.time() + self._interval

    def sleep(self):
        now = time.time()
        sleep_time = self._next_time - now
        if sleep_time > 0:
            time.sleep(sleep_time)
        self._next_time += self._interval


def centre_crop(rgb_frame, depth_frame):
    assert (
        len(rgb_frame.shape) == 3
        and rgb_frame.shape[-1] == 3 
        and len(depth_frame.shape) == 2
        and rgb_frame.shape[:-1] == depth_frame.shape
        ), "RGB frame should be (H, W, 3) and depth frame should be (H, W)"

    H, W = rgb_frame.shape[:-1]
    sq_size = min(H, W)

    h_start = (H - sq_size) // 2
    w_start = (W - sq_size) // 2

    # print(rgb_frame.shape, depth_frame.shape)

    rgb_frame = rgb_frame[h_start : h_start + sq_size, w_start : w_start + sq_size, :]
    depth_frame = depth_frame[h_start : h_start + sq_size, w_start : w_start + sq_size]

    return rgb_frame, depth_frame
