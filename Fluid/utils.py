import zmq
import threading
import time
import json
from PIL import Image
import numpy as np
import cv2

class ZmqMaskReceiver:
    def __init__(self, url="tcp://127.0.0.1:5556", topic="frame", hwm=10, conflated=False, poll_timeout_ms=5):
        self.ctx = zmq.Context.instance()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(url)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, topic)
        self.sub.setsockopt(zmq.RCVHWM, hwm)
        if conflated:
            self.sub.setsockopt(zmq.CONFLATE, 1)  # keep only latest in socket
        self.poller = zmq.Poller()
        self.poller.register(self.sub, zmq.POLLIN)
        self.poll_timeout_ms = poll_timeout_ms

        self._lock = threading.Lock()
        self._latest = None  # latest numpy array (h, w)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self.poller.unregister(self.sub)
        except Exception:
            pass
        self.sub.close(0)

    def _loop(self):
        # Non-blocking poll + recv; drop old frames and keep only the latest
        while not self._stop.is_set():
            socks = dict(self.poller.poll(self.poll_timeout_ms))
            if self.sub in socks and socks[self.sub] & zmq.POLLIN:
                try:
                    topic_b, header_b, raw_b = self.sub.recv_multipart(flags=zmq.NOBLOCK)
                    hdr = json.loads(header_b.decode("utf-8"))
                    h, w = hdr["shape"]
                    arr = np.frombuffer(raw_b, dtype=np.int16).reshape(h, w)

                    with self._lock:
                        self._latest = arr  # overwrite to keep only the newest
                except zmq.Again:
                    pass
            else:
                # tiny sleep to avoid hot loop
                time.sleep(0.001)

    def get_latest(self):
        # Return and keep; caller decides whether to reuse or not
        with self._lock:
            return self._latest

def flat_flow_to_rgb(flow_flat, h=None, w=None):
    """
    Converts a flattened flow array (N, 2) to a flattened RGB color array (N, 3).
    
    Args:
        flow_flat: Numpy array of shape (N, 2). 
                   Column 0 is u (x-velocity), Column 1 is v (y-velocity).
        h, w: (Optional) If provided, reshapes the output to an (H, W, 3) image.
    
    Returns:
        rgb: Numpy array of shape (N, 3) - or (H, W, 3) if dims provided.
    """
    # 1. Calculate Magnitude and Angle
    # We can pass the flat columns directly
    mag, ang = cv2.cartToPolar(flow_flat[:, 0], flow_flat[:, 1])

    # 2. Create HSV wrapper
    # We create a (1, N, 3) array to make cvtColor happy
    n_points = flow_flat.shape[0]
    hsv = np.zeros((1, n_points, 3), dtype=np.uint8)

    # 3. Map Direction (Angle) to Hue (0-179)
    # ang is in radians. Convert to degrees, divide by 2.
    hsv[0, :, 0] = ang * 180 / np.pi / 2

    # 4. Map Speed (Magnitude) to Saturation (0-255)
    # Note: Normalizing based on the max speed in THIS batch.
    # If using for video/animation, consider using a fixed max value (e.g., 50.0) 
    # to prevent flickering colors.
    hsv[0, :, 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # 5. Set Value (Brightness) to 255
    hsv[0, :, 2] = 255

    # 6. Convert HSV to RGB
    # We assume the input is a single row image of width N
    rgb_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # 7. Reshape to desired output
    if h is not None and w is not None:
        # Return as 2D Image (H, W, 3)
        return rgb_image.reshape((h, w, 3))
    else:
        # Return as flat array (N, 3) - useful for vertex colors / particles
        return rgb_image.reshape((n_points, 3))

def rgb_array_to_hex(rgb_array):
    """
    Convert RGB array to hex integer array.
    
    Args:
        rgb_array: np.ndarray of shape (N, 3) with values in [0, 1] or [0, 255]
    
    Returns:
        np.ndarray of shape (N,) with hex integers (0xRRGGBB format)
    """
    # If values are in [0, 1], scale to [0, 255]
    if rgb_array.max() <= 1.0:
        rgb_array = (rgb_array * 255).astype(np.uint8)
    else:
        rgb_array = rgb_array.astype(np.uint8)
    
    # Bit shift and combine: (R << 16) | (G << 8) | B
    hex_array = (rgb_array[:, 0].astype(np.uint32) << 16) | \
                (rgb_array[:, 1].astype(np.uint32) << 8) | \
                 rgb_array[:, 2].astype(np.uint32)
    
    return hex_array