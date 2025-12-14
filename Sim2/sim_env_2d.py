import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import ultralytics


@dataclass(frozen=True)
class Sim2DConfig:
    img_width: int = 640
    img_height: int = 480

    dt: float = 1.0 / 60.0

    drone_start_yaw: float = 0.0
    drone_start_dist: float = 5.0

    yaw_rate_rad_s: float = math.radians(90.0)
    forward_speed: float = 1.0

    min_dist: float = 0.7
    max_dist: float = 8.0

    target_bearing_start: float = 0.0
    target_bearing_rate: float = math.radians(12.0)

    sprite_name: str = "bus.jpg"


class Kinematic2DFollowSim:
    def __init__(self, cfg: Sim2DConfig):
        self.cfg = cfg
        self._yaw = float(cfg.drone_start_yaw)
        self._dist = float(cfg.drone_start_dist)
        self._t = 0.0

        assets_dir = Path(ultralytics.__file__).resolve().parent / "assets"
        sprite_path = assets_dir / cfg.sprite_name
        sprite = cv2.imread(str(sprite_path), cv2.IMREAD_COLOR)
        if sprite is None:
            raise FileNotFoundError(f"Could not load Ultralytics asset sprite: {sprite_path}")
        self._sprite_bgr = sprite

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            np.random.seed(seed)
        self._yaw = float(self.cfg.drone_start_yaw)
        self._dist = float(self.cfg.drone_start_dist)
        self._t = 0.0

    def step(self, action: int) -> None:
        self._t += self.cfg.dt

        if action == 1:
            self._yaw += self.cfg.yaw_rate_rad_s * self.cfg.dt
        elif action == 2:
            self._yaw -= self.cfg.yaw_rate_rad_s * self.cfg.dt
        elif action == 3:
            self._dist -= self.cfg.forward_speed * self.cfg.dt
        elif action == 4:
            self._dist += self.cfg.forward_speed * self.cfg.dt

        self._dist = float(np.clip(self._dist, self.cfg.min_dist, self.cfg.max_dist))

    def get_camera_frame_rgb(self) -> np.ndarray:
        bearing = self.cfg.target_bearing_start + self.cfg.target_bearing_rate * math.sin(0.6 * self._t)
        rel = (bearing - self._yaw + math.pi) % (2.0 * math.pi) - math.pi

        cx = int(self.cfg.img_width / 2.0 + (rel / (math.radians(45.0))) * (self.cfg.img_width / 3.0))
        cy = int(self.cfg.img_height / 2.0)

        closeness = 1.0 / max(0.01, self._dist)
        size = int(np.clip(220.0 * closeness, 30.0, 260.0))

        bg = np.zeros((self.cfg.img_height, self.cfg.img_width, 3), dtype=np.uint8)

        sprite = cv2.resize(self._sprite_bgr, (size, size), interpolation=cv2.INTER_AREA)

        x1 = int(cx - size / 2)
        y1 = int(cy - size / 2)
        x2 = x1 + size
        y2 = y1 + size

        sx1 = 0
        sy1 = 0
        sx2 = size
        sy2 = size

        if x1 < 0:
            sx1 = -x1
            x1 = 0
        if y1 < 0:
            sy1 = -y1
            y1 = 0
        if x2 > self.cfg.img_width:
            sx2 = size - (x2 - self.cfg.img_width)
            x2 = self.cfg.img_width
        if y2 > self.cfg.img_height:
            sy2 = size - (y2 - self.cfg.img_height)
            y2 = self.cfg.img_height

        if x1 < x2 and y1 < y2 and sx1 < sx2 and sy1 < sy2:
            bg[y1:y2, x1:x2] = sprite[sy1:sy2, sx1:sx2]

        rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        return rgb

    def close(self) -> None:
        return

    @property
    def img_size(self) -> Tuple[int, int]:
        return self.cfg.img_width, self.cfg.img_height
