import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pybullet as p
import pybullet_data


@dataclass(frozen=True)
class SimConfig:
    gui: bool = True
    dt: float = 1.0 / 60.0
    steps_per_action: int = 1

    img_width: int = 640
    img_height: int = 480
    fov: float = 70.0
    near: float = 0.02
    far: float = 50.0

    drone_start_pos: Tuple[float, float, float] = (0.0, 0.0, 0.4)
    drone_start_yaw: float = 0.0

    target_start_pos: Tuple[float, float, float] = (2.0, 0.0, 0.4)
    target_radius: float = 0.12

    drone_forward_speed: float = 0.6
    drone_yaw_rate_rad_s: float = math.radians(75.0)

    target_motion: bool = True
    target_speed: float = 0.2


class PyBulletFollowSim:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.client_id: Optional[int] = None
        self.plane_id: Optional[int] = None
        self.drone_id: Optional[int] = None
        self.target_id: Optional[int] = None

        self._drone_pos = np.array(cfg.drone_start_pos, dtype=np.float32)
        self._drone_yaw = float(cfg.drone_start_yaw)
        self._t = 0.0

    def connect(self) -> None:
        if self.client_id is not None:
            return
        self.client_id = p.connect(p.GUI if self.cfg.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client_id)
        p.setTimeStep(self.cfg.dt, physicsClientId=self.client_id)

    def reset(
        self,
        seed: Optional[int] = None,
        drone_start_pos: Optional[Tuple[float, float, float]] = None,
        drone_start_yaw: Optional[float] = None,
    ) -> None:
        self.connect()
        if seed is not None:
            np.random.seed(seed)

        p.resetSimulation(physicsClientId=self.client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client_id)
        p.setTimeStep(self.cfg.dt, physicsClientId=self.client_id)

        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)

        start_pos = self.cfg.drone_start_pos if drone_start_pos is None else drone_start_pos
        start_yaw = self.cfg.drone_start_yaw if drone_start_yaw is None else float(drone_start_yaw)

        self._drone_pos = np.array(start_pos, dtype=np.float32)
        self._drone_yaw = float(start_yaw)
        self._t = 0.0

        drone_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.10, 0.10, 0.03], physicsClientId=self.client_id)
        drone_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.10, 0.10, 0.03], rgbaColor=[0.1, 0.1, 0.8, 1.0], physicsClientId=self.client_id)
        self.drone_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=drone_col,
            baseVisualShapeIndex=drone_vis,
            basePosition=self._drone_pos.tolist(),
            baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, self._drone_yaw]),
            physicsClientId=self.client_id,
        )

        try:
            self.target_id = p.loadURDF(
                "soccerball.urdf",
                basePosition=list(self.cfg.target_start_pos),
                baseOrientation=[0.0, 0.0, 0.0, 1.0],
                useFixedBase=True,
                globalScaling=max(0.1, float(self.cfg.target_radius) / 0.11),
                physicsClientId=self.client_id,
            )
        except Exception:
            target_col = p.createCollisionShape(p.GEOM_SPHERE, radius=self.cfg.target_radius, physicsClientId=self.client_id)
            target_vis = p.createVisualShape(p.GEOM_SPHERE, radius=self.cfg.target_radius, rgbaColor=[0.9, 0.1, 0.1, 1.0], physicsClientId=self.client_id)
            self.target_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=target_col,
                baseVisualShapeIndex=target_vis,
                basePosition=list(self.cfg.target_start_pos),
                baseOrientation=[0.0, 0.0, 0.0, 1.0],
                physicsClientId=self.client_id,
            )

        if self.cfg.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=3.0,
                cameraYaw=35,
                cameraPitch=-30,
                cameraTargetPosition=[1.0, 0.0, 0.4],
                physicsClientId=self.client_id,
            )

        for _ in range(10):
            p.stepSimulation(physicsClientId=self.client_id)
            if self.cfg.gui:
                time.sleep(self.cfg.dt)

    def close(self) -> None:
        if self.client_id is None:
            return
        p.disconnect(physicsClientId=self.client_id)
        self.client_id = None

    def _forward_vec(self) -> np.ndarray:
        return np.array([math.cos(self._drone_yaw), math.sin(self._drone_yaw), 0.0], dtype=np.float32)

    def _update_target(self) -> None:
        if not self.cfg.target_motion or self.target_id is None:
            return
        self._t += self.cfg.dt
        cx, cy, cz = self.cfg.target_start_pos
        x = cx
        y = cy + 0.7 * math.sin(self.cfg.target_speed * self._t)
        z = cz
        p.resetBasePositionAndOrientation(
            self.target_id,
            [x, y, z],
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client_id,
        )

    def step(self, action: int) -> None:
        if self.drone_id is None:
            raise RuntimeError("Call reset() before step().")

        if action == 1:
            self._drone_yaw += self.cfg.drone_yaw_rate_rad_s * self.cfg.dt
        elif action == 2:
            self._drone_yaw -= self.cfg.drone_yaw_rate_rad_s * self.cfg.dt
        elif action == 3:
            self._drone_pos += self._forward_vec() * (self.cfg.drone_forward_speed * self.cfg.dt)
        elif action == 4:
            self._drone_pos -= self._forward_vec() * (self.cfg.drone_forward_speed * self.cfg.dt)

        quat = p.getQuaternionFromEuler([0.0, 0.0, self._drone_yaw])
        p.resetBasePositionAndOrientation(
            self.drone_id,
            self._drone_pos.tolist(),
            quat,
            physicsClientId=self.client_id,
        )

        for _ in range(self.cfg.steps_per_action):
            self._update_target()
            p.stepSimulation(physicsClientId=self.client_id)
            if self.cfg.gui:
                time.sleep(self.cfg.dt)

    def get_camera_frame_rgb(self) -> np.ndarray:
        if self.client_id is None:
            raise RuntimeError("Call reset() before get_camera_frame_rgb().")

        eye = (self._drone_pos + np.array([0.0, 0.0, 0.08], dtype=np.float32)).tolist()
        target = (self._drone_pos + np.array([0.0, 0.0, 0.08], dtype=np.float32) + self._forward_vec()).tolist()
        up = [0.0, 0.0, 1.0]

        view = p.computeViewMatrix(cameraEyePosition=eye, cameraTargetPosition=target, cameraUpVector=up)
        proj = p.computeProjectionMatrixFOV(
            fov=self.cfg.fov,
            aspect=float(self.cfg.img_width) / float(self.cfg.img_height),
            nearVal=self.cfg.near,
            farVal=self.cfg.far,
        )

        _, _, rgba, _, _ = p.getCameraImage(
            width=self.cfg.img_width,
            height=self.cfg.img_height,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=p.ER_BULLET_HARDWARE_OPENGL if self.cfg.gui else p.ER_TINY_RENDERER,
            physicsClientId=self.client_id,
        )

        rgba = np.array(rgba, dtype=np.uint8).reshape((self.cfg.img_height, self.cfg.img_width, 4))
        return rgba[:, :, :3]

    @property
    def img_size(self) -> Tuple[int, int]:
        return self.cfg.img_width, self.cfg.img_height
