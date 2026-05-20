from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import traceback


def extract_rgb_frame(obs, camera="rgb_static"):
    """
    Robustly extract RGB frame from CALVIN-like obs.
    Common structures:
      obs["rgb_obs"]["rgb_static"]
      obs["rgb_obs"]["rgb_gripper"]
      obs["rgb_static"]
      obs["rgb_gripper"]
    """
    frame = None

    if isinstance(obs, dict):
        if "rgb_obs" in obs and isinstance(obs["rgb_obs"], dict):
            frame = obs["rgb_obs"].get(camera)

        if frame is None:
            frame = obs.get(camera)

        if frame is None and camera == "rgb_static":
            # fallback candidates
            for key in ["rgb_static", "image", "image_primary"]:
                if key in obs:
                    frame = obs[key]
                    break

    if frame is None:
        return None

    frame = np.asarray(frame)

    # CHW -> HWC
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (3, 4):
        frame = np.transpose(frame, (1, 2, 0))

    # drop alpha
    if frame.ndim == 3 and frame.shape[-1] == 4:
        frame = frame[..., :3]

    # float [0,1] -> uint8
    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    return frame


class RolloutVideoRecorder:
    def __init__(self, save_dir, fps=15, camera="rgb_static", enabled=True):
        self.save_dir = Path(save_dir)
        self.fps = fps
        self.camera = camera
        self.enabled = enabled
        self.frames = []
        if self.enabled:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def add_obs(self, obs):
        if not self.enabled:
            return
        frame = extract_rgb_frame(obs, self.camera)
        if frame is not None:
            self.frames.append(frame)

    def save(self, name):
        if not self.enabled or not self.frames:
            return None

        path = self.save_dir / f"{name}.mp4"
        try:
            imageio.mimsave(path, self.frames, fps=self.fps)
            print(f"[video] saved {path}", flush=True)
            return path
        except Exception:
            print(f"[video] failed to save {path}", flush=True)
            traceback.print_exc()
            return None
        finally:
            self.frames = []

    def clear(self):
        self.frames = []
