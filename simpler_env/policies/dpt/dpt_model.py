import os
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from typing import Optional, Sequence, Dict
from PIL import Image
from transforms3d.euler import euler2axangle
import hydra
import dill

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torchvision.models import resnet18

class DiffusionPolicyTransformerInference:
    def __init__(
        self,
        saved_model_path: str,
        policy_setup: str = "widowx_bridge",
        image_size: int = 96,
        action_scale: float = 1.0,
        device: str = "cuda:0"
    ) -> None:
        """Initialize DPT inference model.
        
        Args:
            saved_model_path: Path to the checkpoint file
            policy_setup: Robot setup type ("widowx_bridge" or "google_robot")
            image_size: Size to resize input images to
            action_scale: Scale factor for actions
            device: Device to run inference on
        """
        self.device = torch.device(device)
        self.image_size = image_size
        self.action_scale = action_scale
        self.policy_setup = policy_setup
        
        # Load checkpoint
        payload = torch.load(open(saved_model_path, 'rb'), pickle_module=dill)
        print(f"payload keys: {payload.keys()}")
        cfg = payload['cfg']
        
        # Initialize workspace and load model
        cls = hydra.utils.get_class(cfg._target_)
        self.workspace = cls(cfg)
        self.workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        # Get policy from workspace
        self.policy = self.workspace.model
        if cfg.training.get('use_ema', False):
            self.policy = self.workspace.ema_model
            
        self.policy.to(self.device)
        self.policy.eval()

        # Initialize observation window
        self.obs_window = deque(maxlen=2)
        self.task_embedding = None
        self.task_description = None

        # Initialize noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to model's expected input size"""
        image = Image.fromarray(image)
        image = image.resize((self.image_size, self.image_size))
        return np.array(image)

    def reset(self, task_description: str) -> None:
        """Reset model state and compute new task embedding"""
        self.obs_window.clear()
        self.obs_window.extend([None, None])  # Initialize with empty context
        self.task_description = task_description
        self.policy.reset()

    def step(self, image: np.ndarray, task_description: Optional[str] = None, proprioception: Optional[np.ndarray] = None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Process new observation and generate action"""
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)

        # Update observation window
        image = self._resize_image(image)
        self.obs_window.append({
            'image': torch.from_numpy(image).float().to(self.device),
            'agent_pos': torch.from_numpy(proprioception).float().to(self.device) if proprioception is not None else None
        })

        # Run inference
        with torch.no_grad():
            actions = self.policy.predict_action(self.obs_window[-1])
            actions = actions.cpu().numpy()

        # Format raw action output
        raw_action = {
            "world_vector": actions[:3],
            "rotation_delta": actions[3:6],
            "gripper_closedness_action": actions[6:7]
        }

        # Process action for environment
        processed_action = {}
        processed_action["world_vector"] = raw_action["world_vector"] * self.action_scale
        
        # Convert rotation to axis-angle
        roll, pitch, yaw = raw_action["rotation_delta"]
        rot_axis, rot_angle = euler2axangle(roll, pitch, yaw)
        processed_action["rot_axangle"] = rot_axis * rot_angle * self.action_scale

        # Process gripper action based on policy setup
        if self.policy_setup == "widowx_bridge":
            processed_action["gripper"] = 2.0 * (raw_action["gripper_closedness_action"] > 0.5) - 1.0
        else:
            processed_action["gripper"] = raw_action["gripper_closedness_action"]

        processed_action["terminate_episode"] = np.array([0.0])

        return raw_action, processed_action

    def visualize_epoch(self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str) -> None:
        """Visualize trajectory with actions"""
        import matplotlib.pyplot as plt
        
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        
        img_strip = np.concatenate(images[::3], axis=1)
        
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        pred_actions = np.array([
            np.concatenate([
                a["world_vector"],
                a["rotation_delta"],
                a["gripper_closedness_action"]
            ], axis=-1) for a in predicted_raw_actions
        ])

        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)