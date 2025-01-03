from collections import deque
from typing import Optional, Sequence
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transforms3d.euler import euler2axangle

import sys
sys.path.append('/content/RoboticsDiffusionTransformer')

# Import the RDT model creation function
from scripts.maniskill_model import create_model, RoboticDiffusionTransformerModel
import yaml

class RDT1BInference:
    def __init__(
        self,
        saved_model_path: str,
        policy_setup: str = "widowx_bridge",
        image_size: int = 384,  # SigLIP model input size
        action_scale: float = 1.0,
    ) -> None:
        print("is this even updating")
        # Load config and create model
        # config_path = os.path.join(os.path.dirname(saved_model_path), 'configs/base.yaml')
        config_path = os.path.abspath('/content/base.yaml')
        print(config_path)
        with open(config_path, "r") as fp:
            self.config = yaml.safe_load(fp)

        # Initialize model with pretrained encoders
        self.model = create_model(
            args=self.config,
            dtype=torch.bfloat16,
            pretrained=saved_model_path,
            pretrained_text_encoder_name_or_path="google/t5-v1_1-xxl",
            pretrained_vision_encoder_name_or_path="google/siglip-so400m-patch14-384"
        )
        self.model.eval()

        self.image_size = image_size
        self.action_scale = action_scale
        self.policy_setup = policy_setup
        
        # Initialize observation window for temporal context
        self.obs_window = deque(maxlen=2)
        self.task_embedding = None
        self.task_description = None

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
        self.task_embedding = self.model.encode_instruction(task_description)
        self.model.reset()

    def step(self, image: np.ndarray, task_description: Optional[str] = None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Process new observation and generate action"""
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)

        # Update observation window
        image = self._resize_image(image)
        self.obs_window.append(image)
        
        # Prepare inputs for model
        image_arrs = []
        for window_img in self.obs_window:
            image_arrs.append(window_img)
            image_arrs.append(None)
            image_arrs.append(None)
        
        images = [Image.fromarray(arr) if arr is not None else None
                 for arr in image_arrs]

        # Get model predictions
        with torch.no_grad():
            actions = self.model.step(None, images, self.task_embedding)
            actions = actions.squeeze(0).cpu().numpy()
        
        # Take every 4th action since RDT predicts interpolated steps
        action = actions[::4][0]  # Take first action

        # Format raw action output
        raw_action = {
            "world_vector": action[:3],
            "rotation_delta": action[3:6],
            "gripper_closedness_action": action[6:7]
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
        images = [self._resize_image(image) for image in images]
        
        # Setup visualization
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        img_strip = np.concatenate(images[::3], axis=1)

        # Create plot
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # Plot actions
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