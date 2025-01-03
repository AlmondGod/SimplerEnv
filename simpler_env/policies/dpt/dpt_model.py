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

from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import math
import torch
import torch.nn as nn
import torchvision
import collections
import zarr
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

#@markdown ### **Vision Encoder**
#@markdown
#@markdown Defines helper functions:
#@markdown - `get_resnet` to initialize standard ResNet vision encoder
#@markdown - `replace_bn_with_gn` to replace all BatchNorm layers with GroupNorm

def get_resnet(name:str, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()
    return resnet


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module

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

        # if you have multiple camera views, use seperate encoder weights for each view.
        vision_encoder = get_resnet('resnet18')

        # IMPORTANT!
        # replace all BatchNorm with GroupNorm to work with EMA
        # performance will tank if you forget to do this!
        vision_encoder = replace_bn_with_gn(vision_encoder)

        # ResNet18 has output dim of 512
        vision_feature_dim = 512
        # agent_pos is 2 dimensional
        lowdim_obs_dim = 2
        # observation feature has 514 dims in total per step
        obs_dim = vision_feature_dim + lowdim_obs_dim
        action_dim = 2

        # create network object
        noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim*obs_horizon
        )

        # the final arch has 2 parts
        nets = nn.ModuleDict({
            'vision_encoder': vision_encoder,
            'noise_pred_net': noise_pred_net
        })

        self.state_dict = torch.load(saved_model_path, map_location='cuda')
        self.ema_nets = nets
        self.ema_nets.load_state_dict(self.state_dict)

        # self.device = torch.device(device)
        # self.image_size = image_size
        # self.action_scale = action_scale
        # self.policy_setup = policy_setup
        
        # # Initialize observation window
        # self.obs_window = deque(maxlen=2)
        # self.task_embedding = None
        # self.task_description = None

        # # Initialize noise scheduler
        # self.noise_scheduler = DDPMScheduler(
        #     num_train_timesteps=100,
        #     beta_schedule='squaredcos_cap_v2',
        #     clip_sample=True,
        #     prediction_type='epsilon'
        # )

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
        # Resize image if needed
        image = self._resize_image(image)
        
        # Update observation window
        self.obs_window.append({
            'agent_pos': proprioception.squeeze(0).cpu().numpy(),
            "head_cam": torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        })

        # Stack observations from window
        images = torch.stack([x["head_cam"] for x in self.obs_window]).to(self.device)
        agent_poses = torch.from_numpy(
            np.stack([x["agent_pos"] for x in self.obs_window])
        ).to(self.device)

        # Get image features from vision encoder
        with torch.no_grad():
            image_features = self.ema_nets['vision_encoder'](
                images.flatten(end_dim=1)
            )
            image_features = image_features.reshape(*images.shape[:2], -1)

            # Concatenate with proprioception
            obs_features = torch.cat([image_features, agent_poses], dim=-1)
            obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

            # Initialize action from Gaussian noise
            noisy_action = torch.randn(
                (1, self.pred_horizon, self.action_dim), 
                device=self.device
            )
            naction = noisy_action

            # Initialize noise scheduler
            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

            # Denoise
            for k in self.noise_scheduler.timesteps:
                noise_pred = self.ema_nets['noise_pred_net'](
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )
                
                # Inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

        # Get action prediction
        naction = naction.detach().cpu().numpy()[0]
        
        # Process actions into the expected format
        raw_action = {
            "world_vector": naction[0:3],
            "rotation_delta": naction[3:6],
            "gripper_closedness_action": naction[6:7],
            "terminate_episode": np.array([0.0])  # Default to not terminating
        }

        # Process the raw actions into the format expected by the environment
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

        processed_action["terminate_episode"] = raw_action["terminate_episode"]

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