import os
import torch
import dill
import hydra
import numpy as np
from typing import Optional, Sequence, Dict, Any

class DiffusionPolicyTransformerInference:
    def __init__(
        self,
        saved_model_path: str,
        policy_setup: str = "widowx_bridge",
        device: str = "cuda:0"
    ) -> None:
        """Initialize DPT inference model.
        
        Args:
            saved_model_path: Path to the checkpoint file
            policy_setup: Robot setup type ("widowx_bridge" or "google_robot")
            device: Device to run inference on
        """
        self.device = torch.device(device)
        
        # Load checkpoint
        payload = torch.load(open(saved_model_path, 'rb'), pickle_module=dill)
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
        
        # Store config
        self.cfg = cfg
        self.policy_setup = policy_setup

    def step(
        self, 
        obs_dict: Dict[str, np.ndarray],
        task_description: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """Run single inference step.
        
        Args:
            obs_dict: Observation dictionary containing camera images and robot state
            task_description: Optional task description string
            
        Returns:
            Dictionary containing predicted actions
        """
        with torch.no_grad():
            # Convert numpy observations to torch tensors
            obs_dict_torch = {
                k: torch.from_numpy(v).unsqueeze(0).to(self.device) 
                for k, v in obs_dict.items()
            }
            
            # Add task description if provided
            if task_description is not None:
                obs_dict_torch['task_description'] = task_description
                
            # Run inference
            result = self.policy.predict_action(obs_dict_torch)
            
            # Convert predictions back to numpy
            action = result['action'][0].detach().cpu().numpy()
            
            return {'action': action}
    
    def reset(self):
        """Reset policy state between episodes."""
        self.policy.reset()
        
    def visualize_epoch(
        self,
        predicted_actions: Sequence[np.ndarray],
        observations: Sequence[np.ndarray],
        save_path: str
    ) -> None:
        """Visualize predictions for an epoch.
        
        Args:
            predicted_actions: List of predicted action arrays
            observations: List of observation arrays 
            save_path: Path to save visualization
        """
        # Implement visualization logic here
        # Could save action trajectories overlaid on images
        pass