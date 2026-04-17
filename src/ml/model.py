"""PPO model using Stable-Baselines3 for Molty Royale."""

import os
from typing import Optional
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.ml.environment import MoltyRoyaleEnv
from src.constants import (
    TrainingConfig,
    ModelConfig,
    OBSERVATION_SPACE,
    ACTION_SPACE,
)


class ModelSaveCallback(BaseCallback):
    """Callback to save model at regular intervals."""
    
    def __init__(self, save_path: str, save_interval: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_interval = save_interval
        os.makedirs(save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        if self.n_calls % self.save_interval == 0:
            model_path = os.path.join(self.save_path, f"ppo_model_{self.n_calls}_steps")
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Model saved to {model_path}")
        return True


class MoltyRoyalePPO:
    """PPO model wrapper for Molty Royale."""
    
    def __init__(
        self,
        env: Optional[MoltyRoyaleEnv] = None,
        learning_rate: float = TrainingConfig.LEARNING_RATE,
        n_steps: int = 2048,
        batch_size: int = TrainingConfig.BATCH_SIZE,
        n_epochs: int = 10,
        gamma: float = TrainingConfig.GAMMA,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        verbose: int = 1,
        tensorboard_log: Optional[str] = None,
        device: str = "auto"
    ):
        self.env = env
        self.model = None
        self.device = device
        
        # PPO hyperparameters
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.verbose = verbose
        self.tensorboard_log = tensorboard_log
        
        # Create environment if not provided
        if env is None:
            self.env = MoltyRoyaleEnv()
        
        # Wrap environment for vectorized training
        self.vec_env = DummyVecEnv([lambda: self.env])
        
        # Build model
        self._build_model()
    
    def _build_model(self):
        """Build PPO model with MlpPolicy."""
        self.model = PPO(
            "MlpPolicy",
            self.vec_env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            verbose=self.verbose,
            tensorboard_log=self.tensorboard_log,
            device=self.device,
            policy_kwargs={
                "net_arch": ModelConfig.HIDDEN_LAYERS,
                "activation_fn": torch.nn.ReLU,
                "ortho_init": True,
            }
        )
    
    def train(
        self,
        total_timesteps: int,
        save_path: str = "data/models",
        save_interval: int = TrainingConfig.SAVE_INTERVAL,
        callback: Optional[BaseCallback] = None
    ):
        """Train the PPO model."""
        # Setup callbacks
        callbacks = []
        
        # Model save callback
        save_callback = ModelSaveCallback(
            save_path=save_path,
            save_interval=save_interval,
            verbose=self.verbose
        )
        callbacks.append(save_callback)
        
        # Add custom callback if provided
        if callback:
            callbacks.append(callback)
        
        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks
        )
        
        # Save final model
        final_path = os.path.join(save_path, "ppo_model_final")
        self.model.save(final_path)
        print(f"Final model saved to {final_path}")
    
    def predict(self, observation, deterministic: bool = False):
        """Predict action given observation."""
        action, state = self.model.predict(observation, deterministic=deterministic)
        return action, state
    
    def load(self, model_path: str):
        """Load a trained model."""
        self.model = PPO.load(model_path, env=self.vec_env, device=self.device)
        print(f"Model loaded from {model_path}")
    
    def save(self, model_path: str):
        """Save the current model."""
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
    
    def get_env(self):
        """Get the environment."""
        return self.env
    
    def set_env(self, env):
        """Set a new environment."""
        self.env = env
        self.vec_env = DummyVecEnv([lambda: env])
        self.model.set_env(self.vec_env)
    
    def get_parameters(self):
        """Get model parameters."""
        return self.model.get_parameters()
    
    def set_parameters(self, parameters):
        """Set model parameters."""
        self.model.set_parameters(parameters)
