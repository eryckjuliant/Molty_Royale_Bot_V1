"""Gymnasium environment for Molty Royale."""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces

from src.constants import (
    OBSERVATION_SPACE,
    ACTION_SPACE,
    NUM_ACTIONS,
    ALL_ACTIONS,
    IDX_TO_ACTION,
    GameConfig,
)


class MoltyRoyaleEnv(gym.Env):
    """Gymnasium environment for Molty Royale game."""
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 1000
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        
        # Define observation space (128-dimensional)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(128,),
            dtype=np.float32
        )
        
        # Define action space (Discrete with 15 actions for simplified version)
        # Note: Full action space has 26 actions, but we use 15 for PPO training
        self.action_space = spaces.Discrete(15)
        
        # Simplified action mapping for training
        self._simplified_actions = [
            (ALL_ACTIONS[0]),  # move up
            (ALL_ACTIONS[1]),  # move down
            (ALL_ACTIONS[2]),  # move left
            (ALL_ACTIONS[3]),  # move right
            (ALL_ACTIONS[4]),  # move up_left
            (ALL_ACTIONS[5]),  # move up_right
            (ALL_ACTIONS[6]),  # move down_left
            (ALL_ACTIONS[7]),  # move down_right
            (ALL_ACTIONS[9]),  # attack basic
            (ALL_ACTIONS[10]), # attack power
            (ALL_ACTIONS[14]), # collect resource
            (ALL_ACTIONS[19]), # special heal
            (ALL_ACTIONS[20]), # special boost
            (ALL_ACTIONS[24]), # wait
            (ALL_ACTIONS[25]), # interact
        ]
        
        # Environment state
        self.current_state: Optional[np.ndarray] = None
        self.current_step = 0
        self.episode_reward = 0.0
        self.done = False
        self.truncated = False
        
        # External API connection (to be set)
        self.api_client = None
        self.agent_id = None
        self.session_id = None
        
    def set_api_connection(
        self,
        api_client,
        agent_id: str,
        session_id: str
    ) -> None:
        """Set external API connection for real-time interaction."""
        self.api_client = api_client
        self.agent_id = agent_id
        self.session_id = session_id
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.episode_reward = 0.0
        self.done = False
        self.truncated = False
        
        # Get initial state from API if connected
        if self.api_client:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If running in async context, we need to handle differently
                    # For now, return zero state
                    self.current_state = np.zeros(128, dtype=np.float32)
                else:
                    raw_state = loop.run_until_complete(
                        self.api_client.get_state(self.agent_id, self.session_id)
                    )
                    self.current_state = self._parse_state(raw_state)
            except:
                self.current_state = np.zeros(128, dtype=np.float32)
        else:
            self.current_state = np.zeros(128, dtype=np.float32)
        
        info = {"episode_step": self.current_step}
        
        return self.current_state, info
    
    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        self.current_step += 1
        
        # Map simplified action to full action
        full_action = self._simplified_actions[action]
        action_type, action_params = full_action
        
        # Send action to API if connected
        reward = 0.0
        next_state = self.current_state.copy()
        
        if self.api_client:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # For async context, return dummy response
                    reward = 0.0
                else:
                    response = loop.run_until_complete(
                        self.api_client.send_action(
                            self.agent_id,
                            {
                                "action_type": action_type,
                                **action_params
                            },
                            self.session_id
                        )
                    )
                    reward = response.get("reward", 0.0)
                    done = response.get("done", False)
                    next_state_raw = response.get("next_state", {})
                    next_state = self._parse_state(next_state_raw)
                    self.done = done
            except:
                reward = 0.0
        
        # Check truncation
        self.truncated = self.current_step >= self.max_episode_steps
        
        # Update episode reward
        self.episode_reward += reward
        
        # Info dictionary
        info = {
            "episode_step": self.current_step,
            "episode_reward": self.episode_reward,
            "action_type": action_type,
            "action_params": action_params,
        }
        
        return next_state, reward, self.done, self.truncated, info
    
    def _parse_state(self, raw_state: Dict[str, Any]) -> np.ndarray:
        """Parse raw JSON state to 128-dimensional feature vector."""
        # This is a simplified parser - in production, use StateParser
        features = np.zeros(128, dtype=np.float32)
        
        try:
            player = raw_state.get("player", {})
            stats = player.get("stats", {})
            position = player.get("position", {})
            
            # Self features (0-19)
            features[0] = stats.get("health", 0.0)
            features[1] = stats.get("max_health", GameConfig.MAX_HEALTH)
            features[2] = stats.get("energy", 0.0)
            features[3] = stats.get("max_energy", GameConfig.MAX_ENERGY)
            features[4] = stats.get("shield", 0.0)
            features[5] = position.get("x", 0.0)
            features[6] = position.get("y", 0.0)
            features[10] = stats.get("level", 1)
            features[18] = stats.get("kills", 0)
            features[19] = stats.get("deaths", 0)
            
            # Game state features (120-127)
            game_info = raw_state.get("game", {})
            features[120] = game_info.get("score", 0.0)
            features[121] = game_info.get("enemy_score", 0.0)
            features[122] = stats.get("kills", 0)
            features[123] = stats.get("deaths", 0)
            features[124] = stats.get("assists", 0)
            features[125] = stats.get("damage_dealt", 0.0)
            features[126] = stats.get("damage_taken", 0.0)
            features[127] = raw_state.get("action_count", 0)
            
        except Exception:
            pass
        
        return features
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            print(f"Step: {self.current_step}, Reward: {self.episode_reward:.2f}")
            print(f"Done: {self.done}, Truncated: {self.truncated}")
    
    def close(self):
        """Clean up environment resources."""
        pass
    
    def get_action_meanings(self) -> list:
        """Get human-readable action meanings."""
        meanings = []
        for action_type, action_params in self._simplified_actions:
            if action_params:
                meaning = f"{action_type}_{action_params}"
            else:
                meaning = action_type
            meanings.append(meaning)
        return meanings
