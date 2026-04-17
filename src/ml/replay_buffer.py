"""Replay buffer for storing experience data in Parquet/HDF5 format."""

import os
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

try:
    import h5py
    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False

from src.constants import TrainingConfig


class ReplayBuffer:
    """Replay buffer for storing experience (state, action, reward, next_state, done)."""
    
    def __init__(
        self,
        capacity: int = TrainingConfig.BUFFER_SIZE,
        state_dim: int = 128,
        action_dim: int = 1,
        save_path: str = "data/replay",
        storage_format: str = "parquet",  # "parquet" or "hdf5"
        auto_save: bool = True,
        save_interval: int = 1000
    ):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_path = Path(save_path)
        self.storage_format = storage_format
        self.auto_save = auto_save
        self.save_interval = save_interval
        
        # Create save directory
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize buffers
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        # Tracking
        self.size = 0
        self.pointer = 0
        self.total_added = 0
        
        # Validate storage format
        if storage_format == "parquet" and not HAS_PARQUET:
            print("Warning: Parquet not available, falling back to in-memory only")
            self.storage_format = "memory"
        elif storage_format == "hdf5" and not HAS_HDF5:
            print("Warning: HDF5 not available, falling back to in-memory only")
            self.storage_format = "memory"
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add a transition to the replay buffer."""
        # Store at current pointer
        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.next_states[self.pointer] = next_state
        self.dones[self.pointer] = done
        
        # Update pointer and size
        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_added += 1
        
        # Auto-save if enabled
        if self.auto_save and self.total_added % self.save_interval == 0:
            self.save()
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
        }
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size
    
    def save(self, filename: Optional[str] = None) -> str:
        """Save buffer to disk."""
        if filename is None:
            filename = f"replay_buffer_{self.total_added}"
        
        filepath = self.save_path / f"{filename}.{self.storage_format}"
        
        if self.storage_format == "parquet" and HAS_PARQUET:
            self._save_parquet(filepath)
        elif self.storage_format == "hdf5" and HAS_HDF5:
            self._save_hdf5(filepath)
        else:
            # Fallback to numpy
            self._save_numpy(filepath.with_suffix(".npz"))
        
        print(f"Replay buffer saved to {filepath}")
        return str(filepath)
    
    def _save_parquet(self, filepath: Path) -> None:
        """Save buffer to Parquet format."""
        # Create DataFrame
        df = pd.DataFrame({
            "state_idx": np.arange(self.size),
            "action": self.actions[:self.size].flatten(),
            "reward": self.rewards[:self.size],
            "done": self.dones[:self.size],
        })
        
        # Save states and next_states as separate columns
        for i in range(self.state_dim):
            df[f"state_{i}"] = self.states[:self.size, i]
            df[f"next_state_{i}"] = self.next_states[:self.size, i]
        
        # Write to Parquet
        table = pa.Table.from_pandas(df)
        pq.write_table(table, filepath)
    
    def _save_hdf5(self, filepath: Path) -> None:
        """Save buffer to HDF5 format."""
        with h5py.File(filepath, 'w') as f:
            f.create_dataset("states", data=self.states[:self.size])
            f.create_dataset("actions", data=self.actions[:self.size])
            f.create_dataset("rewards", data=self.rewards[:self.size])
            f.create_dataset("next_states", data=self.next_states[:self.size])
            f.create_dataset("dones", data=self.dones[:self.size])
            f.attrs["size"] = self.size
            f.attrs["pointer"] = self.pointer
            f.attrs["total_added"] = self.total_added
    
    def _save_numpy(self, filepath: Path) -> None:
        """Save buffer to NumPy format."""
        np.savez_compressed(
            filepath,
            states=self.states[:self.size],
            actions=self.actions[:self.size],
            rewards=self.rewards[:self.size],
            next_states=self.next_states[:self.size],
            dones=self.dones[:self.size],
            size=self.size,
            pointer=self.pointer,
            total_added=self.total_added
        )
    
    def load(self, filepath: str) -> None:
        """Load buffer from disk."""
        filepath = Path(filepath)
        
        if filepath.suffix == ".parquet" and HAS_PARQUET:
            self._load_parquet(filepath)
        elif filepath.suffix == ".h5" and HAS_HDF5:
            self._load_hdf5(filepath)
        elif filepath.suffix == ".npz":
            self._load_numpy(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        print(f"Replay buffer loaded from {filepath}")
    
    def _load_parquet(self, filepath: Path) -> None:
        """Load buffer from Parquet format."""
        table = pq.read_table(filepath)
        df = table.to_pandas()
        
        self.size = len(df)
        
        # Load states and next_states
        for i in range(self.state_dim):
            self.states[:self.size, i] = df[f"state_{i}"].values
            self.next_states[:self.size, i] = df[f"next_state_{i}"].values
        
        # Load other data
        self.actions[:self.size] = df["action"].values.reshape(-1, 1)
        self.rewards[:self.size] = df["reward"].values
        self.dones[:self.size] = df["done"].values
    
    def _load_hdf5(self, filepath: Path) -> None:
        """Load buffer from HDF5 format."""
        with h5py.File(filepath, 'r') as f:
            self.states[:f.attrs["size"]] = f["states"][:]
            self.actions[:f.attrs["size"]] = f["actions"][:]
            self.rewards[:f.attrs["size"]] = f["rewards"][:]
            self.next_states[:f.attrs["size"]] = f["next_states"][:]
            self.dones[:f.attrs["size"]] = f["dones"][:]
            self.size = f.attrs["size"]
            self.pointer = f.attrs["pointer"]
            self.total_added = f.attrs["total_added"]
    
    def _load_numpy(self, filepath: Path) -> None:
        """Load buffer from NumPy format."""
        data = np.load(filepath)
        self.states[:data["size"]] = data["states"]
        self.actions[:data["size"]] = data["actions"]
        self.rewards[:data["size"]] = data["rewards"]
        self.next_states[:data["size"]] = data["next_states"]
        self.dones[:data["size"]] = data["dones"]
        self.size = int(data["size"])
        self.pointer = int(data["pointer"])
        self.total_added = int(data["total_added"])
    
    def clear(self) -> None:
        """Clear the replay buffer."""
        self.states.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.next_states.fill(0)
        self.dones.fill(False)
        self.size = 0
        self.pointer = 0
        self.total_added = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            "size": self.size,
            "capacity": self.capacity,
            "pointer": self.pointer,
            "total_added": self.total_added,
            "usage": self.size / self.capacity,
            "mean_reward": np.mean(self.rewards[:self.size]) if self.size > 0 else 0.0,
            "std_reward": np.std(self.rewards[:self.size]) if self.size > 0 else 0.0,
        }


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized experience replay buffer."""
    
    def __init__(
        self,
        capacity: int = TrainingConfig.BUFFER_SIZE,
        state_dim: int = 128,
        action_dim: int = 1,
        save_path: str = "data/replay",
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001
    ):
        super().__init__(capacity, state_dim, action_dim, save_path)
        
        # Prioritization parameters
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add a transition with maximum priority."""
        super().add(state, action, reward, next_state, done)
        self.priorities[self.pointer - 1] = self.max_priority
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a batch with prioritization."""
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # Calculate importance weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
            "indices": indices,
            "weights": weights.astype(np.float32),
        }
        
        return batch
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities based on TD errors."""
        priorities = np.abs(td_errors) + 1e-6
        self.priorities[indices] = priorities ** self.alpha
        self.max_priority = max(self.max_priority, priorities.max())
