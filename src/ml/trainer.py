"""AutoTrainer for self-learning Molty Royale bot."""

import asyncio
import os
import time
from typing import Optional, Dict, Any
from pathlib import Path
import threading

from src.ml.model import MoltyRoyalePPO
from src.ml.replay_buffer import ReplayBuffer
from src.ml.evaluator import ModelEvaluator, EvaluationResult
from src.ml.environment import MoltyRoyaleEnv
from src.constants import TrainingConfig
from src.utils.logger import GameLogger


class AutoTrainer:
    """Self-learning trainer that automatically trains, evaluates, and deploys models."""
    
    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        save_path: str = "data/models",
        train_interval_games: int = 20,
        train_duration_minutes: int = 45,
        win_rate_threshold: float = 0.05,
        log_interval: int = 10
    ):
        self.replay_buffer = replay_buffer
        self.save_path = Path(save_path)
        self.train_interval_games = train_interval_games
        self.train_duration_minutes = train_duration_minutes
        self.win_rate_threshold = win_rate_threshold
        self.log_interval = log_interval
        
        # Create directories
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.models_path = self.save_path
        self.best_model_path = self.save_path / "best_model.zip"
        
        # Logger
        self.logger = GameLogger()
        
        # Components
        self.model: Optional[MoltyRoyalePPO] = None
        self.evaluator: Optional[ModelEvaluator] = None
        self.env: Optional[MoltyRoyaleEnv] = None
        
        # Training state
        self.games_since_last_train = 0
        self.is_training = False
        self.is_running = False
        self.training_thread: Optional[threading.Thread] = None
        self.current_model_path: Optional[str] = None
        
        # Statistics
        self.best_win_rate = 0.0
        self.training_count = 0
        self.total_training_time = 0.0
        
        # Callback for game loop
        self.on_model_update = None
        
        self.logger.info("AutoTrainer initialized")
    
    def set_model_update_callback(self, callback):
        """Set callback to notify when model is updated."""
        self.on_model_update = callback
    
    async def run(self):
        """Run the auto trainer loop."""
        self.is_running = True
        self.logger.info("AutoTrainer started")
        
        # Initialize components
        self._initialize_components()
        
        # Load best model if exists
        if self.best_model_path.exists():
            self._load_best_model()
        
        while self.is_running:
            # Check if we should trigger training
            if self.games_since_last_train >= self.train_interval_games:
                if not self.is_training:
                    self.logger.info(f"Training triggered after {self.games_since_last_train} games")
                    await self.trigger_training()
            
            # Sleep for a bit
            await asyncio.sleep(10)
        
        self.logger.info("AutoTrainer stopped")
    
    async def trigger_training(self):
        """Trigger a training session."""
        if self.is_training:
            self.logger.warning("Training already in progress, skipping")
            return
        
        # Check if we have enough data
        if len(self.replay_buffer) < TrainingConfig.BATCH_SIZE * 10:
            self.logger.warning(f"Not enough data in replay buffer: {len(self.replay_buffer)}")
            return
        
        # Start training in background thread
        self.is_training = True
        self.training_thread = threading.Thread(target=self._training_loop)
        self.training_thread.start()
    
    def _training_loop(self):
        """Training loop running in background thread."""
        try:
            self.logger.info(f"Starting training session (duration: {self.train_duration_minutes} min)")
            start_time = time.time()
            
            # Create new model for training
            self.model = MoltyRoyalePPO(
                env=self.env,
                verbose=1,
                tensorboard_log=str(self.save_path / "tensorboard")
            )
            
            # Load previous best model if exists for fine-tuning
            if self.best_model_path.exists():
                self.logger.info("Loading best model for fine-tuning")
                self.model.load(str(self.best_model_path))
            
            # Train for specified duration
            total_timesteps = self._calculate_timesteps()
            
            self.logger.info(f"Training for {total_timesteps} timesteps")
            self.model.train(
                total_timesteps=total_timesteps,
                save_path=str(self.save_path),
                save_interval=self.log_interval * 100
            )
            
            # Save new model
            new_model_path = self.save_path / f"model_{self.training_count}.zip"
            self.model.save(str(new_model_path))
            self.logger.success(f"New model saved: {new_model_path}")
            
            # Evaluate new model
            self.logger.info("Evaluating new model...")
            evaluation_result = self._evaluate_new_model(new_model_path)
            
            # Compare with best model
            if self._should_replace_best(evaluation_result):
                self._replace_best_model(new_model_path, evaluation_result)
            else:
                self.logger.info("New model did not improve win rate, keeping best model")
            
            # Update statistics
            training_time = time.time() - start_time
            self.total_training_time += training_time
            self.training_count += 1
            self.games_since_last_train = 0
            
            self.logger.info(f"Training session completed in {training_time:.2f}s")
            
        except Exception as e:
            self.logger.log_error_with_context(e, "training_loop")
        finally:
            self.is_training = False
    
    def _calculate_timesteps(self) -> int:
        """Calculate number of timesteps based on training duration."""
        # Estimate: ~1000 timesteps per minute (conservative estimate)
        timesteps_per_minute = 1000
        return int(self.train_duration_minutes * timesteps_per_minute)
    
    def _initialize_components(self):
        """Initialize ML components."""
        if self.env is None:
            self.env = MoltyRoyaleEnv()
        
        if self.evaluator is None:
            self.evaluator = ModelEvaluator(
                save_path=str(self.save_path),
                log_path="data/logs/evaluation"
            )
    
    def _load_best_model(self):
        """Load the best model."""
        if self.best_model_path.exists():
            self.logger.info(f"Loading best model from {self.best_model_path}")
            self.model = MoltyRoyalePPO(env=self.env)
            self.model.load(str(self.best_model_path))
            self.current_model_path = str(self.best_model_path)
            
            # Evaluate to get baseline win rate
            result = self.evaluator.evaluate_model(
                self.model,
                num_episodes=5,  # Quick evaluation
                opponent="rule_based",
                verbose=False
            )
            self.best_win_rate = result.win_rate
            self.logger.info(f"Best model baseline win rate: {self.best_win_rate:.2%}")
        else:
            self.logger.info("No best model found, starting fresh")
    
    def _evaluate_new_model(self, model_path: str) -> EvaluationResult:
        """Evaluate a new model."""
        # Load new model
        new_model = MoltyRoyalePPO(env=self.env)
        new_model.load(model_path)
        
        # Evaluate against rule-based agent
        result = self.evaluator.evaluate_model(
            new_model,
            num_episodes=TrainingConfig.NUM_EVAL_EPISODES,
            opponent="rule_based",
            shadow_mode=True,
            verbose=True
        )
        
        return result
    
    def _should_replace_best(self, result: EvaluationResult) -> bool:
        """Determine if new model should replace best model."""
        if not self.best_model_path.exists():
            # No best model exists, use this one
            return True
        
        # Check win rate improvement
        win_rate_improvement = result.win_rate - self.best_win_rate
        
        if win_rate_improvement > self.win_rate_threshold:
            self.logger.info(f"Win rate improved by {win_rate_improvement:.2%} (threshold: {self.win_rate_threshold:.2%})")
            return True
        
        return False
    
    def _replace_best_model(self, new_model_path: Path, result: EvaluationResult):
        """Replace best model with new model."""
        # Backup old best model
        if self.best_model_path.exists():
            backup_path = self.save_path / f"best_model_backup_{self.training_count}.zip"
            self.best_model_path.rename(backup_path)
            self.logger.info(f"Old best model backed up to {backup_path}")
        
        # Copy new model to best model
        import shutil
        shutil.copy(new_model_path, self.best_model_path)
        
        # Update statistics
        self.best_win_rate = result.win_rate
        self.current_model_path = str(self.best_model_path)
        
        self.logger.success(f"Best model replaced! New win rate: {self.best_win_rate:.2%}")
        
        # Notify game loop to load new model
        if self.on_model_update:
            self.on_model_update(str(self.best_model_path))
    
    def increment_game_count(self):
        """Increment game counter (called by game loop)."""
        self.games_since_last_train += 1
        
        if self.games_since_last_train % self.log_interval == 0:
            self.logger.info(f"Games since last train: {self.games_since_last_train}/{self.train_interval_games}")
    
    def get_current_model_path(self) -> Optional[str]:
        """Get the current best model path."""
        return self.current_model_path
    
    def get_model(self) -> Optional[MoltyRoyalePPO]:
        """Get the current model instance."""
        return self.model
    
    def get_stats(self) -> Dict[str, Any]:
        """Get trainer statistics."""
        return {
            "games_since_last_train": self.games_since_last_train,
            "train_interval_games": self.train_interval_games,
            "is_training": self.is_training,
            "training_count": self.training_count,
            "total_training_time": self.total_training_time,
            "best_win_rate": self.best_win_rate,
            "current_model_path": self.current_model_path,
            "replay_buffer_size": len(self.replay_buffer),
            "replay_buffer_usage": len(self.replay_buffer) / self.replay_buffer.capacity,
        }
    
    async def stop(self):
        """Stop the auto trainer."""
        self.is_running = False
        self.logger.info("Stopping AutoTrainer...")
        
        # Wait for training thread to finish
        if self.training_thread and self.training_thread.is_alive():
            self.logger.info("Waiting for training thread to finish...")
            self.training_thread.join(timeout=300)  # 5 minute timeout
        
        # Save final state
        if self.model:
            final_path = self.save_path / "model_final.zip"
            self.model.save(str(final_path))
            self.logger.info(f"Final model saved to {final_path}")
        
        self.logger.info("AutoTrainer stopped")
    
    def force_training(self):
        """Force trigger training immediately."""
        self.logger.info("Force training triggered")
        # This would be called from a different context, need to handle async
        # For now, just increment to trigger on next loop
        self.games_since_last_train = self.train_interval_games


class TrainingManager:
    """Manages the training lifecycle and coordinates with game loop."""
    
    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        save_path: str = "data/models",
        train_interval_games: int = 20,
        train_duration_minutes: int = 45
    ):
        self.trainer = AutoTrainer(
            replay_buffer=replay_buffer,
            save_path=save_path,
            train_interval_games=train_interval_games,
            train_duration_minutes=train_duration_minutes
        )
        
        self.game_loop = None
    
    def set_game_loop(self, game_loop):
        """Set reference to game loop."""
        self.game_loop = game_loop
        
        # Set callback for model updates
        self.trainer.set_model_update_callback(self._on_model_updated)
    
    def _on_model_updated(self, model_path: str):
        """Handle model update event."""
        if self.game_loop:
            # Load new model in game loop
            from src.ml.model import MoltyRoyalePPO
            new_model = MoltyRoyalePPO()
            new_model.load(model_path)
            self.game_loop.set_agent(new_model)
            
            self.trainer.logger.success(f"Game loop updated with new model: {model_path}")
    
    async def run(self):
        """Run the training manager."""
        await self.trainer.run()
    
    async def stop(self):
        """Stop the training manager."""
        await self.trainer.stop()
    
    def notify_game_completed(self):
        """Notify trainer that a game completed."""
        self.trainer.increment_game_count()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return self.trainer.get_stats()
