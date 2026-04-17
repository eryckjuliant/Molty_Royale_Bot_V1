"""Evaluator for testing model win-rate in shadow mode."""

import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from src.ml.model import MoltyRoyalePPO
from src.ml.environment import MoltyRoyaleEnv
from src.strategy.rule_based import RuleBasedAgent
from src.constants import TrainingConfig


@dataclass
class EvaluationResult:
    """Result of model evaluation."""
    model_name: str
    opponent_name: str
    total_episodes: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    avg_reward: float
    avg_episode_length: float
    total_kills: int
    total_deaths: int
    kda_ratio: float


class ModelEvaluator:
    """Evaluate model performance in shadow mode."""
    
    def __init__(
        self,
        save_path: str = "data/models",
        log_path: str = "data/logs/evaluation"
    ):
        self.save_path = save_path
        self.log_path = log_path
        self.results_history: List[EvaluationResult] = []
    
    def evaluate_model(
        self,
        model: MoltyRoyalePPO,
        num_episodes: int = TrainingConfig.NUM_EVAL_EPISODES,
        opponent: Optional[str] = "rule_based",
        shadow_mode: bool = True,
        verbose: bool = True
    ) -> EvaluationResult:
        """Evaluate model against opponent in shadow mode.
        
        Args:
            model: The model to evaluate
            num_episodes: Number of episodes to run
            opponent: Opponent type ("rule_based", "random", or previous model)
            shadow_mode: If True, run in shadow mode (no real API calls)
            verbose: Whether to print progress
            
        Returns:
            EvaluationResult with metrics
        """
        if verbose:
            print(f"Evaluating model against {opponent} for {num_episodes} episodes...")
        
        # Initialize environment
        env = model.get_env()
        
        # Initialize opponent
        if opponent == "rule_based":
            opponent_agent = RuleBasedAgent()
        elif opponent == "random":
            opponent_agent = RandomAgent()
        else:
            # Load previous model as opponent
            opponent_agent = MoltyRoyalePPO()
            opponent_agent.load(opponent)
        
        # Metrics
        total_reward = 0.0
        total_steps = 0
        wins = 0
        losses = 0
        draws = 0
        total_kills = 0
        total_deaths = 0
        
        # Run episodes
        for episode in range(num_episodes):
            if verbose and episode % 10 == 0:
                print(f"Episode {episode}/{num_episodes}")
            
            # Run episode
            episode_reward, episode_steps, episode_won, kills, deaths = self._run_episode(
                model,
                opponent_agent,
                env,
                shadow_mode
            )
            
            # Update metrics
            total_reward += episode_reward
            total_steps += episode_steps
            total_kills += kills
            total_deaths += deaths
            
            if episode_won:
                wins += 1
            elif episode_won is False:
                losses += 1
            else:
                draws += 1
        
        # Calculate statistics
        win_rate = wins / num_episodes if num_episodes > 0 else 0.0
        avg_reward = total_reward / num_episodes if num_episodes > 0 else 0.0
        avg_episode_length = total_steps / num_episodes if num_episodes > 0 else 0.0
        kda_ratio = total_kills / max(total_deaths, 1)
        
        # Create result
        result = EvaluationResult(
            model_name=model.__class__.__name__,
            opponent_name=str(opponent),
            total_episodes=num_episodes,
            wins=wins,
            losses=losses,
            draws=draws,
            win_rate=win_rate,
            avg_reward=avg_reward,
            avg_episode_length=avg_episode_length,
            total_kills=total_kills,
            total_deaths=total_deaths,
            kda_ratio=kda_ratio
        )
        
        # Store result
        self.results_history.append(result)
        
        if verbose:
            self._print_result(result)
        
        return result
    
    def _run_episode(
        self,
        model: MoltyRoyalePPO,
        opponent_agent,
        env: MoltyRoyaleEnv,
        shadow_mode: bool
    ) -> Tuple[float, int, Optional[bool], int, int]:
        """Run a single evaluation episode."""
        obs, info = env.reset()
        
        episode_reward = 0.0
        episode_steps = 0
        done = False
        truncated = False
        
        kills = 0
        deaths = 0
        
        while not done and not truncated:
            # Model action
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            next_obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            
            # Extract stats if available
            kills += info.get("kills", 0)
            deaths += info.get("deaths", 0)
            
            obs = next_obs
        
        # Determine winner (simplified)
        # In real implementation, this would come from game state
        episode_won = episode_reward > 0 if episode_reward != 0 else None
        
        return episode_reward, episode_steps, episode_won, kills, deaths
    
    def _print_result(self, result: EvaluationResult) -> None:
        """Print evaluation result."""
        print("\n" + "="*60)
        print(f"Evaluation Result: {result.model_name} vs {result.opponent_name}")
        print("="*60)
        print(f"Episodes: {result.total_episodes}")
        print(f"Win Rate: {result.win_rate:.2%}")
        print(f"Wins: {result.wins} | Losses: {result.losses} | Draws: {result.draws}")
        print(f"Average Reward: {result.avg_reward:.2f}")
        print(f"Average Episode Length: {result.avg_episode_length:.2f}")
        print(f"K/D/A Ratio: {result.kda_ratio:.2f}")
        print("="*60 + "\n")
    
    def compare_models(
        self,
        model1: MoltyRoyalePPO,
        model2: MoltyRoyalePPO,
        num_episodes: int = 50,
        verbose: bool = True
    ) -> Tuple[EvaluationResult, EvaluationResult]:
        """Compare two models against each other."""
        if verbose:
            print("Comparing two models...")
        
        # Evaluate model1 vs model2
        result1 = self.evaluate_model(
            model1,
            num_episodes=num_episodes,
            opponent="model2_placeholder",  # Would use model2 path
            verbose=False
        )
        
        # Evaluate model2 vs model1
        result2 = self.evaluate_model(
            model2,
            num_episodes=num_episodes,
            opponent="model1_placeholder",  # Would use model1 path
            verbose=False
        )
        
        if verbose:
            print("\nModel Comparison:")
            print(f"{model1.__class__.__name__}: {result1.win_rate:.2%} win rate")
            print(f"{model2.__class__.__name__}: {result2.win_rate:.2%} win rate")
            
            if result1.win_rate > result2.win_rate:
                print(f"{model1.__class__.__name__} performs better")
            elif result2.win_rate > result1.win_rate:
                print(f"{model2.__class__.__name__} performs better")
            else:
                print("Models perform similarly")
        
        return result1, result2
    
    def get_best_model(self) -> Optional[EvaluationResult]:
        """Get the best performing model from history."""
        if not self.results_history:
            return None
        
        return max(self.results_history, key=lambda x: x.win_rate)
    
    def save_results(self, filepath: str) -> None:
        """Save evaluation results to file."""
        import json
        from dataclasses import asdict
        
        results_data = [asdict(r) for r in self.results_history]
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Evaluation results saved to {filepath}")


class RandomAgent:
    """Random agent for baseline comparison."""
    
    def __init__(self, num_actions: int = 15):
        self.num_actions = num_actions
    
    def predict(self, obs: np.ndarray) -> Tuple[int, None]:
        """Return random action."""
        action = np.random.randint(0, self.num_actions)
        return action, None


def evaluate_model(
    model_path: str,
    num_episodes: int = 50,
    opponent: str = "rule_based",
    save_path: str = "data/models"
) -> EvaluationResult:
    """Convenience function to evaluate a saved model."""
    # Load model
    model = MoltyRoyalePPO()
    model.load(model_path)
    
    # Evaluate
    evaluator = ModelEvaluator(save_path=save_path)
    result = evaluator.evaluate_model(
        model,
        num_episodes=num_episodes,
        opponent=opponent
    )
    
    return result
