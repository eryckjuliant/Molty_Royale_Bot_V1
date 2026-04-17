"""Game loop for Molty Royale Self-Learning Bot."""

import asyncio
from typing import Dict, Any, Optional
import numpy as np

from src.api_client import MoltyRoyaleClient
from src.state_parser import StateParser
from src.strategy.rule_based import RuleBasedAgent
from src.utils.logger import GameLogger
from src.constants import (
    ACTION_TO_IDX,
    IDX_TO_ACTION,
    TrainingConfig,
)


class GameLoop:
    """Main game loop that connects to API, plays games, and collects experience."""
    
    def __init__(
        self,
        agent_type: str = "rule_based",
        config_path: str = "config/config.yaml",
        secrets_path: str = "config/secrets.yaml",
        turn_duration: float = 60.0,
        use_erc8004: bool = False
    ):
        self.agent_type = agent_type
        self.turn_duration = turn_duration
        self.use_erc8004 = use_erc8004
        
        # Initialize components
        self.client = MoltyRoyaleClient(config_path, secrets_path)
        self.parser = StateParser()
        self.logger = GameLogger()
        
        # Initialize agent
        if agent_type == "rule_based":
            self.agent = RuleBasedAgent()
        else:
            # Placeholder for RL agent
            self.agent = RuleBasedAgent()  # Will be replaced by RL agent
        
        # Game state
        self.agent_id: Optional[str] = None
        self.session_id: Optional[str] = None
        self.is_running = False
        self.episode_count = 0
        self.total_steps = 0
        
        # Replay buffer (will be connected to ML component)
        self.replay_buffer = None
        
        self.logger.info(f"GameLoop initialized with {agent_type} agent")
    
    async def connect(self) -> bool:
        """Connect to Molty Royale API and register agent."""
        try:
            self.logger.info("Connecting to Molty Royale API...")
            
            # Register agent
            agent_name = f"MoltyBot_{self.agent_type}_{self.episode_count}"
            registration = await self.client.register_agent(
                agent_name=agent_name,
                agent_type=self.agent_type,
                description="Self-learning Molty Royale bot"
            )
            
            self.agent_id = registration.get("agent_id")
            self.logger.success(f"Agent registered: {self.agent_id}")
            
            # ERC-8004 identity registration (optional)
            if self.use_erc8004:
                await self._register_erc8004_identity()
            
            return True
        
        except Exception as e:
            self.logger.log_error_with_context(e, "connect")
            return False
    
    async def _register_erc8004_identity(self) -> None:
        """Register ERC-8004 on-chain identity (optional)."""
        try:
            self.logger.info("Registering ERC-8004 identity...")
            # Placeholder for ERC-8004 registration
            # This would use src/utils/onchain.py
            from src.utils.onchain import OnChainManager
            onchain = OnChainManager()
            await onchain.register_identity(self.agent_id)
            self.logger.success("ERC-8004 identity registered")
        except Exception as e:
            self.logger.warning(f"ERC-8004 registration failed: {e}")
    
    async def join_game(self, game_id: Optional[str] = None, mode: str = "standard") -> bool:
        """Join a game room."""
        try:
            target_game_id = game_id or self.client.game_id
            self.logger.info(f"Joining game: {target_game_id} (mode: {mode})")
            
            session_info = await self.client.join_game(
                game_id=target_game_id,
                agent_id=self.agent_id,
                mode=mode
            )
            
            self.session_id = session_info.get("session_id")
            self.logger.success(f"Joined game session: {self.session_id}")
            
            return True
        
        except Exception as e:
            self.logger.log_error_with_context(e, "join_game")
            return False
    
    async def game_step(self) -> Optional[Dict[str, Any]]:
        """Execute one game step: get state → parse → choose action → send."""
        try:
            # Get current state
            raw_state = await self.client.get_state(
                agent_id=self.agent_id,
                game_session_id=self.session_id
            )
            
            # Log raw state
            self.logger.log_state(raw_state)
            
            # Parse state to features
            features = self.parser.parse(raw_state)
            
            # Choose action
            action_type, action_params = self.agent.choose_action(raw_state)
            
            # Log action
            self.logger.log_action(action_type, str(action_params))
            
            # Send action to API
            action_response = await self.client.send_action(
                agent_id=self.agent_id,
                action={
                    "action_type": action_type,
                    **action_params
                },
                game_session_id=self.session_id
            )
            
            # Extract reward and done flag
            reward = action_response.get("reward", 0.0)
            done = action_response.get("done", False)
            next_state = action_response.get("next_state", raw_state)
            
            # Log reward
            self.logger.log_reward(reward, self.episode_count, self.total_steps)
            
            # Save experience to replay buffer
            if self.replay_buffer:
                action_idx = self.agent.get_action_index(action_type, action_params)
                next_features = self.parser.parse(next_state)
                
                self.replay_buffer.add(
                    state=features,
                    action=action_idx,
                    reward=reward,
                    next_state=next_features,
                    done=done
                )
            
            self.total_steps += 1
            
            return {
                "state": features,
                "action": (action_type, action_params),
                "reward": reward,
                "done": done,
                "next_state": next_state
            }
        
        except Exception as e:
            self.logger.log_error_with_context(e, "game_step")
            return None
    
    async def run_episode(self, max_steps: int = 1000) -> Dict[str, Any]:
        """Run a complete episode until game ends or max steps reached."""
        self.episode_count += 1
        self.logger.info(f"Starting episode {self.episode_count}")
        
        episode_data = {
            "episode_num": self.episode_count,
            "steps": 0,
            "total_reward": 0.0,
            "kills": 0,
            "deaths": 0,
            "win": False
        }
        
        step = 0
        while step < max_steps:
            # Execute step
            result = await self.game_step()
            
            if result is None:
                self.logger.error("Game step failed, ending episode")
                break
            
            episode_data["total_reward"] += result["reward"]
            episode_data["steps"] += 1
            step += 1
            
            # Check if episode ended
            if result["done"]:
                # Extract final stats
                final_state = result["next_state"]
                player_stats = final_state.get("player", {}).get("stats", {})
                episode_data["kills"] = player_stats.get("kills", 0)
                episode_data["deaths"] = player_stats.get("deaths", 0)
                episode_data["win"] = final_state.get("game", {}).get("winner") == self.agent_id
                
                self.logger.info(f"Episode {self.episode_count} ended after {step} steps")
                self.logger.info(f"Total reward: {episode_data['total_reward']:.2f}")
                self.logger.info(f"Kills: {episode_data['kills']}, Deaths: {episode_data['deaths']}")
                self.logger.info(f"Win: {episode_data['win']}")
                break
            
            # Wait for turn duration
            await asyncio.sleep(self.turn_duration)
        
        # Log episode summary
        self.logger.log_episode(episode_data)
        
        # Reset agent for next episode
        self.agent.reset()
        
        return episode_data
    
    async def run(self, num_episodes: int = 10, max_steps: int = 1000) -> None:
        """Run multiple episodes."""
        self.is_running = True
        self.logger.info(f"Starting game loop for {num_episodes} episodes")
        
        try:
            # Connect and register
            if not await self.connect():
                self.logger.error("Failed to connect to API")
                return
            
            # Join game
            if not await self.join_game():
                self.logger.error("Failed to join game")
                return
            
            # Run episodes
            for episode in range(num_episodes):
                if not self.is_running:
                    self.logger.info("Game loop stopped")
                    break
                
                await self.run_episode(max_steps)
                
                # Check if we need to rejoin
                if episode < num_episodes - 1:
                    await asyncio.sleep(1.0)  # Brief pause between episodes
        
        except KeyboardInterrupt:
            self.logger.info("Game loop interrupted by user")
        except Exception as e:
            self.logger.log_error_with_context(e, "run")
        finally:
            await self.stop()
    
    async def stop(self) -> None:
        """Stop the game loop and cleanup."""
        self.is_running = False
        self.logger.info("Stopping game loop...")
        
        try:
            # Leave game session if active
            if self.session_id:
                await self.client.leave_game(self.session_id, self.agent_id)
                self.logger.info("Left game session")
        
        except Exception as e:
            self.logger.log_error_with_context(e, "stop")
        
        # Close client
        await self.client.close()
        
        # Close logger
        self.logger.close()
    
    def set_replay_buffer(self, replay_buffer) -> None:
        """Set the replay buffer for experience storage."""
        self.replay_buffer = replay_buffer
        self.logger.info("Replay buffer connected")
    
    def set_api_client(self, client) -> None:
        """Set the API client (for external client from main.py)."""
        self.client = client
        self.logger.info("API client connected")
    
    def set_agent(self, agent) -> None:
        """Set the agent (for switching between rule-based and RL)."""
        self.agent = agent
        self.agent_type = getattr(agent, "__class__.__name__", "custom")
        self.logger.info(f"Agent switched to {self.agent_type}")
