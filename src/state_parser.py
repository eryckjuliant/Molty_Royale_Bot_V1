"""State parser for converting raw JSON game state into 128-dimensional feature vectors."""

import numpy as np
from typing import Dict, Any, Optional, List
import math

from src.constants import (
    FEATURE_NAMES,
    FEATURE_TO_IDX,
    GameConfig,
)


class StateParser:
    """Parse raw JSON state from Molty Royale API into 128-dimensional feature vector."""
    
    def __init__(self):
        self.feature_dim = 128
        self._validate_feature_count()
    
    def _validate_feature_count(self) -> None:
        """Validate that we have exactly 128 features."""
        assert len(FEATURE_NAMES) == 128, f"Expected 128 features, got {len(FEATURE_NAMES)}"
    
    def parse(self, raw_state: Dict[str, Any]) -> np.ndarray:
        """Parse raw JSON state into 128-dimensional feature vector.
        
        Args:
            raw_state: Raw JSON state from API
            
        Returns:
            numpy array of shape (128,) containing all features
        """
        features = np.zeros(self.feature_dim, dtype=np.float32)
        
        # Parse self features (0-19)
        self._parse_self_features(raw_state, features)
        
        # Parse enemy features (20-39)
        self._parse_enemy_features(raw_state, features)
        
        # Parse teammate features (40-59)
        self._parse_teammate_features(raw_state, features)
        
        # Parse environment features (60-79)
        self._parse_environment_features(raw_state, features)
        
        # Parse resource features (80-99)
        self._parse_resource_features(raw_state, features)
        
        # Parse powerup features (100-114)
        self._parse_powerup_features(raw_state, features)
        
        # Parse objective features (115-119)
        self._parse_objective_features(raw_state, features)
        
        # Parse game state features (120-127)
        self._parse_game_state_features(raw_state, features)
        
        return features
    
    def _parse_self_features(self, state: Dict[str, Any], features: np.ndarray) -> None:
        """Parse self-related features (indices 0-19)."""
        player = state.get("player", {})
        stats = player.get("stats", {})
        
        features[FEATURE_TO_IDX["self_health"]] = stats.get("health", 0.0)
        features[FEATURE_TO_IDX["self_max_health"]] = stats.get("max_health", GameConfig.MAX_HEALTH)
        features[FEATURE_TO_IDX["self_energy"]] = stats.get("energy", 0.0)
        features[FEATURE_TO_IDX["self_max_energy"]] = stats.get("max_energy", GameConfig.MAX_ENERGY)
        features[FEATURE_TO_IDX["self_shield"]] = stats.get("shield", 0.0)
        
        position = player.get("position", {})
        features[FEATURE_TO_IDX["self_x"]] = position.get("x", 0.0)
        features[FEATURE_TO_IDX["self_y"]] = position.get("y", 0.0)
        
        velocity = player.get("velocity", {})
        features[FEATURE_TO_IDX["self_velocity_x"]] = velocity.get("x", 0.0)
        features[FEATURE_TO_IDX["self_velocity_y"]] = velocity.get("y", 0.0)
        
        features[FEATURE_TO_IDX["self_orientation"]] = player.get("orientation", 0.0)
        features[FEATURE_TO_IDX["self_level"]] = stats.get("level", 1)
        features[FEATURE_TO_IDX["self_experience"]] = stats.get("experience", 0.0)
        
        cooldowns = player.get("cooldowns", {})
        features[FEATURE_TO_IDX["self_cooldown_1"]] = cooldowns.get("ability_1", 0.0)
        features[FEATURE_TO_IDX["self_cooldown_2"]] = cooldowns.get("ability_2", 0.0)
        features[FEATURE_TO_IDX["self_cooldown_3"]] = cooldowns.get("ability_3", 0.0)
        features[FEATURE_TO_IDX["self_cooldown_special"]] = cooldowns.get("special", 0.0)
        
        buffs = player.get("buffs", [])
        features[FEATURE_TO_IDX["self_buff_count"]] = len(buffs)
        
        debuffs = player.get("debuffs", [])
        features[FEATURE_TO_IDX["self_debuff_count"]] = len(debuffs)
        
        features[FEATURE_TO_IDX["self_kills"]] = stats.get("kills", 0)
        features[FEATURE_TO_IDX["self_deaths"]] = stats.get("deaths", 0)
    
    def _parse_enemy_features(self, state: Dict[str, Any], features: np.ndarray) -> None:
        """Parse enemy features (indices 20-39)."""
        enemies = state.get("enemies", [])
        self_pos = state.get("player", {}).get("position", {"x": 0.0, "y": 0.0})
        
        for i in range(5):
            if i < len(enemies):
                enemy = enemies[i]
                base_idx = 20 + i * 10
                
                features[base_idx + 0] = enemy.get("health", 0.0)
                features[base_idx + 1] = enemy.get("max_health", GameConfig.MAX_HEALTH)
                features[base_idx + 2] = enemy.get("energy", 0.0)
                
                enemy_pos = enemy.get("position", {"x": 0.0, "y": 0.0})
                features[base_idx + 3] = enemy_pos.get("x", 0.0)
                features[base_idx + 4] = enemy_pos.get("y", 0.0)
                
                # Calculate distance and angle
                dx = enemy_pos.get("x", 0.0) - self_pos.get("x", 0.0)
                dy = enemy_pos.get("y", 0.0) - self_pos.get("y", 0.0)
                distance = math.sqrt(dx * dx + dy * dy)
                angle = math.atan2(dy, dx)
                
                features[base_idx + 5] = distance
                features[base_idx + 6] = angle
                features[base_idx + 7] = enemy.get("threat_level", 0.0)
                features[base_idx + 8] = enemy.get("cooldown", 0.0)
                features[base_idx + 9] = 1.0 if enemy.get("is_alive", True) else 0.0
            else:
                # Pad with zeros for missing enemies
                base_idx = 20 + i * 10
                for j in range(10):
                    features[base_idx + j] = 0.0
    
    def _parse_teammate_features(self, state: Dict[str, Any], features: np.ndarray) -> None:
        """Parse teammate features (indices 40-59)."""
        teammates = state.get("teammates", [])
        self_pos = state.get("player", {}).get("position", {"x": 0.0, "y": 0.0})
        
        for i in range(4):
            if i < len(teammates):
                teammate = teammates[i]
                base_idx = 40 + i * 10
                
                features[base_idx + 0] = teammate.get("health", 0.0)
                features[base_idx + 1] = teammate.get("max_health", GameConfig.MAX_HEALTH)
                features[base_idx + 2] = teammate.get("energy", 0.0)
                
                teammate_pos = teammate.get("position", {"x": 0.0, "y": 0.0})
                features[base_idx + 3] = teammate_pos.get("x", 0.0)
                features[base_idx + 4] = teammate_pos.get("y", 0.0)
                
                # Calculate distance and angle
                dx = teammate_pos.get("x", 0.0) - self_pos.get("x", 0.0)
                dy = teammate_pos.get("y", 0.0) - self_pos.get("y", 0.0)
                distance = math.sqrt(dx * dx + dy * dy)
                angle = math.atan2(dy, dx)
                
                features[base_idx + 5] = distance
                features[base_idx + 6] = angle
                features[base_idx + 7] = teammate.get("cooldown", 0.0)
                features[base_idx + 8] = 1.0 if teammate.get("is_alive", True) else 0.0
                features[base_idx + 9] = 1.0 if teammate.get("health", 0.0) < 30.0 else 0.0  # needs_help
            else:
                # Pad with zeros for missing teammates
                base_idx = 40 + i * 10
                for j in range(10):
                    features[base_idx + j] = 0.0
    
    def _parse_environment_features(self, state: Dict[str, Any], features: np.ndarray) -> None:
        """Parse environment features (indices 60-79)."""
        game_info = state.get("game", {})
        map_info = game_info.get("map", {})
        zone = game_info.get("zone", {})
        weather = game_info.get("weather", {})
        
        features[FEATURE_TO_IDX["map_width"]] = map_info.get("width", GameConfig.MAP_WIDTH)
        features[FEATURE_TO_IDX["map_height"]] = map_info.get("height", GameConfig.MAP_HEIGHT)
        features[FEATURE_TO_IDX["time_remaining"]] = game_info.get("time_remaining", 0.0)
        features[FEATURE_TO_IDX["round_number"]] = game_info.get("round_number", 1)
        features[FEATURE_TO_IDX["total_rounds"]] = game_info.get("total_rounds", GameConfig.MAX_ROUNDS)
        
        features[FEATURE_TO_IDX["zone_x"]] = zone.get("x", 0.0)
        features[FEATURE_TO_IDX["zone_y"]] = zone.get("y", 0.0)
        features[FEATURE_TO_IDX["zone_radius"]] = zone.get("radius", 0.0)
        features[FEATURE_TO_IDX["zone_shrinking"]] = 1.0 if zone.get("shrinking", False) else 0.0
        
        features[FEATURE_TO_IDX["weather_condition"]] = self._encode_weather(weather.get("type", "clear"))
        features[FEATURE_TO_IDX["terrain_type"]] = self._encode_terrain(map_info.get("terrain", "normal"))
        features[FEATURE_TO_IDX["visibility"]] = weather.get("visibility", 1.0)
        features[FEATURE_TO_IDX["day_night_cycle"]] = game_info.get("day_night", 0.0)  # 0-1 cycle
        
        features[FEATURE_TO_IDX["resource_spawn_rate"]] = game_info.get("resource_spawn_rate", 0.0)
        features[FEATURE_TO_IDX["powerup_spawn_rate"]] = game_info.get("powerup_spawn_rate", 0.0)
        features[FEATURE_TO_IDX["hazard_count"]] = len(game_info.get("hazards", []))
        features[FEATURE_TO_IDX["obstacle_count"]] = len(game_info.get("obstacles", []))
        
        # Calculate distances
        self_pos = state.get("player", {}).get("position", {"x": 0.0, "y": 0.0})
        map_w = map_info.get("width", GameConfig.MAP_WIDTH)
        map_h = map_info.get("height", GameConfig.MAP_HEIGHT)
        
        # Distance to safe zone
        zone_x = zone.get("x", 0.0)
        zone_y = zone.get("y", 0.0)
        dx = zone_x - self_pos.get("x", 0.0)
        dy = zone_y - self_pos.get("y", 0.0)
        safe_zone_dist = math.sqrt(dx * dx + dy * dy)
        features[FEATURE_TO_IDX["safe_zone_distance"]] = safe_zone_dist
        
        # Distance to nearest edge
        edge_dist = min(
            self_pos.get("x", 0.0),
            map_w - self_pos.get("x", 0.0),
            self_pos.get("y", 0.0),
            map_h - self_pos.get("y", 0.0)
        )
        features[FEATURE_TO_IDX["nearest_edge_distance"]] = edge_dist
        
        # Distance to center
        center_x = map_w / 2
        center_y = map_h / 2
        dx = center_x - self_pos.get("x", 0.0)
        dy = center_y - self_pos.get("y", 0.0)
        center_dist = math.sqrt(dx * dx + dy * dy)
        features[FEATURE_TO_IDX["center_distance"]] = center_dist
    
    def _parse_resource_features(self, state: Dict[str, Any], features: np.ndarray) -> None:
        """Parse resource features (indices 80-99)."""
        resources = state.get("resources", [])
        self_pos = state.get("player", {}).get("position", {"x": 0.0, "y": 0.0})
        
        for i in range(5):
            if i < len(resources):
                resource = resources[i]
                base_idx = 80 + i * 7
                
                features[base_idx + 0] = self._encode_resource_type(resource.get("type", "coin"))
                
                res_pos = resource.get("position", {"x": 0.0, "y": 0.0})
                features[base_idx + 1] = res_pos.get("x", 0.0)
                features[base_idx + 2] = res_pos.get("y", 0.0)
                
                # Calculate distance
                dx = res_pos.get("x", 0.0) - self_pos.get("x", 0.0)
                dy = res_pos.get("y", 0.0) - self_pos.get("y", 0.0)
                distance = math.sqrt(dx * dx + dy * dy)
                
                features[base_idx + 3] = distance
                features[base_idx + 4] = resource.get("value", 0.0)
                features[base_idx + 5] = resource.get("expires_in", 0.0)
                features[base_idx + 6] = 1.0 if resource.get("is_collected", False) else 0.0
            else:
                # Pad with zeros for missing resources
                base_idx = 80 + i * 7
                for j in range(7):
                    features[base_idx + j] = 0.0
    
    def _parse_powerup_features(self, state: Dict[str, Any], features: np.ndarray) -> None:
        """Parse powerup features (indices 100-114)."""
        powerups = state.get("powerups", [])
        self_pos = state.get("player", {}).get("position", {"x": 0.0, "y": 0.0})
        
        for i in range(3):
            if i < len(powerups):
                powerup = powerups[i]
                base_idx = 100 + i * 6
                
                features[base_idx + 0] = self._encode_powerup_type(powerup.get("type", "speed"))
                
                pu_pos = powerup.get("position", {"x": 0.0, "y": 0.0})
                features[base_idx + 1] = pu_pos.get("x", 0.0)
                features[base_idx + 2] = pu_pos.get("y", 0.0)
                
                # Calculate distance
                dx = pu_pos.get("x", 0.0) - self_pos.get("x", 0.0)
                dy = pu_pos.get("y", 0.0) - self_pos.get("y", 0.0)
                distance = math.sqrt(dx * dx + dy * dy)
                
                features[base_idx + 3] = distance
                features[base_idx + 4] = powerup.get("duration", 0.0)
                features[base_idx + 5] = powerup.get("expires_in", 0.0)
            else:
                # Pad with zeros for missing powerups
                base_idx = 100 + i * 6
                for j in range(6):
                    features[base_idx + j] = 0.0
    
    def _parse_objective_features(self, state: Dict[str, Any], features: np.ndarray) -> None:
        """Parse objective features (indices 115-119)."""
        objective = state.get("objective", {})
        
        features[FEATURE_TO_IDX["objective_type"]] = self._encode_objective_type(objective.get("type", "none"))
        
        obj_pos = objective.get("position", {"x": 0.0, "y": 0.0})
        features[FEATURE_TO_IDX["objective_x"]] = obj_pos.get("x", 0.0)
        features[FEATURE_TO_IDX["objective_y"]] = obj_pos.get("y", 0.0)
        features[FEATURE_TO_IDX["objective_progress"]] = objective.get("progress", 0.0)
        features[FEATURE_TO_IDX["objective_time_left"]] = objective.get("time_left", 0.0)
    
    def _parse_game_state_features(self, state: Dict[str, Any], features: np.ndarray) -> None:
        """Parse game state features (indices 120-127)."""
        game_info = state.get("game", {})
        player_stats = state.get("player", {}).get("stats", {})
        
        features[FEATURE_TO_IDX["score"]] = game_info.get("score", 0.0)
        features[FEATURE_TO_IDX["enemy_score"]] = game_info.get("enemy_score", 0.0)
        features[FEATURE_TO_IDX["kill_count"]] = player_stats.get("kills", 0)
        features[FEATURE_TO_IDX["death_count"]] = player_stats.get("deaths", 0)
        features[FEATURE_TO_IDX["assists"]] = player_stats.get("assists", 0)
        features[FEATURE_TO_IDX["damage_dealt"]] = player_stats.get("damage_dealt", 0.0)
        features[FEATURE_TO_IDX["damage_taken"]] = player_stats.get("damage_taken", 0.0)
        features[FEATURE_TO_IDX["action_count"]] = state.get("action_count", 0)
    
    def _encode_weather(self, weather_type: str) -> float:
        """Encode weather type as float."""
        weather_map = {
            "clear": 0.0,
            "rain": 1.0,
            "storm": 2.0,
            "fog": 3.0,
            "snow": 4.0,
        }
        return weather_map.get(weather_type.lower(), 0.0)
    
    def _encode_terrain(self, terrain_type: str) -> float:
        """Encode terrain type as float."""
        terrain_map = {
            "normal": 0.0,
            "water": 1.0,
            "lava": 2.0,
            "ice": 3.0,
            "mud": 4.0,
        }
        return terrain_map.get(terrain_type.lower(), 0.0)
    
    def _encode_resource_type(self, resource_type: str) -> float:
        """Encode resource type as float."""
        resource_map = {
            "coin": 0.0,
            "gem": 1.0,
            "health": 2.0,
            "energy": 3.0,
            "ammo": 4.0,
        }
        return resource_map.get(resource_type.lower(), 0.0)
    
    def _encode_powerup_type(self, powerup_type: str) -> float:
        """Encode powerup type as float."""
        powerup_map = {
            "speed": 0.0,
            "strength": 1.0,
            "shield": 2.0,
            "health": 3.0,
            "invisibility": 4.0,
        }
        return powerup_map.get(powerup_type.lower(), 0.0)
    
    def _encode_objective_type(self, objective_type: str) -> float:
        """Encode objective type as float."""
        objective_map = {
            "none": 0.0,
            "capture": 1.0,
            "defend": 2.0,
            "collect": 3.0,
            "survive": 4.0,
        }
        return objective_map.get(objective_type.lower(), 0.0)
    
    def parse_batch(self, raw_states: List[Dict[str, Any]]) -> np.ndarray:
        """Parse multiple states into a batch of feature vectors.
        
        Args:
            raw_states: List of raw JSON states
            
        Returns:
            numpy array of shape (batch_size, 128)
        """
        batch_size = len(raw_states)
        batch_features = np.zeros((batch_size, self.feature_dim), dtype=np.float32)
        
        for i, state in enumerate(raw_states):
            batch_features[i] = self.parse(state)
        
        return batch_features
