"""Rule-based agent for Molty Royale with heuristic decision making."""

import math
from typing import Dict, Any, Optional, Tuple
import numpy as np

from src.constants import (
    ActionType,
    MoveDirection,
    AttackType,
    DefenseType,
    CollectType,
    SpecialAction,
    ALL_ACTIONS,
    ACTION_TO_IDX,
    GameConfig,
    Reward,
)


class RuleBasedAgent:
    """Rule-based agent with heuristic decision making and EP management."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Thresholds
        self.health_critical = self.config.get("health_critical", 30.0)
        self.energy_attack_min = self.config.get("energy_attack_min", 20.0)
        self.energy_special_min = self.config.get("energy_special_min", 50.0)
        self.enemy_threat_distance = self.config.get("enemy_threat_distance", 150.0)
        self.safe_zone_margin = self.config.get("safe_zone_margin", 50.0)
        self.resource_pickup_distance = self.config.get("resource_pickup_distance", 30.0)
        
        # Weights for decision scoring
        self.weights = {
            "survival": 10.0,
            "offense": 5.0,
            "collection": 3.0,
            "exploration": 1.0,
        }
        
        self.last_action = None
        self.action_history = []
    
    def choose_action(self, state: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Choose an action based on heuristic rules.
        
        Args:
            state: Current game state (raw JSON or parsed features)
            
        Returns:
            Tuple of (action_type, action_parameters)
        """
        player = state.get("player", {})
        player_stats = player.get("stats", {})
        player_pos = player.get("position", {"x": 0.0, "y": 0.0})
        
        # Extract key information
        health = player_stats.get("health", 0.0)
        energy = player_stats.get("energy", 0.0)
        max_health = player_stats.get("max_health", GameConfig.MAX_HEALTH)
        max_energy = player_stats.get("max_energy", GameConfig.MAX_ENERGY)
        
        # Get environment info
        game_info = state.get("game", {})
        zone = game_info.get("zone", {})
        map_info = game_info.get("map", {})
        
        # Priority 1: Critical health - heal or defend
        if health < self.health_critical:
            action = self._handle_critical_health(state, player, player_pos, zone)
            if action:
                return action
        
        # Priority 2: Pickup & equip best item
        action = self._handle_item_pickup(state, player_pos)
        if action:
            return action
        
        # Priority 3: Attack nearest enemy if EP sufficient
        action = self._handle_enemy_attack(state, player, player_pos, energy)
        if action:
            return action
        
        # Priority 4: Move to safe zone / avoid deathzone
        action = self._handle_safe_zone(state, player_pos, zone, map_info)
        if action:
            return action
        
        # Priority 5: Explore if no threat
        action = self._handle_exploration(state, player_pos, map_info)
        if action:
            return action
        
        # Default: wait
        return (ActionType.WAIT, {})
    
    def _handle_critical_health(
        self,
        state: Dict[str, Any],
        player: Dict[str, Any],
        player_pos: Dict[str, float],
        zone: Dict[str, Any]
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Handle critical health situation."""
        player_stats = player.get("stats", {})
        health = player_stats.get("health", 0.0)
        energy = player_stats.get("energy", 0.0)
        
        # Check for nearby health resources
        resources = state.get("resources", [])
        best_health_resource = None
        best_health_dist = float('inf')
        
        for resource in resources:
            if resource.get("type") == "health" and not resource.get("is_collected", False):
                res_pos = resource.get("position", {"x": 0.0, "y": 0.0})
                dist = self._calculate_distance(player_pos, res_pos)
                if dist < best_health_dist:
                    best_health_dist = dist
                    best_health_resource = resource
        
        # Move to health resource if available
        if best_health_resource and best_health_dist < 200.0:
            direction = self._get_direction_toward(player_pos, best_health_resource["position"])
            return (ActionType.MOVE, {"direction": direction})
        
        # Use special heal if available
        if energy >= self.energy_special_min:
            return (ActionType.SPECIAL, {"type": SpecialAction.HEAL})
        
        # Shield/defend if being attacked
        enemies = state.get("enemies", [])
        nearby_enemy = self._get_nearest_enemy(enemies, player_pos)
        if nearby_enemy and nearby_enemy["distance"] < self.enemy_threat_distance:
            return (ActionType.DEFEND, {"type": DefenseType.SHIELD})
        
        # Move to safe zone
        if zone.get("shrinking", False):
            direction = self._get_direction_toward(player_pos, {"x": zone["x"], "y": zone["y"]})
            return (ActionType.MOVE, {"direction": direction})
        
        return None
    
    def _handle_item_pickup(
        self,
        state: Dict[str, Any],
        player_pos: Dict[str, float]
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Handle item pickup and equipment."""
        resources = state.get("resources", [])
        powerups = state.get("powerups", [])
        
        # Collect all collectible items
        items = []
        
        # Resources
        for resource in resources:
            if not resource.get("is_collected", False):
                res_pos = resource.get("position", {"x": 0.0, "y": 0.0})
                dist = self._calculate_distance(player_pos, res_pos)
                value = resource.get("value", 0.0)
                item_type = resource.get("type", "coin")
                
                # Priority scoring
                priority = value
                if item_type == "health":
                    priority *= 2.0
                elif item_type == "energy":
                    priority *= 1.5
                
                items.append({
                    "type": "resource",
                    "data": resource,
                    "distance": dist,
                    "priority": priority,
                    "position": res_pos,
                })
        
        # Powerups
        for powerup in powerups:
            pu_pos = powerup.get("position", {"x": 0.0, "y": 0.0})
            dist = self._calculate_distance(player_pos, pu_pos)
            pu_type = powerup.get("type", "speed")
            
            # Priority scoring for powerups
            priority_map = {
                "health": 10.0,
                "shield": 8.0,
                "strength": 7.0,
                "speed": 6.0,
                "invisibility": 5.0,
            }
            priority = priority_map.get(pu_type, 5.0)
            
            items.append({
                "type": "powerup",
                "data": powerup,
                "distance": dist,
                "priority": priority,
                "position": pu_pos,
            })
        
        # Sort by priority/distance ratio
        items.sort(key=lambda x: x["priority"] / (x["distance"] + 1.0), reverse=True)
        
        # Pick best item if within range
        if items:
            best_item = items[0]
            if best_item["distance"] <= self.resource_pickup_distance:
                if best_item["type"] == "resource":
                    return (ActionType.COLLECT, {"type": CollectType.RESOURCE})
                else:
                    return (ActionType.COLLECT, {"type": CollectType.POWERUP})
            elif best_item["distance"] < 100.0:
                direction = self._get_direction_toward(player_pos, best_item["position"])
                return (ActionType.MOVE, {"direction": direction})
        
        return None
    
    def _handle_enemy_attack(
        self,
        state: Dict[str, Any],
        player: Dict[str, Any],
        player_pos: Dict[str, float],
        energy: float
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Handle enemy attack decision."""
        enemies = state.get("enemies", [])
        player_stats = player.get("stats", {})
        
        # Get nearest enemy
        nearest_enemy = self._get_nearest_enemy(enemies, player_pos)
        
        if not nearest_enemy:
            return None
        
        enemy_dist = nearest_enemy["distance"]
        enemy_health = nearest_enemy.get("health", 0.0)
        enemy_threat = nearest_enemy.get("threat_level", 0.0)
        
        # Check if enemy is in attack range
        if enemy_dist > self.enemy_threat_distance:
            return None
        
        # EP management: only attack if sufficient energy
        if energy < self.energy_attack_min:
            return None
        
        # Choose attack type based on situation
        if enemy_health < 30.0 and energy >= self.energy_special_min:
            # Finish with special attack
            return (ActionType.ATTACK, {"type": AttackType.SPECIAL})
        elif enemy_threat > 0.7 and energy >= 30.0:
            # High threat - use power attack
            return (ActionType.ATTACK, {"type": AttackType.POWER})
        elif enemy_dist < 50.0:
            # Close range - area attack
            return (ActionType.ATTACK, {"type": AttackType.AREA})
        else:
            # Basic attack
            return (ActionType.ATTACK, {"type": AttackType.BASIC})
    
    def _handle_safe_zone(
        self,
        state: Dict[str, Any],
        player_pos: Dict[str, float],
        zone: Dict[str, Any],
        map_info: Dict[str, Any]
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Handle safe zone movement and deathzone avoidance."""
        zone_x = zone.get("x", 0.0)
        zone_y = zone.get("y", 0.0)
        zone_radius = zone.get("radius", 0.0)
        is_shrinking = zone.get("shrinking", False)
        
        # Calculate distance to zone center
        dist_to_zone = self._calculate_distance(player_pos, {"x": zone_x, "y": zone_y})
        
        # Check if outside safe zone
        if dist_to_zone > zone_radius - self.safe_zone_margin:
            # Move toward safe zone
            direction = self._get_direction_toward(player_pos, {"x": zone_x, "y": zone_y})
            return (ActionType.MOVE, {"direction": direction})
        
        # If zone is shrinking, move closer to center
        if is_shrinking and dist_to_zone > zone_radius * 0.5:
            direction = self._get_direction_toward(player_pos, {"x": zone_x, "y": zone_y})
            return (ActionType.MOVE, {"direction": direction})
        
        # Terrain analysis: avoid hazards
        hazards = state.get("game", {}).get("hazards", [])
        nearest_hazard = self._get_nearest_hazard(hazards, player_pos)
        
        if nearest_hazard and nearest_hazard["distance"] < 50.0:
            # Move away from hazard
            direction = self._get_direction_away(player_pos, nearest_hazard["position"])
            return (ActionType.MOVE, {"direction": direction})
        
        # Terrain analysis: avoid obstacles
        obstacles = state.get("game", {}).get("obstacles", [])
        nearest_obstacle = self._get_nearest_obstacle(obstacles, player_pos)
        
        if nearest_obstacle and nearest_obstacle["distance"] < 30.0:
            # Move away from obstacle
            direction = self._get_direction_away(player_pos, nearest_obstacle["position"])
            return (ActionType.MOVE, {"direction": direction})
        
        return None
    
    def _handle_exploration(
        self,
        state: Dict[str, Any],
        player_pos: Dict[str, float],
        map_info: Dict[str, Any]
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Handle exploration when no immediate threats."""
        player = state.get("player", {})
        player_stats = player.get("stats", {})
        
        # Check if we have enough energy for exploration
        energy = player_stats.get("energy", 0.0)
        if energy < 10.0:
            return (ActionType.WAIT, {})
        
        # Use terrain analysis for exploration
        terrain_type = map_info.get("terrain", "normal")
        
        # If in dangerous terrain, move to safer area
        if terrain_type in ["lava", "water"]:
            # Move toward center of map
            center_x = map_info.get("width", GameConfig.MAP_WIDTH) / 2
            center_y = map_info.get("height", GameConfig.MAP_HEIGHT) / 2
            direction = self._get_direction_toward(player_pos, {"x": center_x, "y": center_y})
            return (ActionType.MOVE, {"direction": direction})
        
        # Explore toward unvisited areas (heuristic: move away from current position)
        # Use action history to avoid backtracking
        if len(self.action_history) > 3:
            last_direction = self.action_history[-1]
            # Try different direction
            directions = [
                MoveDirection.UP,
                MoveDirection.DOWN,
                MoveDirection.LEFT,
                MoveDirection.RIGHT,
                MoveDirection.UP_LEFT,
                MoveDirection.UP_RIGHT,
                MoveDirection.DOWN_LEFT,
                MoveDirection.DOWN_RIGHT,
            ]
            if last_direction in directions:
                directions.remove(last_direction)
            if directions:
                import random
                direction = random.choice(directions)
                return (ActionType.MOVE, {"direction": direction})
        
        # Default: move in a pattern (spiral or random)
        import random
        direction = random.choice([
            MoveDirection.UP,
            MoveDirection.DOWN,
            MoveDirection.LEFT,
            MoveDirection.RIGHT,
        ])
        return (ActionType.MOVE, {"direction": direction})
    
    def _calculate_distance(self, pos1: Dict[str, float], pos2: Dict[str, float]) -> float:
        """Calculate Euclidean distance between two positions."""
        dx = pos2.get("x", 0.0) - pos1.get("x", 0.0)
        dy = pos2.get("y", 0.0) - pos1.get("y", 0.0)
        return math.sqrt(dx * dx + dy * dy)
    
    def _get_direction_toward(self, from_pos: Dict[str, float], to_pos: Dict[str, float]) -> str:
        """Get movement direction toward a target position."""
        dx = to_pos.get("x", 0.0) - from_pos.get("x", 0.0)
        dy = to_pos.get("y", 0.0) - from_pos.get("y", 0.0)
        
        # Normalize and determine direction
        angle = math.atan2(dy, dx)
        degrees = math.degrees(angle)
        
        # Convert angle to direction
        if -22.5 <= degrees < 22.5:
            return MoveDirection.RIGHT
        elif 22.5 <= degrees < 67.5:
            return MoveDirection.DOWN_RIGHT
        elif 67.5 <= degrees < 112.5:
            return MoveDirection.DOWN
        elif 112.5 <= degrees < 157.5:
            return MoveDirection.DOWN_LEFT
        elif 157.5 <= degrees <= 180 or -180 <= degrees < -157.5:
            return MoveDirection.LEFT
        elif -157.5 <= degrees < -112.5:
            return MoveDirection.UP_LEFT
        elif -112.5 <= degrees < -67.5:
            return MoveDirection.UP
        else:  # -67.5 <= degrees < -22.5
            return MoveDirection.UP_RIGHT
    
    def _get_direction_away(self, from_pos: Dict[str, float], away_pos: Dict[str, float]) -> str:
        """Get movement direction away from a position."""
        dx = away_pos.get("x", 0.0) - from_pos.get("x", 0.0)
        dy = away_pos.get("y", 0.0) - from_pos.get("y", 0.0)
        
        # Invert direction
        return self._get_direction_toward(from_pos, {"x": from_pos["x"] - dx, "y": from_pos["y"] - dy})
    
    def _get_nearest_enemy(self, enemies: list, player_pos: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Get the nearest alive enemy."""
        nearest = None
        min_dist = float('inf')
        
        for enemy in enemies:
            if not enemy.get("is_alive", True):
                continue
            
            enemy_pos = enemy.get("position", {"x": 0.0, "y": 0.0})
            dist = self._calculate_distance(player_pos, enemy_pos)
            
            if dist < min_dist:
                min_dist = dist
                nearest = enemy
                nearest["distance"] = dist
        
        return nearest
    
    def _get_nearest_hazard(self, hazards: list, player_pos: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Get the nearest hazard."""
        nearest = None
        min_dist = float('inf')
        
        for hazard in hazards:
            hazard_pos = hazard.get("position", {"x": 0.0, "y": 0.0})
            dist = self._calculate_distance(player_pos, hazard_pos)
            
            if dist < min_dist:
                min_dist = dist
                nearest = hazard
                nearest["distance"] = dist
        
        return nearest
    
    def _get_nearest_obstacle(self, obstacles: list, player_pos: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Get the nearest obstacle."""
        nearest = None
        min_dist = float('inf')
        
        for obstacle in obstacles:
            obs_pos = obstacle.get("position", {"x": 0.0, "y": 0.0})
            dist = self._calculate_distance(player_pos, obs_pos)
            
            if dist < min_dist:
                min_dist = dist
                nearest = obstacle
                nearest["distance"] = dist
        
        return nearest
    
    def reset(self) -> None:
        """Reset agent state."""
        self.last_action = None
        self.action_history = []
    
    def get_action_index(self, action_type: str, action_params: Dict[str, Any]) -> int:
        """Convert action to index for compatibility with RL agent."""
        action = (action_type, action_params.get("direction") or action_params.get("type"))
        return ACTION_TO_IDX.get(action, 0)
