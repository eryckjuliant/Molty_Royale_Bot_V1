"""Constants for Molty Royale Self-Learning Bot.

Contains all possible actions, action types, reward definitions,
feature names, and other game constants.
"""

# ============================================================================
# ACTION TYPES
# ============================================================================

class ActionType:
    """Types of actions available in the game."""
    MOVE = "move"
    ATTACK = "attack"
    DEFEND = "defend"
    COLLECT = "collect"
    SPECIAL = "special"
    WAIT = "wait"
    INTERACT = "interact"


# ============================================================================
# MOVE ACTIONS
# ============================================================================

class MoveDirection:
    """Movement directions."""
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    UP_LEFT = "up_left"
    UP_RIGHT = "up_right"
    DOWN_LEFT = "down_left"
    DOWN_RIGHT = "down_right"
    STAY = "stay"


MOVE_ACTIONS = [
    MoveDirection.UP,
    MoveDirection.DOWN,
    MoveDirection.LEFT,
    MoveDirection.RIGHT,
    MoveDirection.UP_LEFT,
    MoveDirection.UP_RIGHT,
    MoveDirection.DOWN_LEFT,
    MoveDirection.DOWN_RIGHT,
    MoveDirection.STAY,
]


# ============================================================================
# ATTACK ACTIONS
# ============================================================================

class AttackType:
    """Types of attacks."""
    BASIC = "basic"
    POWER = "power"
    AREA = "area"
    RANGED = "ranged"
    SPECIAL = "special"


ATTACK_ACTIONS = [
    AttackType.BASIC,
    AttackType.POWER,
    AttackType.AREA,
    AttackType.RANGED,
    AttackType.SPECIAL,
]


# ============================================================================
# DEFEND ACTIONS
# ============================================================================

class DefenseType:
    """Types of defensive actions."""
    BLOCK = "block"
    DODGE = "dodge"
    SHIELD = "shield"
    COUNTER = "counter"


DEFEND_ACTIONS = [
    DefenseType.BLOCK,
    DefenseType.DODGE,
    DefenseType.SHIELD,
    DefenseType.COUNTER,
]


# ============================================================================
# COLLECT ACTIONS
# ============================================================================

class CollectType:
    """Types of collection actions."""
    RESOURCE = "resource"
    POWERUP = "powerup"
    ITEM = "item"
    COIN = "coin"


COLLECT_ACTIONS = [
    CollectType.RESOURCE,
    CollectType.POWERUP,
    CollectType.ITEM,
    CollectType.COIN,
]


# ============================================================================
# SPECIAL ACTIONS
# ============================================================================

class SpecialAction:
    """Special abilities."""
    ULTIMATE = "ultimate"
    HEAL = "heal"
    BOOST = "boost"
    STEALTH = "stealth"
    TELEPORT = "teleport"


SPECIAL_ACTIONS = [
    SpecialAction.ULTIMATE,
    SpecialAction.HEAL,
    SpecialAction.BOOST,
    SpecialAction.STEALTH,
    SpecialAction.TELEPORT,
]


# ============================================================================
# ALL POSSIBLE ACTIONS
# ============================================================================

ALL_ACTIONS = (
    [(ActionType.MOVE, direction) for direction in MOVE_ACTIONS] +
    [(ActionType.ATTACK, attack) for attack in ATTACK_ACTIONS] +
    [(ActionType.DEFEND, defense) for defense in DEFEND_ACTIONS] +
    [(ActionType.COLLECT, collect) for collect in COLLECT_ACTIONS] +
    [(ActionType.SPECIAL, special) for special in SPECIAL_ACTIONS] +
    [(ActionType.WAIT, None)] +
    [(ActionType.INTERACT, None)]
)

NUM_ACTIONS = len(ALL_ACTIONS)


# ============================================================================
# ACTION MAPPING (for discrete action space)
# ============================================================================

ACTION_TO_IDX = {action: idx for idx, action in enumerate(ALL_ACTIONS)}
IDX_TO_ACTION = {idx: action for idx, action in enumerate(ALL_ACTIONS)}


# ============================================================================
# REWARD DEFINITIONS
# ============================================================================

class Reward:
    """Reward values for different events."""
    # Positive rewards
    KILL_ENEMY = 100.0
    COLLECT_RESOURCE = 10.0
    COLLECT_POWERUP = 25.0
    COLLECT_COIN = 5.0
    SURVIVE_ROUND = 20.0
    DEAL_DAMAGE = 1.0
    HEAL = 15.0
    SUCCESSFUL_ATTACK = 5.0
    
    # Negative rewards
    TAKE_DAMAGE = -2.0
    DIE = -50.0
    MISS_ATTACK = -1.0
    WASTE_ACTION = -0.5
    OUT_OF_BOUNDS = -10.0
    COLLISION = -3.0
    
    # Time/effort penalties
    TIME_STEP = -0.01
    INACTION = -0.05
    
    # Win/loss
    WIN_GAME = 500.0
    LOSE_GAME = -200.0
    
    # Objective-based
    CAPTURE_FLAG = 150.0
    DEFEND_OBJECTIVE = 10.0
    COMPLETE_OBJECTIVE = 100.0


# ============================================================================
# FEATURE NAMES (128 features for state representation)
# ============================================================================

# Self features (0-19)
SELF_FEATURES = [
    "self_health",           # 0
    "self_max_health",       # 1
    "self_energy",           # 2
    "self_max_energy",       # 3
    "self_shield",           # 4
    "self_x",                # 5
    "self_y",                # 6
    "self_velocity_x",       # 7
    "self_velocity_y",       # 8
    "self_orientation",      # 9
    "self_level",            # 10
    "self_experience",       # 11
    "self_cooldown_1",       # 12
    "self_cooldown_2",       # 13
    "self_cooldown_3",       # 14
    "self_cooldown_special", # 15
    "self_buff_count",       # 16
    "self_debuff_count",     # 17
    "self_kills",            # 18
    "self_deaths",           # 19
]

# Enemy features (20-49) - up to 3 enemies
ENEMY_FEATURES = [
    f"enemy_{i}_{feat}" for i in range(3) for feat in [
        "health", "max_health", "energy", "x", "y",
        "distance", "angle", "threat_level", "cooldown", "is_alive"
    ]
]

# Teammate features (50-69) - up to 2 teammates
TEAMMATE_FEATURES = [
    f"teammate_{i}_{feat}" for i in range(2) for feat in [
        "health", "max_health", "energy", "x", "y",
        "distance", "angle", "cooldown", "is_alive", "needs_help"
    ]
]

# Environment features (70-81)
ENV_FEATURES = [
    "map_width",              # 70
    "map_height",             # 71
    "time_remaining",         # 72
    "round_number",           # 73
    "zone_x",                 # 74
    "zone_y",                 # 75
    "zone_radius",            # 76
    "zone_shrinking",         # 77
    "weather_condition",      # 78
    "terrain_type",           # 79
    "visibility",             # 80
    "safe_zone_distance",     # 81
]

# Resource features (82-102) - up to 3 resources
RESOURCE_FEATURES = [
    f"resource_{i}_{feat}" for i in range(3) for feat in [
        "type", "x", "y", "distance", "value", "expires_in", "is_collected"
    ]
]

# Powerup features (103-114) - up to 2 powerups
POWERUP_FEATURES = [
    f"powerup_{i}_{feat}" for i in range(2) for feat in [
        "type", "x", "y", "distance", "duration", "expires_in"
    ]
]

# Objective features (115-119)
OBJECTIVE_FEATURES = [
    "objective_type",         # 115
    "objective_x",            # 116
    "objective_y",            # 117
    "objective_progress",     # 118
    "objective_time_left",   # 119
]

# Game state features (120-127)
GAME_STATE_FEATURES = [
    "score",                  # 120
    "enemy_score",            # 121
    "kill_count",             # 122
    "death_count",            # 123
    "assists",                # 124
    "damage_dealt",           # 125
    "damage_taken",           # 126
    "action_count",           # 127
]

# Combined feature list (128 features)
FEATURE_NAMES = (
    SELF_FEATURES +
    ENEMY_FEATURES +
    TEAMMATE_FEATURES +
    ENV_FEATURES +
    RESOURCE_FEATURES +
    POWERUP_FEATURES +
    OBJECTIVE_FEATURES +
    GAME_STATE_FEATURES
)

# Verify we have exactly 128 features
assert len(FEATURE_NAMES) == 128, f"Expected 128 features, got {len(FEATURE_NAMES)}"

# Feature index mapping
FEATURE_TO_IDX = {name: idx for idx, name in enumerate(FEATURE_NAMES)}


# ============================================================================
# GAME CONSTANTS
# ============================================================================

class GameConfig:
    """Game configuration constants."""
    MAX_PLAYERS = 100
    TEAM_SIZE = 4
    MAP_WIDTH = 1000
    MAP_HEIGHT = 1000
    MAX_HEALTH = 100
    MAX_ENERGY = 100
    ROUND_DURATION = 300  # seconds
    MAX_ROUNDS = 10
    RESPAWN_TIME = 10  # seconds
    COOLDOWN_MAX = 30  # seconds


# ============================================================================
# OBSERVATION SPACE
# ============================================================================

OBSERVATION_SPACE = {
    "type": "Box",
    "shape": (128,),
    "low": -float('inf'),
    "high": float('inf'),
}


# ============================================================================
# ACTION SPACE
# ============================================================================

ACTION_SPACE = {
    "type": "Discrete",
    "n": NUM_ACTIONS,
}


# ============================================================================
# TRAINING CONSTANTS
# ============================================================================

class TrainingConfig:
    """Training configuration constants."""
    LEARNING_RATE = 3e-4
    GAMMA = 0.99
    TAU = 0.005
    BUFFER_SIZE = 100000
    BATCH_SIZE = 256
    EPISODE_LENGTH = 1000
    TRAINING_INTERVAL = 20
    SAVE_INTERVAL = 100
    EVAL_INTERVAL = 50
    NUM_EVAL_EPISODES = 10


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class ModelConfig:
    """Neural network model configuration."""
    HIDDEN_LAYERS = [256, 256, 128]
    ACTIVATION = "relu"
    USE_LAYER_NORM = True
    DROPOUT_RATE = 0.1
    INIT_SCALE = 0.01


# ============================================================================
# LOGGING CONSTANTS
# ============================================================================

class LogConfig:
    """Logging configuration."""
    LOG_DIR = "data/logs"
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    TENSORBOARD_DIR = "data/logs/tensorboard"
