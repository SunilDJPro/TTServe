# F5-TTS WebSocket Server - Configuration
# This config matches the actual F5-TTS repository structure

import os
from pathlib import Path

# =============================================================================
# SERVER CONFIGURATION
# =============================================================================

SERVER_HOST = os.getenv("F5TTS_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("F5TTS_PORT", "8765"))

# Worker Configuration
MAX_CONCURRENT_WORKERS = 1  # Keep at 1 for single GPU (RTX 3080Ti)

# =============================================================================
# F5-TTS MODEL CONFIGURATION
# =============================================================================

# Model Selection
# Available config models: "F5TTS_Base", "F5TTS_Small", "E2TTS_Base", "E2TTS_Small"
# Note: F5TTS_Base config uses F5TTS_v1_Base checkpoint from HuggingFace
MODEL_NAME = "F5TTS_v1_Base" # Config file name in F5-TTS/configs/

# Checkpoint Path
# Option 1: Set to None to auto-download from HuggingFace (recommended)
CKPT_FILE = None

# Option 2: Use local checkpoint (if you have it downloaded)
# CKPT_FILE = "./ckpts/F5TTS_Base/model_1250000.safetensors"

# Option 3: Use cached HuggingFace model directly
# CKPT_FILE = "~/.cache/huggingface/hub/models--SWivid--F5-TTS/snapshots/{hash}/F5TTS_v1_Base/model_1250000.safetensors"
# Or let the server find it automatically in cache (recommended)

# Vocabulary File - Leave empty to use default
VOCAB_FILE = ""  # This is what F5-TTS expects as dataset name

# =============================================================================
# INFERENCE CONFIGURATION
# =============================================================================

ODE_METHOD = "euler"  # euler or midpoint
USE_EMA = True  # Use exponential moving average weights
NFE_STEP = 16  # 16=fast, 32=balanced, 64=quality
CFG_STRENGTH = 2.0  # 1.0-3.0, higher = more style adherence
SWAY_SAMPLING_COEF = -1.0
SPEED = 1.0  # Speech speed multiplier

# =============================================================================
# STORAGE & CACHE
# =============================================================================

VOICE_PROFILES_DIR = Path("./voice_profiles")
LRU_CACHE_SIZE = 4
LRU_EXPIRY_MINUTES = 5

# =============================================================================
# AUDIO & OUTPUT
# =============================================================================

SAMPLE_RATE = 24000
AUDIO_CHANNELS = 1
SAMPLE_WIDTH = 2
SAVE_GENERATIONS = True
GENERATIONS_DIR = Path("./generations")
LOG_LEVEL = "INFO"