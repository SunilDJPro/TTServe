"""
Configuration constants for TTS Library
"""

from pathlib import Path


class Config:
    """Internal configuration constants - DO NOT MODIFY"""
    
    # Audio settings
    SAMPLE_RATE = 24000  # F5-TTS standard
    WEBRTC_SAMPLE_RATE = 48000  # WebRTC standard
    AUDIO_CHANNELS = 1
    
    # F5-TTS model settings
    MODEL_TYPE = "F5-TTS"
    MODEL_NAME = "F5TTS_v1_Base"
    CKPT_FILE = None  # Auto-download from HuggingFace
    
    # Voice registry
    VOICE_REGISTRY_DIR = Path("./voice_registry")
    VOICE_REGISTRY_FILE = "registry.json"
    
    # WebRTC settings
    CODEC = "opus"
    BITRATE = 128000
    
    # TTS generation settings
    TARGET_RMS = 0.1
    CROSS_FADE_DURATION = 0.15
    NFE_STEP = 32
    CFG_STRENGTH = 2.0
    SWAY_SAMPLING_COEF = -1.0
    SPEED = 1.0
    
    # Queue settings
    DEFAULT_MAX_WORKERS = 1
    QUEUE_TIMEOUT = 300  # 5 minutes
    
    # Audio validation
    MIN_REFERENCE_DURATION = 5.0  # seconds
    MAX_REFERENCE_DURATION = 15.0  # seconds


config = Config()