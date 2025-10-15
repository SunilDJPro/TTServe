"""
F5-TTS engine wrapper with model loading and inference
"""

import asyncio
import logging
import io
import tempfile
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
import soundfile as sf
import librosa
from importlib.resources import files
from hydra.utils import get_class
from omegaconf import OmegaConf

from ..config import config
from ..exceptions import ModelNotLoadedError, TTSGenerationError, AudioProcessingError

logger = logging.getLogger(__name__)


class F5TTSEngine:
    """F5-TTS inference engine with singleton pattern"""
    
    def __init__(self):
        self.device = None
        self.model = None
        self.vocoder = None
        self.model_cfg = None
        self.sampling_rate = None
        self.vocoder_name = None
        self._is_loaded = False
        self._load_lock = asyncio.Lock()
        logger.info("F5TTSEngine instance created")
    
    async def load_model(self, device: str = "cuda"):
        """Load F5-TTS model and vocoder"""
        async with self._load_lock:
            if self._is_loaded:
                logger.info("Model already loaded")
                return
            
            self.device = device if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading F5-TTS model on device: {self.device}")
            
            try:
                from f5_tts.infer.utils_infer import load_model, load_vocoder
                
                # Load model config
                model_name = config.MODEL_NAME
                self.model_cfg = OmegaConf.load(
                    str(files("f5_tts").joinpath(f"configs/{model_name}.yaml"))
                )
                
                # Get model details
                model_cls = get_class(f"f5_tts.model.{self.model_cfg.model.backbone}")
                model_arch = self.model_cfg.model.arch
                vocoder_name = self.model_cfg.model.mel_spec.mel_spec_type
                self.vocoder_name = vocoder_name
                self.sampling_rate = self.model_cfg.model.mel_spec.target_sample_rate
                
                # Determine checkpoint path
                ckpt_file = config.CKPT_FILE
                if ckpt_file is None:
                    from huggingface_hub import hf_hub_download
                    logger.info("Downloading model from HuggingFace...")
                    
                    hf_model_map = {
                        "F5TTS_Base": "F5TTS_v1_Base",
                        "F5TTS_Small": "F5TTS_v1_Small",
                        "E2TTS_Base": "E2TTS_Base",
                        "E2TTS_Small": "E2TTS_Small",
                    }
                    
                    hf_model_name = hf_model_map.get(model_name, model_name)
                    ckpt_file = hf_hub_download(
                        repo_id="SWivid/F5-TTS",
                        filename=f"{hf_model_name}/model_1250000.safetensors"
                    )
                
                vocab_file = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))
                
                # Load vocoder
                logger.info("Loading vocoder...")
                self.vocoder = load_vocoder(
                    vocoder_name=vocoder_name,
                    is_local=False,
                    local_path=None,
                    device=self.device
                )
                
                # Load model
                logger.info("Loading F5-TTS model...")
                self.model = load_model(
                    model_cls,
                    model_arch,
                    ckpt_file,
                    mel_spec_type=vocoder_name,
                    vocab_file=vocab_file,
                    device=self.device,
                )
                
                self._is_loaded = True
                logger.info(f"Model loaded successfully: {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to load F5-TTS model: {e}")
                raise ModelNotLoadedError(f"Failed to load model: {e}")
    
    async def unload_model(self):
        """Unload model and free GPU memory"""
        async with self._load_lock:
            if not self._is_loaded:
                return
            
            logger.info("Unloading model...")
            
            self.model = None
            self.vocoder = None
            self.model_cfg = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._is_loaded = False
            logger.info("Model unloaded")
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._is_loaded
    
    async def generate_speech(
        self,
        text: str,
        reference_audio: bytes,
        reference_text: str
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech using F5-TTS
        
        Args:
            text: Text to synthesize
            reference_audio: Reference audio as bytes (WAV format, 24kHz)
            reference_text: Transcript of reference audio
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if not self._is_loaded:
            raise ModelNotLoadedError("Model not loaded. Call load_model() first.")
        
        try:
            logger.info(f"Generating speech for text: {text[:50]}...")
            
            # Save reference audio to temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_ref:
                temp_ref.write(reference_audio)
                ref_audio_path = temp_ref.name
            
            try:
                # Generate using F5-TTS
                from f5_tts.infer.utils_infer import infer_process
                
                wav, sr, spec = infer_process(
                    ref_audio_path,
                    reference_text,
                    text,
                    self.model,
                    self.vocoder,
                    mel_spec_type=self.vocoder_name,
                    show_info=logger.info,
                    progress=None,
                    target_rms=config.TARGET_RMS,
                    cross_fade_duration=config.CROSS_FADE_DURATION,
                    nfe_step=config.NFE_STEP,
                    cfg_strength=config.CFG_STRENGTH,
                    sway_sampling_coef=config.SWAY_SAMPLING_COEF,
                    speed=config.SPEED,
                    device=self.device,
                )
                
                logger.info(f"Generated audio: shape={wav.shape}, sr={sr}")
                return wav, sr
                
            finally:
                # Clean up temp file
                Path(ref_audio_path).unlink(missing_ok=True)
            
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            raise TTSGenerationError(f"Speech generation failed: {e}")
    
    async def validate_and_process_audio(
        self,
        audio_data: bytes,
        min_duration: float = None,
        max_duration: float = None
    ) -> Tuple[bytes, int, float]:
        """
        Validate and process audio data
        
        Args:
            audio_data: Raw audio bytes
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
            
        Returns:
            Tuple of (processed_audio_bytes, sample_rate, duration)
        """
        try:
            # Load audio
            audio_array, sr = sf.read(io.BytesIO(audio_data))
            
            logger.debug(f"Audio loaded: sr={sr}, shape={audio_array.shape}")
            
            # Resample to 24kHz if needed
            if sr != config.SAMPLE_RATE:
                audio_array = librosa.resample(
                    audio_array,
                    orig_sr=sr,
                    target_sr=config.SAMPLE_RATE
                )
                sr = config.SAMPLE_RATE
                logger.debug(f"Resampled to {config.SAMPLE_RATE}Hz")
            
            # Calculate duration
            duration = len(audio_array) / sr
            
            # Validate duration
            if min_duration is not None and duration < min_duration:
                raise AudioProcessingError(
                    f"Audio too short: {duration:.1f}s (min: {min_duration}s)"
                )
            
            if max_duration is not None and duration > max_duration:
                raise AudioProcessingError(
                    f"Audio too long: {duration:.1f}s (max: {max_duration}s)"
                )
            
            # Convert back to bytes
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, audio_array, sr, format='WAV')
            audio_bytes = audio_buffer.getvalue()
            
            logger.debug(f"Audio processed: duration={duration:.2f}s")
            
            return audio_bytes, sr, duration
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            raise AudioProcessingError(f"Audio processing failed: {e}")