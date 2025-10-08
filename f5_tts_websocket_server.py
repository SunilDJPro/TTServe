"""
F5-TTS WebSocket Server
Async WebSocket server for live voice cloning and TTS with persistent voice profiles
"""

import asyncio
import json
import base64
import logging
import wave
import io
import time
import uuid
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from collections import OrderedDict
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn

# F5-TTS imports
# Make sure F5-TTS is installed: pip install -e ./F5-TTS
from importlib.resources import files

from hydra.utils import get_class
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    infer_batch_process,
    chunk_text,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Configuration
class Config:
    # Server settings
    HOST = "0.0.0.0"
    PORT = 8765
    MAX_CONCURRENT_WORKERS = 1  # RTX 3080Ti - best to process one at a time
    
    # Voice profile settings
    VOICE_PROFILES_DIR = Path("./voice_profiles")
    LRU_CACHE_SIZE = 4
    LRU_EXPIRY_MINUTES = 5
    
    # Audio settings
    SAMPLE_RATE = 24000
    AUDIO_CHANNELS = 1
    SAMPLE_WIDTH = 2  # 16-bit
    
    # F5-TTS model settings
    # Model configuration
    MODEL_TYPE = "F5-TTS"  # or "E2-TTS"
    MODEL_NAME = "F5TTS_v1_Base"  
    
    # Checkpoint - either local path or will download from HuggingFace
    CKPT_FILE = None  # Set to None to auto-download, or provide local path
    # Example local: "./F5-TTS/ckpts/F5TTS_Base/model_1200000.safetensors"
    
    # Vocab file (usually auto-loaded from config, set if customized)
    VOCAB_FILE = ""  # Leave empty for default
    
    # Reference audio for testing (optional)
    DEFAULT_REF_AUDIO = None  # Will be provided per voice profile
    DEFAULT_REF_TEXT = ""
    
    # Inference settings
    ODE_METHOD = "euler"  # ODE solver method
    USE_EMA = True  # Use EMA weights
    NFE_STEP = 32  # Number of function evaluations (quality vs speed)
    CFG_STRENGTH = 2.0  # Classifier-free guidance strength
    SWAY_SAMPLING_COEF = -1.0  # Sway sampling coefficient
    SPEED = 1.0  # Speech speed multiplier
    
    # Audio processing
    TARGET_SAMPLE_RATE = 24000  # Will be set from model config
    CHUNK_SIZE = 2048  # Audio chunk size for streaming
    
    # Testing
    SAVE_GENERATIONS = True
    GENERATIONS_DIR = Path("./generations")


config = Config()


@dataclass
class VoiceProfile:
    """Voice profile data structure"""
    voice_id: str
    reference_audio_path: str
    reference_text: str
    created_at: float
    last_accessed: float
    # Cached tensors (not serialized)
    ref_audio_tensor: Optional[torch.Tensor] = None
    ref_audio_duration: Optional[float] = None


class LRUCache:
    """LRU cache with time-based expiration for voice profiles"""
    
    def __init__(self, max_size: int = 4, expiry_minutes: int = 5):
        self.max_size = max_size
        self.expiry_seconds = expiry_minutes * 60
        self.cache: OrderedDict[str, VoiceProfile] = OrderedDict()
        self._lock = asyncio.Lock()
    
    async def get(self, voice_id: str) -> Optional[VoiceProfile]:
        """Get voice profile from cache"""
        async with self._lock:
            if voice_id not in self.cache:
                return None
            
            profile = self.cache[voice_id]
            
            # Check expiry
            if time.time() - profile.last_accessed > self.expiry_seconds:
                logger.info(f"Voice profile {voice_id} expired from cache")
                self.cache.pop(voice_id)
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(voice_id)
            profile.last_accessed = time.time()
            
            return profile
    
    async def put(self, voice_id: str, profile: VoiceProfile):
        """Add or update voice profile in cache"""
        async with self._lock:
            # Update access time
            profile.last_accessed = time.time()
            
            # Remove if already exists (to update position)
            if voice_id in self.cache:
                self.cache.pop(voice_id)
            
            # Add to cache
            self.cache[voice_id] = profile
            self.cache.move_to_end(voice_id)
            
            # Evict oldest if over capacity
            if len(self.cache) > self.max_size:
                oldest_id, oldest_profile = self.cache.popitem(last=False)
                logger.info(f"Evicted voice profile {oldest_id} from cache (LRU)")
    
    async def remove(self, voice_id: str):
        """Remove voice profile from cache"""
        async with self._lock:
            if voice_id in self.cache:
                self.cache.pop(voice_id)
    
    async def clear(self):
        """Clear entire cache"""
        async with self._lock:
            self.cache.clear()


class VoiceProfileManager:
    """Manages voice profiles with disk persistence and LRU caching"""
    
    def __init__(self):
        self.profiles_dir = config.VOICE_PROFILES_DIR
        self.profiles_dir.mkdir(exist_ok=True)
        self.cache = LRUCache(
            max_size=config.LRU_CACHE_SIZE,
            expiry_minutes=config.LRU_EXPIRY_MINUTES
        )
        self._lock = asyncio.Lock()
    
    def _get_profile_dir(self, voice_id: str) -> Path:
        """Get directory path for a voice profile"""
        return self.profiles_dir / voice_id
    
    def _get_metadata_path(self, voice_id: str) -> Path:
        """Get metadata file path for a voice profile"""
        return self._get_profile_dir(voice_id) / "metadata.json"
    
    def _get_audio_path(self, voice_id: str) -> Path:
        """Get audio file path for a voice profile"""
        return self._get_profile_dir(voice_id) / "reference.wav"
    
    async def create_profile(
        self,
        audio_data: bytes,
        reference_text: str
    ) -> str:
        """Create a new voice profile"""
        voice_id = str(uuid.uuid4())
        profile_dir = self._get_profile_dir(voice_id)
        
        async with self._lock:
            # Create profile directory
            profile_dir.mkdir(parents=True, exist_ok=True)
            
            # Load audio and resample to 24000 Hz
            import soundfile as sf
            import io
            audio_array, sr = sf.read(io.BytesIO(audio_data))
            
            logger.info(f"Creating profile: original sr={sr}")
            
            # Resample to 24000 Hz if needed
            if sr != 24000:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=24000)
                sr = 24000
                logger.info(f"Resampled to 24000 Hz")
            
            # Save audio file at 24000 Hz
            audio_path = self._get_audio_path(voice_id)
            sf.write(str(audio_path), audio_array, 24000)
            
            # Create profile
            profile = VoiceProfile(
                voice_id=voice_id,
                reference_audio_path=str(audio_path),
                reference_text=reference_text,
                created_at=datetime.now(),
                last_accessed=datetime.now()
            )
            
            # Save metadata
            metadata = {
                'voice_id': profile.voice_id,
                'reference_audio_path': profile.reference_audio_path,
                'reference_text': profile.reference_text,
                'created_at': profile.created_at.isoformat(),
                'last_accessed': profile.last_accessed.isoformat()
            }
            
            with open(self._get_metadata_path(voice_id), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Created voice profile: {voice_id}")
            return voice_id
    
    async def get_profile(self, voice_id: str) -> Optional[VoiceProfile]:
        """Get voice profile (from cache or disk)"""
        # Try cache first
        profile = await self.cache.get(voice_id)
        if profile is not None:
            logger.debug(f"Voice profile {voice_id} retrieved from cache")
            return profile
        
        # Load from disk
        async with self._lock:
            metadata_path = self._get_metadata_path(voice_id)
            
            if not metadata_path.exists():
                logger.warning(f"Voice profile {voice_id} not found")
                return None
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Create profile object
            profile = VoiceProfile(
                voice_id=metadata['voice_id'],
                reference_audio_path=metadata['reference_audio_path'],
                reference_text=metadata['reference_text'],
                created_at=metadata['created_at'],
                last_accessed=time.time()  # Update access time
            )
            
            # Update last_accessed on disk
            metadata['last_accessed'] = profile.last_accessed
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Add to cache
            await self.cache.put(voice_id, profile)
            
            logger.info(f"Voice profile {voice_id} loaded from disk")
            return profile
    
    async def delete_profile(self, voice_id: str) -> bool:
        """Delete a voice profile"""
        async with self._lock:
            profile_dir = self._get_profile_dir(voice_id)
            
            if not profile_dir.exists():
                logger.warning(f"Voice profile {voice_id} not found for deletion")
                return False
            
            # Remove from cache
            await self.cache.remove(voice_id)
            
            # Delete from disk
            shutil.rmtree(profile_dir)
            
            logger.info(f"Deleted voice profile: {voice_id}")
            return True
    
    async def list_profiles(self) -> list[str]:
        """List all voice profile IDs"""
        async with self._lock:
            profiles = []
            for item in self.profiles_dir.iterdir():
                if item.is_dir() and (item / "metadata.json").exists():
                    profiles.append(item.name)
            return profiles

    async def save_profile(self, profile: VoiceProfile, audio_data: np.ndarray, sample_rate: int):
        """Save voice profile with audio at correct sample rate"""
        async with self._lock:
            profile_dir = self._get_profile_dir(profile.voice_id)
            profile_dir.mkdir(parents=True, exist_ok=True)
            
            # Save audio at 24000 Hz
            audio_path = profile_dir / "reference.wav"
            import soundfile as sf
            sf.write(str(audio_path), audio_data, 24000)  # Force 24000 Hz
            
            # Update profile with actual path
            profile.reference_audio_path = str(audio_path)
            
            # Save metadata
            metadata = {
                'voice_id': profile.voice_id,
                'reference_audio_path': profile.reference_audio_path,
                'reference_text': profile.reference_text,
                'created_at': profile.created_at.isoformat(),
                'last_accessed': profile.last_accessed.isoformat()
            }
            
            with open(self._get_metadata_path(profile.voice_id), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Add to cache
            await self.cache.put(profile.voice_id, profile)
            
            logger.info(f"Saved voice profile: {profile.voice_id}")


class F5TTSEngine:
    """F5-TTS inference engine wrapper using official F5-TTS API"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.vocoder = None
        self.model_cfg = None
        self.sampling_rate = None
        self.mel_spec_type = None
        logger.info(f"F5-TTS Engine initialized on device: {self.device}")
    
    async def load_model(self):
        """Load F5-TTS model and vocoder"""
        logger.info("Loading F5-TTS model...")
        
        try:
            # Load model config
            model_name = config.MODEL_NAME
            self.model_cfg = OmegaConf.load(
                str(files("f5_tts").joinpath(f"configs/{model_name}.yaml"))
            )
            
            

            # Get model class
            model_cls = get_class(f"f5_tts.model.{self.model_cfg.model.backbone}")
            model_arch = self.model_cfg.model.arch
            vocoder_name = self.model_cfg.model.mel_spec.mel_spec_type
            self.mel_spec_type = vocoder_name  # Store as instance variable
            self.vocoder_name = vocoder_name   # ADD THIS LINE - store separately
            self.sampling_rate = self.model_cfg.model.mel_spec.target_sample_rate
            
            
            
            # Determine checkpoint path
            ckpt_file = config.CKPT_FILE
            if ckpt_file is None:
                # Auto-download from HuggingFace
                from huggingface_hub import hf_hub_download
                logger.info("Checking for model checkpoint...")
                
                # Map config model names to HuggingFace paths
                # F5TTS_Base config uses F5TTS_v1_Base checkpoint
                hf_model_map = {
                    "F5TTS_Base": "F5TTS_v1_Base",
                    "F5TTS_Small": "F5TTS_v1_Small",
                    "E2TTS_Base": "E2TTS_Base",
                    "E2TTS_Small": "E2TTS_Small",
                }
                
                hf_model_name = hf_model_map.get(model_name, model_name)
                
                try:
                    logger.info(f"Downloading {hf_model_name}/model_1250000.safetensors from HuggingFace...")
                    ckpt_file = hf_hub_download(
                        repo_id="SWivid/F5-TTS",
                        filename=f"{hf_model_name}/model_1250000.safetensors"
                    )
                except Exception as e:
                    logger.error(f"Failed to download from HuggingFace: {e}")
                    logger.info("Checking local cache...")
                    # Try to find in local cache
                    import os
                    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--SWivid--F5-TTS/snapshots")
                    if os.path.exists(cache_dir):
                        for snapshot in os.listdir(cache_dir):
                            snapshot_path = os.path.join(cache_dir, snapshot, hf_model_name, "model_1250000.safetensors")
                            if os.path.exists(snapshot_path):
                                logger.info(f"Found model in local cache: {snapshot_path}")
                                ckpt_file = snapshot_path
                                break
                    
                    if ckpt_file is None:
                        raise Exception("Model checkpoint not found. Please set CKPT_FILE in config.py")
            
            vocab_file = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))
            logger.info(f"Using vocab: {vocab_file}")
            # Load model
            # 

            # Load vocoder FIRST
            logger.info("Loading vocoder...")
            self.vocoder = load_vocoder(
                vocoder_name=vocoder_name,  # Use vocoder_name
                is_local=False,
                local_path=None,
                device=self.device
            )
            
            self.model = load_model(
                model_cls,
                model_arch,  # This is model_cfg parameter
                ckpt_file,   # This is ckpt_path parameter
                mel_spec_type=vocoder_name,  # Must match vocoder
                vocab_file=vocab_file,
                device=self.device,
                # Don't pass ode_method or use_ema - use defaults
            )
            
            logger.info(f"Model loaded: {model_name}")
            
            # # Load vocoder
            # logger.info("Loading vocoder...")
            # self.vocoder = load_vocoder(
            #     vocoder_name=self.mel_spec_type,
            #     is_local=False,
            #     local_path=None,
            #     device=self.device
            # )
            
            logger.info("F5-TTS model and vocoder loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load F5-TTS model: {e}")
            raise
    
    def load_audio(self, audio_path: str) -> tuple[torch.Tensor, int]:
        """Load and preprocess audio file"""
        audio, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
            audio = resampler(audio)
            sr = self.sampling_rate
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        return audio, sr
    
    async def generate_speech(
        self,
        text: str,
        voice_profile: VoiceProfile
    ) -> bytes:
        """Generate speech using F5-TTS"""
        
        try:
            # Just use the saved reference audio path directly
            ref_audio_path = voice_profile.reference_audio_path
            ref_text = voice_profile.reference_text
            
            logger.info(f"Using ref_audio: {ref_audio_path}")
            logger.info(f"Using ref_text: {ref_text}")
            
            # Don't preprocess - use the stored file directly
            # infer_process will handle loading and preprocessing
            
            from f5_tts.infer.utils_infer import infer_process
            
            logger.info(f"=== TTS Parameters ===")
            logger.info(f"ref_audio_path: {ref_audio_path}")
            logger.info(f"ref_text: {ref_text}")
            logger.info(f"gen_text: {text}")
            logger.info(f"model: {type(self.model)}")
            logger.info(f"vocoder: {type(self.vocoder)}")
            logger.info(f"mel_spec_type: {self.vocoder_name}")
            logger.info(f"device: {self.device}")
            logger.info(f"=====================")

            wav, sr, spec = infer_process(
                ref_audio_path,  # Use stored path directly
                ref_text,
                text,
                self.model,
                self.vocoder,
                mel_spec_type=self.vocoder_name,
                show_info=logger.info,
                progress=None,
                target_rms=0.1,
                cross_fade_duration=0.15,
                nfe_step=32,
                cfg_strength=2.0,
                sway_sampling_coef=-1.0,
                speed=1.0,
                device=self.device,
            )
            
            logger.info(f"Generated audio: shape={wav.shape}, sr={sr}")
            
            # Save with soundfile
            import soundfile as sf
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()
            
            return wav_bytes
            
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            raise

    # async def generate_speech(
    #     self,
    #     text: str,
    #     voice_profile: VoiceProfile
    # ) -> bytes:
    #     """Generate speech using F5-TTS"""
        
    #     try:
    #         # Load reference audio if not cached
    #         if voice_profile.ref_audio_tensor is None:
    #             logger.info(f"Loading reference audio for voice {voice_profile.voice_id}")
                
    #             # Preprocess reference audio
    #             ref_audio_path, ref_text = preprocess_ref_audio_text(
    #                 voice_profile.reference_audio_path,
    #                 voice_profile.reference_text
    #             )
                
    #             # Load audio
    #             audio, sr = self.load_audio(ref_audio_path)
                
    #             # Cache in profile
    #             voice_profile.ref_audio_tensor = audio
    #             voice_profile.ref_audio_duration = audio.shape[-1] / sr
    #             voice_profile.reference_text = ref_text  # Use preprocessed text
            
    #         ref_audio = voice_profile.ref_audio_tensor
    #         ref_text = voice_profile.reference_text
    #         sr = self.sampling_rate
            
    #         # Prepare text for generation
    #         gen_text = text
            
    #         # Chunk text for better processing
    #         text_batches = chunk_text(
    #             gen_text,
    #             max_chars=int(len(ref_text.encode("utf-8")) / voice_profile.ref_audio_duration * 25)
    #         )
            
    #         logger.info(f"Generating speech for {len(text_batches)} text batch(es)")

    #         # Generate audio using F5-TTS inference - yields (audio, sr, spectrogram)
    #         audio_chunks = []
    #         for result in infer_batch_process(
    #             (ref_audio, sr),
    #             ref_text,
    #             text_batches,
    #             self.model,
    #             self.vocoder,
    #             device=self.device,
    #         ):
    #             # Extract just the audio from the result tuple
    #             if isinstance(result, tuple):
    #                 audio_chunk = result[0]  # First element is audio
    #             else:
    #                 audio_chunk = result
    #             audio_chunks.append(audio_chunk)

    #         # Concatenate all chunks
    #         if len(audio_chunks) > 0:
    #             generated_audio = np.concatenate(audio_chunks, axis=0)

    #             #DEBUG
    #             import scipy.io.wavfile as wavfile
    #             wavfile.write('./debug_raw.wav', self.sampling_rate, generated_audio)
    #             logger.info(f"Saved raw audio to ./debug_raw.wav")
    #         else:
    #             raise Exception("No audio generated")

    #         # Ensure it's 1D array
    #         if generated_audio.ndim > 1:
    #             generated_audio = generated_audio.flatten()

    #         # Normalize audio properly
    #         max_val = np.abs(generated_audio).max()
    #         if max_val > 0:
    #             generated_audio = generated_audio / max_val

    #         # Normalize audio
    #         generated_audio = np.clip(generated_audio, -1.0, 1.0) #generated_audio / np.max(np.abs(generated_audio))
    #         audio_int16 = (generated_audio * 32767).astype(np.int16)
            
    #         logger.info(f"Audio shape: {generated_audio.shape}, dtype: {generated_audio.dtype}, sample_rate: {self.sampling_rate}")

    #         # Convert to WAV bytes
    #         wav_buffer = io.BytesIO()
    #         with wave.open(wav_buffer, 'wb') as wav_file:
    #             wav_file.setnchannels(config.AUDIO_CHANNELS)
    #             wav_file.setsampwidth(config.SAMPLE_WIDTH)
    #             wav_file.setframerate(self.sampling_rate)
    #             wav_file.writeframes(audio_int16.tobytes())
            
    #         wav_bytes = wav_buffer.getvalue()
    #         logger.info(f"Generated {len(wav_bytes)} bytes of audio")
            
    #         return wav_bytes
            
    #     except Exception as e:
    #         logger.error(f"Error generating speech: {e}")
    #         raise


class RequestQueue:
    """Async request queue with worker pool"""
    
    def __init__(self, max_workers: int = 1):
        self.max_workers = max_workers
        self.queue: asyncio.Queue = asyncio.Queue()
        self.workers: list[asyncio.Task] = []
        self.is_running = False
    
    async def start(self):
        """Start worker tasks"""
        self.is_running = True
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(i))
            self.workers.append(worker)
        logger.info(f"Started {self.max_workers} worker(s)")
    
    async def stop(self):
        """Stop all workers"""
        self.is_running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        logger.info("All workers stopped")
    
    async def _worker(self, worker_id: int):
        """Worker task that processes queue items"""
        logger.info(f"Worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get task from queue
                task_func, future = await self.queue.get()
                
                logger.info(f"Worker {worker_id} processing task")
                
                try:
                    # Execute task
                    result = await task_func()
                    future.set_result(result)
                except Exception as e:
                    logger.error(f"Worker {worker_id} task failed: {e}")
                    future.set_exception(e)
                finally:
                    self.queue.task_done()
                    
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
    
    async def submit(self, task_func):
        """Submit a task to the queue"""
        future = asyncio.Future()
        await self.queue.put((task_func, future))
        logger.info(f"Task submitted to queue (queue size: {self.queue.qsize()})")
        return await future


# Global instances
app = FastAPI(title="F5-TTS WebSocket Server")
voice_manager = VoiceProfileManager()
tts_engine = F5TTSEngine()
request_queue = RequestQueue(max_workers=config.MAX_CONCURRENT_WORKERS)


@app.on_event("startup")
async def startup_event():
    """Initialize server components"""
    logger.info("Starting F5-TTS WebSocket Server...")
    
    # Create directories
    config.VOICE_PROFILES_DIR.mkdir(exist_ok=True)
    if config.SAVE_GENERATIONS:
        config.GENERATIONS_DIR.mkdir(exist_ok=True)
    
    # Load TTS model
    await tts_engine.load_model()
    
    # Start request queue
    await request_queue.start()
    
    logger.info("Server startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown"""
    logger.info("Shutting down server...")
    await request_queue.stop()
    logger.info("Server shutdown complete")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "F5-TTS WebSocket Server",
        "endpoints": {
            "websocket": "/ws",
            "voice_profiles": "/voices"
        }
    }


@app.get("/voices")
async def list_voices():
    """List all voice profiles"""
    voices = await voice_manager.list_profiles()
    return {"voices": voices, "count": len(voices)}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint"""
    await websocket.accept()
    websocket.client_state.ping_timeout = 300
    logger.info(f"WebSocket connection established: {websocket.client}")
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            logger.info(f"Received message: action={message.get('action')}")
            action = message.get("action")
            
            if action == "register_voice":
                await handle_register_voice(websocket, message)
            
            elif action == "tts":
                await handle_tts(websocket, message)
            
            elif action == "delete_voice":
                await handle_delete_voice(websocket, message)
            
            elif action == "list_voices":
                await handle_list_voices(websocket, message)
            
            else:
                await websocket.send_json({
                    "status": "error",
                    "message": f"Unknown action: {action}"
                })
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {websocket.client}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "status": "error",
                "message": str(e)
            })
        except:
            pass


async def handle_register_voice(websocket: WebSocket, message: dict):
    """Handle voice registration"""
    try:
        audio_base64 = message.get("audio")
        reference_text = message.get("reference_text", "")
        #profile_manager = VoiceProfileManager()
        # Decode audio
        audio_data = base64.b64decode(audio_base64)
        
        # Load and resample audio
        import soundfile as sf
        import io
        audio_array, sr = sf.read(io.BytesIO(audio_data))
        
        logger.info(f"Received audio: sr={sr}, shape={audio_array.shape}")
        
        # Resample to 24000 Hz if needed
        if sr != 24000:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=24000)
            sr = 24000
            logger.info(f"Resampled to 24000 Hz")
        
        # Convert back to bytes at 24000 Hz
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio_array, 24000, format='WAV')
        audio_bytes = audio_buffer.getvalue()
        
        # Use create_profile which handles everything
        voice_id = await voice_manager.create_profile(
            audio_data=audio_bytes,
            reference_text=reference_text
        )
        
        await websocket.send_json({
            "action": "register_voice",
            "status": "success",
            "voice_id": voice_id
        })
        
        logger.info(f"Voice registered: {voice_id}")
        
    except Exception as e:
        logger.error(f"Error registering voice: {e}")
        await websocket.send_json({
            "action": "register_voice",
            "status": "error",
            "message": str(e)
        })

async def handle_tts(websocket: WebSocket, message: dict):
    """Handle TTS request with chunking for large audio"""
    logger.info(f"TTS request received: {message}")
    try:
        voice_id = message.get("voice_id")
        text = message.get("text")
        save_file = message.get("save_file", False)
        
        if not voice_id or not text:
            await websocket.send_json({
                "status": "error",
                "message": "Missing voice_id or text"
            })
            return
        
        # Get voice profile
        voice_profile = await voice_manager.get_profile(voice_id)
        
        if voice_profile is None:
            await websocket.send_json({
                "status": "error",
                "message": f"Voice profile not found: {voice_id}"
            })
            return
        
        # Submit TTS task to queue
        async def tts_task():
            return await tts_engine.generate_speech(text, voice_profile)
        
        # Wait for generation
        audio_bytes = await request_queue.submit(tts_task)
        
        # Optionally save to disk
        saved_path = None
        if save_file or config.SAVE_GENERATIONS:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{voice_id}_{timestamp}.wav"
            saved_path = config.GENERATIONS_DIR / filename
            with open(saved_path, 'wb') as f:
                f.write(audio_bytes)
            logger.info(f"Saved generation to: {saved_path}")
        
        MAX_AUDIO_CHUNK_SIZE = 700000  # Leaves room for base64 encoding and JSON overhead

        if len(audio_bytes) > MAX_AUDIO_CHUNK_SIZE:
            # Send metadata first
            await websocket.send_json({
                "status": "streaming",
                "action": "tts",
                "voice_id": voice_id,
                "total_size": len(audio_bytes),
                "total_chunks": (len(audio_bytes) + MAX_AUDIO_CHUNK_SIZE - 1) // MAX_AUDIO_CHUNK_SIZE,
                "format": "wav",
                "sample_rate": config.SAMPLE_RATE,
                "channels": config.AUDIO_CHANNELS
            })
            
            # Send audio in chunks
            num_chunks = 0
            for i in range(0, len(audio_bytes), MAX_AUDIO_CHUNK_SIZE):
                chunk = audio_bytes[i:i+MAX_AUDIO_CHUNK_SIZE]
                chunk_base64 = base64.b64encode(chunk).decode('utf-8')
                
                await websocket.send_json({
                    "status": "chunk",
                    "action": "tts_chunk",
                    "chunk_index": num_chunks,
                    "chunk": chunk_base64
                })
                num_chunks += 1
            
            # Send completion
            response = {
                "status": "complete",
                "action": "tts",
                "voice_id": voice_id,
                "total_chunks": num_chunks
            }
            
            if saved_path:
                response["saved_path"] = str(saved_path)
            
            await websocket.send_json(response)
            
            logger.info(f"TTS completed (chunked) for voice: {voice_id}, chunks: {num_chunks}, text length: {len(text)}")
        
        else:
            # Send as single message for small audio
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            response = {
                "status": "success",
                "action": "tts",
                "voice_id": voice_id,
                "audio": audio_base64,
                "format": "wav",
                "sample_rate": config.SAMPLE_RATE,
                "channels": config.AUDIO_CHANNELS
            }
            
            if saved_path:
                response["saved_path"] = str(saved_path)
            
            await websocket.send_json(response)
            
            logger.info(f"TTS completed for voice: {voice_id}, text length: {len(text)}")
    
    except Exception as e:
        logger.error(f"Error in TTS: {e}")
        await websocket.send_json({
            "status": "error",
            "action": "tts",
            "message": str(e)
        })


async def handle_delete_voice(websocket: WebSocket, message: dict):
    """Handle voice deletion"""
    try:
        voice_id = message.get("voice_id")
        
        if not voice_id:
            await websocket.send_json({
                "status": "error",
                "message": "Missing voice_id"
            })
            return
        
        # Delete voice profile
        success = await voice_manager.delete_profile(voice_id)
        
        if success:
            await websocket.send_json({
                "status": "success",
                "action": "delete_voice",
                "voice_id": voice_id,
                "message": "Voice profile deleted successfully"
            })
        else:
            await websocket.send_json({
                "status": "error",
                "action": "delete_voice",
                "message": f"Voice profile not found: {voice_id}"
            })
    
    except Exception as e:
        logger.error(f"Error deleting voice: {e}")
        await websocket.send_json({
            "status": "error",
            "action": "delete_voice",
            "message": str(e)
        })


async def handle_list_voices(websocket: WebSocket, message: dict):
    """Handle list voices request"""
    try:
        voices = await voice_manager.list_profiles()
        
        await websocket.send_json({
            "status": "success",
            "action": "list_voices",
            "voices": voices,
            "count": len(voices)
        })
    
    except Exception as e:
        logger.error(f"Error listing voices: {e}")
        await websocket.send_json({
            "status": "error",
            "action": "list_voices",
            "message": str(e)
        })


def main():
    """Run the server"""
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level="info"
    )


if __name__ == "__main__":
    main()