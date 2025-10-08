"""
F5-TTS WebRTC Server
WebRTC-based streaming server for live voice cloning and TTS
Uses aiortc for WebRTC and HTTP endpoints for signaling
"""

import asyncio
import json
import base64
import logging
import io
import time
import uuid
import os
import shutil
import psutil
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import numpy as np

import torch
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# WebRTC imports
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
from av import AudioFrame
import soundfile as sf
import librosa

# F5-TTS imports
from importlib.resources import files
from hydra.utils import get_class
from omegaconf import OmegaConf
from f5_tts.infer.utils_infer import infer_process


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Resource monitoring utility
class ResourceMonitor:
    """Monitor GPU, CPU, and RAM usage during TTS generation"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.gpu_available = torch.cuda.is_available()
        
    def get_gpu_memory_mb(self):
        """Get current GPU memory usage in MB (allocated + cached)"""
        if self.gpu_available:
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024
            return allocated, reserved
        return 0, 0
    
    def get_cpu_percent(self):
        """Get current CPU usage percentage with proper interval"""
        # Use longer interval for accurate measurement
        return self.process.cpu_percent(interval=1.0)
    
    def get_ram_mb(self):
        """Get current RAM usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_system_cpu_percent(self):
        """Get system-wide CPU usage"""
        return psutil.cpu_percent(interval=1.0)
    
    def start_monitoring(self):
        """Start monitoring - capture initial state"""
        start_gpu_allocated, start_gpu_reserved = self.get_gpu_memory_mb()
        
        return {
            'start_time': time.time(),
            'start_gpu_allocated_mb': start_gpu_allocated,
            'start_gpu_reserved_mb': start_gpu_reserved,
            'start_ram_mb': self.get_ram_mb(),
            'cpu_samples': []  # Collect CPU samples during generation
        }
    
    def sample_cpu(self, start_metrics):
        """Sample CPU usage during generation"""
        cpu_percent = psutil.cpu_percent(interval=None, percpu=False)
        start_metrics['cpu_samples'].append(cpu_percent)
    
    def end_monitoring(self, start_metrics):
        """End monitoring and calculate resource usage"""
        end_time = time.time()
        duration = end_time - start_metrics['start_time']
        
        end_gpu_allocated, end_gpu_reserved = self.get_gpu_memory_mb()
        end_ram_mb = self.get_ram_mb()
        
        # Calculate GPU memory delta
        gpu_allocated_delta = end_gpu_allocated - start_metrics['start_gpu_allocated_mb']
        gpu_reserved_delta = end_gpu_reserved - start_metrics['start_gpu_reserved_mb']
        
        # RAM delta
        ram_delta = end_ram_mb - start_metrics['start_ram_mb']
        
        # CPU average from samples (if we collected any)
        if start_metrics['cpu_samples']:
            avg_cpu = sum(start_metrics['cpu_samples']) / len(start_metrics['cpu_samples'])
        else:
            avg_cpu = psutil.cpu_percent(interval=0.5)
        
        # GPU hours (convert seconds to hours)
        gpu_hours = duration / 3600 if self.gpu_available else 0
        
        return {
            'duration_seconds': round(duration, 2),
            'gpu_hours': round(gpu_hours, 6),
            'gpu_allocated_mb': round(end_gpu_allocated, 2),
            'gpu_reserved_mb': round(end_gpu_reserved, 2),
            'gpu_allocated_delta_mb': round(gpu_allocated_delta, 2),
            'gpu_reserved_delta_mb': round(gpu_reserved_delta, 2),
            'cpu_percent_avg': round(avg_cpu, 2),
            'cpu_samples_count': len(start_metrics['cpu_samples']),
            'ram_used_mb': round(ram_delta, 2),
            'ram_peak_mb': round(end_ram_mb, 2),
            'gpu_available': self.gpu_available
        }
    
    def log_resources(self, metrics, voice_id, text_length):
        """Log resource usage in a formatted way"""
        logger.info("=" * 80)
        logger.info("RESOURCE USAGE REPORT")
        logger.info("=" * 80)
        logger.info(f"Voice ID: {voice_id}")
        logger.info(f"Text Length: {text_length} characters")
        logger.info("-" * 80)
        logger.info(f"Processing Time: {metrics['duration_seconds']} seconds")
        logger.info(f"GPU Hours Consumed: {metrics['gpu_hours']} hours ({metrics['duration_seconds']}s)")
        logger.info(f"GPU Memory Allocated: {metrics['gpu_allocated_mb']} MB (Δ {metrics['gpu_allocated_delta_mb']} MB)")
        logger.info(f"GPU Memory Reserved: {metrics['gpu_reserved_mb']} MB (Δ {metrics['gpu_reserved_delta_mb']} MB)")
        logger.info(f"CPU Usage (Avg): {metrics['cpu_percent_avg']}% ({metrics['cpu_samples_count']} samples)")
        logger.info(f"RAM Used: {metrics['ram_used_mb']} MB (Peak: {metrics['ram_peak_mb']} MB)")
        logger.info(f"GPU Available: {metrics['gpu_available']}")
        logger.info("=" * 80)


# Global resource monitor
resource_monitor = ResourceMonitor()


# Configuration
class Config:
    # Server settings
    HOST = "0.0.0.0"
    PORT = 8766  # Different port from WebSocket server
    MAX_CONCURRENT_WORKERS = 1
    
    # Voice profile settings
    VOICE_PROFILES_DIR = Path("./voice_profiles")
    
    # Audio settings
    SAMPLE_RATE = 24000  # F5-TTS output
    WEBRTC_SAMPLE_RATE = 48000  # WebRTC standard
    AUDIO_CHANNELS = 1
    
    # F5-TTS model settings
    MODEL_TYPE = "F5-TTS"
    MODEL_NAME = "F5TTS_v1_Base"
    CKPT_FILE = None  # Auto-download from HuggingFace
    
    # WebRTC settings
    CODEC = "opus"  # Opus codec for WebRTC
    BITRATE = 128000  # 128 kbps
    
    # Output settings
    SAVE_GENERATIONS = True
    GENERATIONS_DIR = Path("./generations")


config = Config()


# Pydantic models for API
class VoiceRegisterRequest(BaseModel):
    audio: str  # base64 encoded
    reference_text: str


class TTSRequest(BaseModel):
    voice_id: str
    text: str


class OfferRequest(BaseModel):
    sdp: str
    type: str
    voice_id: str
    text: str


class VoiceProfile:
    """Voice profile data structure"""
    def __init__(self, voice_id: str, reference_audio_path: str, reference_text: str):
        self.voice_id = voice_id
        self.reference_audio_path = reference_audio_path
        self.reference_text = reference_text
        self.created_at = time.time()


class AudioStreamTrack(MediaStreamTrack):
    """Custom audio track that streams generated audio via WebRTC"""
    
    kind = "audio"
    
    def __init__(self, audio_data: np.ndarray, sample_rate: int = 48000):
        super().__init__()
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.timestamp = 0
        self.samples_per_frame = int(sample_rate * 0.020)  # 20ms frames
        self.current_position = 0
        
        logger.info(f"AudioStreamTrack initialized: {len(audio_data)} samples at {sample_rate}Hz")
    
    async def recv(self):
        """Generate audio frames for WebRTC"""
        # Check if we've sent all audio
        if self.current_position >= len(self.audio_data):
            logger.info(f"Audio streaming complete - sent {self.current_position} samples total")
            # Don't call stop() here, just let it end naturally
            # This prevents the AttributeError
            await asyncio.sleep(0.1)  # Small delay before ending
            raise StopAsyncIteration  # Proper way to end async iteration
        
        # Get next chunk of audio
        end_position = min(
            self.current_position + self.samples_per_frame,
            len(self.audio_data)
        )
        
        audio_chunk = self.audio_data[self.current_position:end_position]
        
        # Pad if necessary (last frame might be shorter)
        if len(audio_chunk) < self.samples_per_frame:
            audio_chunk = np.pad(
                audio_chunk,
                (0, self.samples_per_frame - len(audio_chunk)),
                mode='constant'
            )
        
        # Convert to int16 and create STEREO by duplicating mono to both channels
        audio_int16 = (audio_chunk * 32767).astype(np.int16)
        
        # Interleave L and R channels: [L0, R0, L1, R1, L2, R2, ...]
        audio_stereo = np.empty(len(audio_int16) * 2, dtype=np.int16)
        audio_stereo[0::2] = audio_int16  # Left channel
        audio_stereo[1::2] = audio_int16  # Right channel (duplicate)
        
        # Reshape to (1, samples*2) for packed stereo format
        audio_stereo = audio_stereo.reshape(1, -1)
        
        # Create stereo AudioFrame
        frame = AudioFrame.from_ndarray(audio_stereo, format='s16', layout='stereo')
        frame.sample_rate = self.sample_rate
        frame.pts = self.timestamp
        from fractions import Fraction
        frame.time_base = Fraction(1, self.sample_rate)
        
        # Update position and timestamp
        self.current_position = end_position
        self.timestamp += self.samples_per_frame
        
        # Small delay to simulate real-time streaming
        await asyncio.sleep(0.020)  # 20ms
        
        return frame


class VoiceProfileManager:
    """Manages voice profiles with disk persistence"""
    
    def __init__(self):
        self.profiles_dir = config.VOICE_PROFILES_DIR
        self.profiles_dir.mkdir(exist_ok=True)
    
    def _get_profile_dir(self, voice_id: str) -> Path:
        return self.profiles_dir / voice_id
    
    def _get_metadata_path(self, voice_id: str) -> Path:
        return self._get_profile_dir(voice_id) / "metadata.json"
    
    def _get_audio_path(self, voice_id: str) -> Path:
        return self._get_profile_dir(voice_id) / "reference.wav"
    
    async def create_profile(self, audio_data: bytes, reference_text: str) -> str:
        """Create a new voice profile"""
        voice_id = str(uuid.uuid4())
        profile_dir = self._get_profile_dir(voice_id)
        
        # Create profile directory
        profile_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and resample audio to 24000 Hz
        audio_array, sr = sf.read(io.BytesIO(audio_data))
        
        logger.info(f"Creating profile: original sr={sr}, shape={audio_array.shape}")
        
        # Resample to 24000 Hz if needed
        if sr != 24000:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=24000)
            sr = 24000
            logger.info(f"Resampled to 24000 Hz")
        
        # Validate duration
        duration = len(audio_array) / sr
        if duration < 5 or duration > 15:
            raise ValueError(f"Audio duration ({duration:.1f}s) must be between 5-15 seconds")
        
        # Save audio file at 24000 Hz
        audio_path = self._get_audio_path(voice_id)
        sf.write(str(audio_path), audio_array, 24000)
        
        # Save metadata
        metadata = {
            'voice_id': voice_id,
            'reference_audio_path': str(audio_path),
            'reference_text': reference_text,
            'created_at': time.time()
        }
        
        with open(self._get_metadata_path(voice_id), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created voice profile: {voice_id}")
        return voice_id
    
    async def get_profile(self, voice_id: str) -> Optional[VoiceProfile]:
        """Get voice profile from disk"""
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
            reference_text=metadata['reference_text']
        )
        
        return profile
    
    async def delete_profile(self, voice_id: str) -> bool:
        """Delete a voice profile"""
        profile_dir = self._get_profile_dir(voice_id)
        
        if not profile_dir.exists():
            return False
        
        shutil.rmtree(profile_dir)
        logger.info(f"Deleted voice profile: {voice_id}")
        return True
    
    async def list_profiles(self) -> list[str]:
        """List all voice profile IDs"""
        profiles = []
        for item in self.profiles_dir.iterdir():
            if item.is_dir() and (item / "metadata.json").exists():
                profiles.append(item.name)
        return profiles


class F5TTSEngine:
    """F5-TTS inference engine wrapper"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.vocoder = None
        self.model_cfg = None
        self.sampling_rate = None
        self.vocoder_name = None
        logger.info(f"F5-TTS Engine initialized on device: {self.device}")
    
    async def load_model(self):
        """Load F5-TTS model and vocoder"""
        logger.info("Loading F5-TTS model...")
        
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
            
            # Load vocoder first
            logger.info("Loading vocoder...")
            self.vocoder = load_vocoder(
                vocoder_name=vocoder_name,
                is_local=False,
                local_path=None,
                device=self.device
            )
            
            # Load model
            self.model = load_model(
                model_cls,
                model_arch,
                ckpt_file,
                mel_spec_type=vocoder_name,
                vocab_file=vocab_file,
                device=self.device,
            )
            
            logger.info(f"Model loaded successfully: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load F5-TTS model: {e}")
            raise
    
    async def generate_speech(
        self,
        text: str,
        voice_profile: VoiceProfile
    ) -> np.ndarray:
        """Generate speech using F5-TTS and return numpy array"""
        
        try:
            ref_audio_path = voice_profile.reference_audio_path
            ref_text = voice_profile.reference_text
            
            logger.info(f"Generating speech for text: {text[:50]}...")
            
            # Generate using F5-TTS
            wav, sr, spec = infer_process(
                ref_audio_path,
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
            
            return wav, sr
            
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            raise


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
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        logger.info("All workers stopped")
    
    async def _worker(self, worker_id: int):
        """Worker task that processes queue items"""
        logger.info(f"Worker {worker_id} started")
        
        while self.is_running:
            try:
                task_func, future = await self.queue.get()
                logger.info(f"Worker {worker_id} processing task")
                
                try:
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
app = FastAPI(title="F5-TTS WebRTC Server")
voice_manager = VoiceProfileManager()
tts_engine = F5TTSEngine()
request_queue = RequestQueue(max_workers=config.MAX_CONCURRENT_WORKERS)

# Store active peer connections
peer_connections: Dict[str, RTCPeerConnection] = {}

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize server components"""
    logger.info("Starting F5-TTS WebRTC Server...")
    
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
    
    # Close all peer connections
    for pc in peer_connections.values():
        await pc.close()
    peer_connections.clear()
    
    await request_queue.stop()
    logger.info("Server shutdown complete")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "F5-TTS WebRTC Server",
        "endpoints": {
            "register_voice": "/api/register_voice",
            "list_voices": "/api/voices",
            "webrtc_offer": "/api/offer",
        }
    }


@app.post("/api/register_voice")
async def register_voice(request: VoiceRegisterRequest):
    """Register a voice profile"""
    try:
        # Decode audio
        audio_data = base64.b64decode(request.audio)
        
        # Create profile
        voice_id = await voice_manager.create_profile(
            audio_data=audio_data,
            reference_text=request.reference_text
        )
        
        return {
            "status": "success",
            "voice_id": voice_id,
            "message": "Voice registered successfully"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error registering voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/voices")
async def list_voices():
    """List all voice profiles"""
    voices = await voice_manager.list_profiles()
    return {"voices": voices, "count": len(voices)}


@app.delete("/api/voices/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete a voice profile"""
    success = await voice_manager.delete_profile(voice_id)
    if success:
        return {"status": "success", "message": "Voice deleted"}
    else:
        raise HTTPException(status_code=404, detail="Voice not found")


@app.post("/api/offer")
async def handle_offer(request: OfferRequest):
    """Handle WebRTC offer and return answer"""
    try:
        voice_id = request.voice_id
        text = request.text
        
        logger.info(f"Received WebRTC offer for voice: {voice_id}")
        logger.info(f"Text length: {len(text)} characters")
        logger.info(f"SDP length: {len(request.sdp)} characters")
        
        # Get voice profile
        voice_profile = await voice_manager.get_profile(voice_id)
        if voice_profile is None:
            logger.error(f"Voice profile not found: {voice_id}")
            raise HTTPException(status_code=404, detail=f"Voice profile not found: {voice_id}")
        
        logger.info("Voice profile loaded, starting TTS generation...")
        
        # Start resource monitoring
        start_metrics = resource_monitor.start_monitoring()
        
        # Start CPU sampling task (sample every 0.5 seconds during generation)
        cpu_sampling_task = None
        sampling_active = True
        
        async def sample_cpu_periodically():
            while sampling_active:
                resource_monitor.sample_cpu(start_metrics)
                await asyncio.sleep(0.5)
        
        cpu_sampling_task = asyncio.create_task(sample_cpu_periodically())
        
        # Generate speech
        async def tts_task():
            return await tts_engine.generate_speech(text, voice_profile)
        
        try:
            audio_array, sr = await request_queue.submit(tts_task)
        finally:
            # Stop CPU sampling
            sampling_active = False
            if cpu_sampling_task:
                cpu_sampling_task.cancel()
                try:
                    await cpu_sampling_task
                except asyncio.CancelledError:
                    pass
        
        # End resource monitoring and log
        end_metrics = resource_monitor.end_monitoring(start_metrics)
        resource_monitor.log_resources(end_metrics, voice_id, len(text))
        
        logger.info(f"TTS generation complete: {len(audio_array)} samples at {sr}Hz")
        
        # Save to disk for archive
        if config.SAVE_GENERATIONS:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{voice_id}_{timestamp}.wav"
            saved_path = config.GENERATIONS_DIR / filename
            sf.write(str(saved_path), audio_array, sr)
            logger.info(f"Saved generation to: {saved_path}")
        
        # Resample to 48kHz for WebRTC
        if sr != config.WEBRTC_SAMPLE_RATE:
            logger.info(f"Resampling from {sr}Hz to {config.WEBRTC_SAMPLE_RATE}Hz")
            audio_array = librosa.resample(
                audio_array,
                orig_sr=sr,
                target_sr=config.WEBRTC_SAMPLE_RATE
            )
        
        # Normalize audio
        max_val = np.abs(audio_array).max()
        if max_val > 0:
            audio_array = audio_array / max_val
        
        logger.info("Creating WebRTC peer connection...")
        
        # Create peer connection
        pc = RTCPeerConnection()
        connection_id = str(uuid.uuid4())
        peer_connections[connection_id] = pc
        
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection state: {pc.connectionState}")
            if pc.connectionState == "failed" or pc.connectionState == "closed":
                if connection_id in peer_connections:
                    del peer_connections[connection_id]
        
        logger.info("Setting remote description...")
        
        # Set remote description FIRST (before adding track)
        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=request.sdp, type=request.type)
        )
        
        logger.info("Creating and adding audio track...")
        
        # Create audio track and add it AFTER setting remote description
        audio_track = AudioStreamTrack(audio_array, config.WEBRTC_SAMPLE_RATE)
        pc.addTrack(audio_track)
        
        logger.info("Creating answer...")
        
        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        duration = len(audio_array) / config.WEBRTC_SAMPLE_RATE
        
        logger.info(f"WebRTC answer created for connection: {connection_id}")
        logger.info(f"Audio duration: {duration:.2f}s")
        
        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "connection_id": connection_id,
            "duration": duration
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error handling offer: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Run the server"""
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level="info",
        timeout_keep_alive=300,  # 5 minutes
        limit_concurrency=100,
        limit_max_requests=1000,
    )


if __name__ == "__main__":
    main()