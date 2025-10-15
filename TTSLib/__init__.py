"""
TTS Library - Main public API
A modular library for F5-TTS with WebRTC streaming support
"""

import asyncio
import logging
import io
from typing import Optional, List, Tuple
import soundfile as sf

from .config import config
from .models import VoiceRegistrationResult, VoiceMetadata, WebRTCResponse
from .exceptions import (
    LibraryNotInitializedError,
    VoiceNotFoundError,
    AudioProcessingError,
    TTSGenerationError
)
from .core.engine import F5TTSEngine
from .core.voice_registry import VoiceRegistry
from .core.queue import AsyncQueueManager
from .core.webrtc import WebRTCHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TTSLibrary:
    """
    Main TTS Library class for voice cloning and speech synthesis
    
    Features:
    - Voice registration and management
    - Direct WAV generation
    - WebRTC streaming support
    - Async queue management for concurrent requests
    - GPU memory management
    
    Example:
        tts = await TTSLibrary.initialize(max_workers=2, device="cuda")
        
        # Register voice
        voice_id, audio_bytes = await tts.register_voice(audio_data, "Hello world")
        
        # Generate speech
        wav_bytes = await tts.generate_wav(voice_id, audio_bytes, "Test speech")
        
        # WebRTC streaming
        response = await tts.generate_webrtc_stream(voice_id, audio_bytes, "Test", sdp_offer)
        
        await tts.shutdown()
    """
    
    _instance: Optional['TTSLibrary'] = None
    _initialized: bool = False
    
    def __init__(self):
        """Private constructor - use initialize() instead"""
        self.engine: Optional[F5TTSEngine] = None
        self.voice_registry: Optional[VoiceRegistry] = None
        self.queue_manager: Optional[AsyncQueueManager] = None
        self.webrtc_handler: Optional[WebRTCHandler] = None
        self._shutdown = False
        logger.info("TTSLibrary instance created")
    
    @classmethod
    async def initialize(
        cls,
        max_workers: int = 1,
        device: str = "cuda"
    ) -> 'TTSLibrary':
        """
        Initialize the TTS Library (singleton)
        
        Args:
            max_workers: Number of concurrent TTS workers (default: 1)
            device: Device for model inference ("cuda" or "cpu")
            
        Returns:
            Initialized TTSLibrary instance
        """
        if cls._instance is not None and cls._initialized:
            logger.warning("Library already initialized, returning existing instance")
            return cls._instance
        
        logger.info("Initializing TTS Library...")
        
        instance = cls()
        
        # Initialize components
        instance.engine = F5TTSEngine()
        instance.voice_registry = VoiceRegistry()
        instance.queue_manager = AsyncQueueManager(max_workers=max_workers)
        instance.webrtc_handler = WebRTCHandler()
        
        # Load model
        await instance.engine.load_model(device=device)
        
        # Initialize registry
        await instance.voice_registry.initialize()
        
        # Start queue manager
        await instance.queue_manager.start()
        
        cls._instance = instance
        cls._initialized = True
        
        logger.info("TTS Library initialized successfully")
        return instance
    
    @classmethod
    def get_instance(cls) -> 'TTSLibrary':
        """Get the singleton instance"""
        if cls._instance is None or not cls._initialized:
            raise LibraryNotInitializedError(
                "Library not initialized. Call TTSLibrary.initialize() first."
            )
        return cls._instance
    
    def _check_initialized(self):
        """Check if library is initialized"""
        if not self._initialized or self._shutdown:
            raise LibraryNotInitializedError("Library not initialized or already shutdown")
    
    async def register_voice(
        self,
        audio_data: bytes,
        reference_text: str
    ) -> Tuple[str, bytes]:
        """
        Register a voice profile
        
        Args:
            audio_data: Audio data as bytes (any format, will be converted to 24kHz WAV)
            reference_text: Transcript of the audio
            
        Returns:
            Tuple of (voice_id, processed_audio_bytes)
            - voice_id: UUID for the registered voice
            - processed_audio_bytes: Audio processed to 24kHz WAV format
            
        Raises:
            AudioProcessingError: If audio processing fails
            VoiceRegistrationError: If registration fails
        """
        self._check_initialized()
        
        logger.info(f"Registering voice with text: {reference_text[:50]}...")
        
        # Validate and process audio
        processed_audio, sr, duration = await self.engine.validate_and_process_audio(
            audio_data,
            min_duration=config.MIN_REFERENCE_DURATION,
            max_duration=config.MAX_REFERENCE_DURATION
        )
        
        # Register in voice registry
        voice_id = await self.voice_registry.register(
            reference_text=reference_text,
            sample_rate=sr,
            duration=duration
        )
        
        logger.info(f"Voice registered: {voice_id} ({duration:.2f}s)")
        
        return voice_id, processed_audio
    
    async def generate_wav(
        self,
        voice_id: str,
        reference_audio: bytes,
        text: str
    ) -> bytes:
        """
        Generate speech as WAV file
        
        Args:
            voice_id: Registered voice ID
            reference_audio: Reference audio bytes (24kHz WAV)
            text: Text to synthesize
            
        Returns:
            Generated speech as WAV bytes (24kHz)
            
        Raises:
            VoiceNotFoundError: If voice_id is not found
            TTSGenerationError: If generation fails
        """
        self._check_initialized()
        
        # Verify voice exists
        metadata = await self.voice_registry.get(voice_id)
        
        logger.info(f"Generating WAV for voice: {voice_id}")
        
        # Submit to queue for processing
        async def tts_task():
            return await self.engine.generate_speech(
                text=text,
                reference_audio=reference_audio,
                reference_text=metadata.reference_text
            )
        
        audio_array, sr = await self.queue_manager.submit(tts_task)
        
        # Convert to WAV bytes
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio_array, sr, format='WAV')
        wav_bytes = audio_buffer.getvalue()
        
        logger.info(f"Generated WAV: {len(wav_bytes)} bytes")
        
        return wav_bytes
    
    async def generate_webrtc_stream(
        self,
        voice_id: str,
        reference_audio: bytes,
        text: str,
        client_sdp_offer: str,
        client_sdp_type: str = "offer"
    ) -> WebRTCResponse:
        """
        Generate speech and stream via WebRTC
        
        Args:
            voice_id: Registered voice ID
            reference_audio: Reference audio bytes (24kHz WAV)
            text: Text to synthesize
            client_sdp_offer: SDP offer from client
            client_sdp_type: SDP type (default: "offer")
            
        Returns:
            WebRTCResponse with SDP answer and connection details
            
        Raises:
            VoiceNotFoundError: If voice_id is not found
            TTSGenerationError: If generation fails
            WebRTCError: If WebRTC setup fails
        """
        self._check_initialized()
        
        # Verify voice exists
        metadata = await self.voice_registry.get(voice_id)
        
        logger.info(f"Generating WebRTC stream for voice: {voice_id}")
        
        # Submit to queue for processing
        async def tts_task():
            return await self.engine.generate_speech(
                text=text,
                reference_audio=reference_audio,
                reference_text=metadata.reference_text
            )
        
        audio_array, sr = await self.queue_manager.submit(tts_task)
        
        # Create WebRTC answer
        response = await self.webrtc_handler.create_answer(
            audio_array=audio_array,
            sample_rate=sr,
            client_sdp_offer=client_sdp_offer,
            client_sdp_type=client_sdp_type
        )
        
        logger.info(f"WebRTC stream created: {response.connection_id}")
        
        return response
    
    async def list_voices(self) -> List[str]:
        """
        List all registered voice IDs
        
        Returns:
            List of voice IDs
        """
        self._check_initialized()
        return await self.voice_registry.list_voices()
    
    async def get_voice_metadata(self, voice_id: str) -> VoiceMetadata:
        """
        Get metadata for a voice
        
        Args:
            voice_id: Voice ID
            
        Returns:
            VoiceMetadata object
            
        Raises:
            VoiceNotFoundError: If voice_id is not found
        """
        self._check_initialized()
        return await self.voice_registry.get(voice_id)
    
    async def delete_voice(self, voice_id: str) -> bool:
        """
        Delete a voice profile
        
        Args:
            voice_id: Voice ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        self._check_initialized()
        return await self.voice_registry.delete(voice_id)
    
    async def voice_exists(self, voice_id: str) -> bool:
        """
        Check if a voice ID exists
        
        Args:
            voice_id: Voice ID to check
            
        Returns:
            True if exists, False otherwise
        """
        self._check_initialized()
        return await self.voice_registry.exists(voice_id)
    
    def get_queue_size(self) -> int:
        """Get current queue size"""
        self._check_initialized()
        return self.queue_manager.queue_size
    
    def get_active_workers(self) -> int:
        """Get number of active workers"""
        self._check_initialized()
        return self.queue_manager.active_workers
    
    def get_active_webrtc_connections(self) -> int:
        """Get number of active WebRTC connections"""
        self._check_initialized()
        return self.webrtc_handler.get_active_connections()
    
    def is_model_loaded(self) -> bool:
        """Check if TTS model is loaded"""
        if self.engine is None:
            return False
        return self.engine.is_loaded()
    
    async def shutdown(self):
        """Shutdown the library and cleanup resources"""
        if self._shutdown:
            logger.warning("Library already shutdown")
            return
        
        logger.info("Shutting down TTS Library...")
        
        self._shutdown = True
        
        # Stop queue manager
        if self.queue_manager is not None:
            await self.queue_manager.stop()
        
        # Close WebRTC connections
        if self.webrtc_handler is not None:
            await self.webrtc_handler.close_all_connections()
        
        # Unload model
        if self.engine is not None:
            await self.engine.unload_model()
        
        # Reset singleton
        TTSLibrary._instance = None
        TTSLibrary._initialized = False
        
        logger.info("TTS Library shutdown complete")


# Convenience function for quick initialization
async def create_tts_library(max_workers: int = 1, device: str = "cuda") -> TTSLibrary:
    """
    Convenience function to create and initialize TTS Library
    
    Args:
        max_workers: Number of concurrent TTS workers
        device: Device for inference ("cuda" or "cpu")
        
    Returns:
        Initialized TTSLibrary instance
    """
    return await TTSLibrary.initialize(max_workers=max_workers, device=device)