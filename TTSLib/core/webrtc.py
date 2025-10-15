"""
WebRTC handler for generating SDP offers/answers
"""

import asyncio
import logging
import uuid
from typing import Optional, Dict
import numpy as np
import librosa
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from av import AudioFrame
from fractions import Fraction

from ..config import config
from ..exceptions import WebRTCError
from ..models import WebRTCResponse

logger = logging.getLogger(__name__)


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
        
        logger.debug(f"AudioStreamTrack: {len(audio_data)} samples at {sample_rate}Hz")
    
    async def recv(self):
        """Generate audio frames for WebRTC"""
        # Check if we've sent all audio
        if self.current_position >= len(self.audio_data):
            logger.debug(f"Audio streaming complete - {self.current_position} samples")
            await asyncio.sleep(0.1)
            raise StopAsyncIteration
        
        # Get next chunk of audio
        end_position = min(
            self.current_position + self.samples_per_frame,
            len(self.audio_data)
        )
        
        audio_chunk = self.audio_data[self.current_position:end_position]
        
        # Pad if necessary
        if len(audio_chunk) < self.samples_per_frame:
            audio_chunk = np.pad(
                audio_chunk,
                (0, self.samples_per_frame - len(audio_chunk)),
                mode='constant'
            )
        
        # Convert to int16 and create STEREO
        audio_int16 = (audio_chunk * 32767).astype(np.int16)
        
        # Interleave L and R channels
        audio_stereo = np.empty(len(audio_int16) * 2, dtype=np.int16)
        audio_stereo[0::2] = audio_int16  # Left
        audio_stereo[1::2] = audio_int16  # Right
        
        # Reshape for stereo format
        audio_stereo = audio_stereo.reshape(1, -1)
        
        # Create stereo AudioFrame
        frame = AudioFrame.from_ndarray(audio_stereo, format='s16', layout='stereo')
        frame.sample_rate = self.sample_rate
        frame.pts = self.timestamp
        frame.time_base = Fraction(1, self.sample_rate)
        
        # Update position and timestamp
        self.current_position = end_position
        self.timestamp += self.samples_per_frame
        
        # Simulate real-time streaming
        await asyncio.sleep(0.020)
        
        return frame


class WebRTCHandler:
    """Manages WebRTC connections and SDP generation"""
    
    def __init__(self):
        self.active_connections: Dict[str, RTCPeerConnection] = {}
        self._lock = asyncio.Lock()
        logger.info("WebRTCHandler initialized")
    
    async def create_answer(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        client_sdp_offer: str,
        client_sdp_type: str = "offer"
    ) -> WebRTCResponse:
        """
        Create WebRTC answer for client offer
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate of audio
            client_sdp_offer: SDP offer from client
            client_sdp_type: SDP type (default: "offer")
            
        Returns:
            WebRTCResponse with SDP answer
        """
        try:
            logger.info("Creating WebRTC answer...")
            
            # Resample to 48kHz for WebRTC if needed
            if sample_rate != config.WEBRTC_SAMPLE_RATE:
                logger.debug(f"Resampling {sample_rate}Hz -> {config.WEBRTC_SAMPLE_RATE}Hz")
                audio_array = librosa.resample(
                    audio_array,
                    orig_sr=sample_rate,
                    target_sr=config.WEBRTC_SAMPLE_RATE
                )
                sample_rate = config.WEBRTC_SAMPLE_RATE
            
            # Normalize audio
            max_val = np.abs(audio_array).max()
            if max_val > 0:
                audio_array = audio_array / max_val
            
            # Create peer connection
            pc = RTCPeerConnection()
            connection_id = str(uuid.uuid4())
            
            async with self._lock:
                self.active_connections[connection_id] = pc
            
            # Set up connection state handler
            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                logger.debug(f"Connection {connection_id}: {pc.connectionState}")
                if pc.connectionState in ["failed", "closed"]:
                    await self._remove_connection(connection_id)
            
            # Set remote description (client offer)
            logger.debug("Setting remote description...")
            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=client_sdp_offer, type=client_sdp_type)
            )
            
            # Create and add audio track
            logger.debug("Creating audio track...")
            audio_track = AudioStreamTrack(audio_array, sample_rate)
            pc.addTrack(audio_track)
            
            # Create answer
            logger.debug("Creating answer...")
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            
            duration = len(audio_array) / sample_rate
            
            logger.info(f"WebRTC answer created: {connection_id} ({duration:.2f}s)")
            
            return WebRTCResponse(
                sdp=pc.localDescription.sdp,
                type=pc.localDescription.type,
                connection_id=connection_id,
                duration=duration
            )
            
        except Exception as e:
            logger.error(f"Error creating WebRTC answer: {e}")
            raise WebRTCError(f"Failed to create WebRTC answer: {e}")
    
    async def _remove_connection(self, connection_id: str):
        """Remove connection from active connections"""
        async with self._lock:
            if connection_id in self.active_connections:
                try:
                    pc = self.active_connections[connection_id]
                    await pc.close()
                except Exception as e:
                    logger.error(f"Error closing connection {connection_id}: {e}")
                finally:
                    del self.active_connections[connection_id]
                    logger.debug(f"Removed connection: {connection_id}")
    
    async def close_connection(self, connection_id: str) -> bool:
        """Close a specific connection"""
        async with self._lock:
            if connection_id not in self.active_connections:
                return False
            
            await self._remove_connection(connection_id)
            return True
    
    async def close_all_connections(self):
        """Close all active connections"""
        async with self._lock:
            connection_ids = list(self.active_connections.keys())
        
        for conn_id in connection_ids:
            await self._remove_connection(conn_id)
        
        logger.info("All WebRTC connections closed")
    
    def get_active_connections(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)