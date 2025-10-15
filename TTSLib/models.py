"""
Data models for TTS Library
"""

from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class VoiceMetadata:
    """Voice profile metadata"""
    voice_id: str
    reference_text: str
    sample_rate: int
    duration: float
    created_at: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'voice_id': self.voice_id,
            'reference_text': self.reference_text,
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'VoiceMetadata':
        """Create from dictionary"""
        return cls(
            voice_id=data['voice_id'],
            reference_text=data['reference_text'],
            sample_rate=data['sample_rate'],
            duration=data['duration'],
            created_at=data['created_at']
        )


@dataclass
class VoiceRegistrationResult:
    """Result of voice registration"""
    voice_id: str
    audio_bytes: bytes
    metadata: VoiceMetadata


@dataclass
class WebRTCResponse:
    """WebRTC SDP response"""
    sdp: str
    type: str
    connection_id: str
    duration: float