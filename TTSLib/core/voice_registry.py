"""
Voice registry manager for tracking voice IDs
"""

import json
import uuid
import asyncio
import logging
from pathlib import Path
from typing import Optional, List
from ..config import config
from ..models import VoiceMetadata
from ..exceptions import VoiceNotFoundError, VoiceRegistrationError

logger = logging.getLogger(__name__)


class VoiceRegistry:
    """Manages voice ID registry with JSON persistence"""
    
    def __init__(self):
        self.registry_dir = config.VOICE_REGISTRY_DIR
        self.registry_file = self.registry_dir / config.VOICE_REGISTRY_FILE
        self._registry: dict[str, VoiceMetadata] = {}
        self._lock = asyncio.Lock()
        logger.info("VoiceRegistry initialized")
    
    async def initialize(self):
        """Initialize registry directory and load existing data"""
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        await self._load_registry()
        logger.info(f"Voice registry loaded: {len(self._registry)} voices")
    
    async def _load_registry(self):
        """Load registry from disk"""
        if not self.registry_file.exists():
            self._registry = {}
            await self._save_registry()
            return
        
        try:
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
            
            self._registry = {
                voice_id: VoiceMetadata.from_dict(metadata)
                for voice_id, metadata in data.items()
            }
            
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            self._registry = {}
    
    async def _save_registry(self):
        """Save registry to disk"""
        try:
            data = {
                voice_id: metadata.to_dict()
                for voice_id, metadata in self._registry.items()
            }
            
            # Write atomically using temp file
            temp_file = self.registry_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            temp_file.replace(self.registry_file)
            
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            raise VoiceRegistrationError(f"Failed to save registry: {e}")
    
    async def register(self, reference_text: str, sample_rate: int, duration: float) -> str:
        """Register a new voice and return voice_id"""
        async with self._lock:
            voice_id = str(uuid.uuid4())
            
            metadata = VoiceMetadata(
                voice_id=voice_id,
                reference_text=reference_text,
                sample_rate=sample_rate,
                duration=duration,
                created_at=asyncio.get_event_loop().time()
            )
            
            self._registry[voice_id] = metadata
            await self._save_registry()
            
            logger.info(f"Registered voice: {voice_id}")
            return voice_id
    
    async def get(self, voice_id: str) -> VoiceMetadata:
        """Get voice metadata by ID"""
        async with self._lock:
            if voice_id not in self._registry:
                raise VoiceNotFoundError(f"Voice ID not found: {voice_id}")
            return self._registry[voice_id]
    
    async def exists(self, voice_id: str) -> bool:
        """Check if voice ID exists"""
        async with self._lock:
            return voice_id in self._registry
    
    async def delete(self, voice_id: str) -> bool:
        """Delete a voice from registry"""
        async with self._lock:
            if voice_id not in self._registry:
                return False
            
            del self._registry[voice_id]
            await self._save_registry()
            
            logger.info(f"Deleted voice: {voice_id}")
            return True
    
    async def list_voices(self) -> List[str]:
        """List all voice IDs"""
        async with self._lock:
            return list(self._registry.keys())
    
    async def get_all_metadata(self) -> dict[str, VoiceMetadata]:
        """Get all voice metadata"""
        async with self._lock:
            return self._registry.copy()
    
    async def clear(self):
        """Clear all voices (for testing)"""
        async with self._lock:
            self._registry.clear()
            await self._save_registry()
            logger.warning("Registry cleared")