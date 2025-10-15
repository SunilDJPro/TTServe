"""
Custom exceptions for TTS Library
"""


class TTSLibraryError(Exception):
    """Base exception for TTS Library"""
    pass


class ModelNotLoadedError(TTSLibraryError):
    """Raised when TTS model is not loaded"""
    pass


class VoiceNotFoundError(TTSLibraryError):
    """Raised when voice ID is not found"""
    pass


class VoiceRegistrationError(TTSLibraryError):
    """Raised when voice registration fails"""
    pass


class AudioProcessingError(TTSLibraryError):
    """Raised when audio processing fails"""
    pass


class TTSGenerationError(TTSLibraryError):
    """Raised when TTS generation fails"""
    pass


class WebRTCError(TTSLibraryError):
    """Raised when WebRTC operations fail"""
    pass


class QueueError(TTSLibraryError):
    """Raised when queue operations fail"""
    pass

class InvalidAudioError(TTSLibraryError):
    """Raised when audio data is invalid"""
    pass


class LibraryNotInitializedError(TTSLibraryError):
    """Raised when library is used before initialization"""
    pass