#!/usr/bin/env python3
"""
Real-world system test for TTS Library
Tests actual functionality with real F5-TTS models and audio files
"""

import asyncio
import sys, os
from pathlib import Path
import soundfile as sf
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TTSLib import TTSLibrary


# Configuration
INPUT_AUDIO = Path("tests/nareshAI10.wav")
REFERENCE_TEXT = "I stand before you today to discuss a topic that is revolutionizing our world in ways we would have only imagined few decades ago"  # Replace with actual transcript
OUTPUT_DIR = Path("tests/results")
REGISTERED_VOICE_OUTPUT = Path("tests/registered_vs_naresh.wav")


def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def print_step(step_num, description):
    """Print a step with formatting"""
    print(f"\n[Step {step_num}] {description}")
    print("-" * 70)


async def main():
    
    print_header("TTSLib SYSTEM LEVEL TEST")
    
    # Check if input file exists
    if not INPUT_AUDIO.exists():
        print(f"ERROR: Input audio file not found: {INPUT_AUDIO}")
        print(f"   Please ensure {INPUT_AUDIO} exists")
        return 1
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_AUDIO.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Input audio: {INPUT_AUDIO}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Reference text: '{REFERENCE_TEXT}'")
    
    # Step 1: Initialize library
    print_step(1, "Initializing TTS Library")
    print("Loading F5-TTS model to GPU... (this may take a minute)")
    
    try:
        tts = await TTSLibrary.initialize(
            max_workers=2,
            device="cuda"  # Change to "cpu" if no GPU available
        )
        print("Library initialized successfully")
        print(f"  - Model loaded: {tts.is_model_loaded()}")
        print(f"  - Active workers: {tts.get_active_workers()}")
    except Exception as e:
        print(f"Failed to initialize library: {e}")
        return 1
    
    try:
        # Step 2: Register voice
        print_step(2, "Registering Voice")
        print(f"Reading audio from: {INPUT_AUDIO}")
        
        with open(INPUT_AUDIO, "rb") as f:
            audio_data = f.read()
        
        print(f"Audio file size: {len(audio_data):,} bytes")
        print("Registering voice...")
        
        voice_id, processed_audio = await tts.register_voice(
            audio_data=audio_data,
            reference_text=REFERENCE_TEXT
        )
        
        print(f"✓ Voice registered successfully!")
        print(f"  - Voice ID: {voice_id}")
        print(f"  - Processed audio size: {len(processed_audio):,} bytes")
        
        # Save registered voice sample
        print(f"\nSaving registered voice sample to: {REGISTERED_VOICE_OUTPUT}")
        with open(REGISTERED_VOICE_OUTPUT, "wb") as f:
            f.write(processed_audio)
        print("✓ Voice sample saved")
        
        # Get voice metadata
        metadata = await tts.get_voice_metadata(voice_id)
        print(f"\nVoice Metadata:")
        print(f"  - Duration: {metadata.duration:.2f}s")
        print(f"  - Sample rate: {metadata.sample_rate}Hz")
        print(f"  - Reference text: {metadata.reference_text}")
        
        # Step 3: List registered voices
        print_step(3, "Listing Registered Voices")
        voices = await tts.list_voices()
        print(f"Total registered voices: {len(voices)}")
        for i, vid in enumerate(voices, 1):
            print(f"  {i}. {vid}")
        
        # Step 4: Generate speech (WAV)
        print_step(4, "Generating Speech - WAV Format")
        print("\nEnter text to synthesize (or press Enter for default):")
        user_text = input("> ").strip()
        
        if not user_text:
            user_text = "This is a test of the text to speech system using my cloned voice."
            print(f"Using default text: '{user_text}'")
        
        print(f"\nGenerating speech for: '{user_text}'")
        print("Please wait... (TTS generation in progress)")
        
        wav_bytes = await tts.generate_wav(
            voice_id=voice_id,
            reference_audio=processed_audio,
            text=user_text
        )
        
        print(f"Speech generated successfully!")
        print(f"  - Output size: {len(wav_bytes):,} bytes")
        
        # Save WAV output
        wav_output_path = OUTPUT_DIR / f"output_wav_{voice_id[:8]}.wav"
        with open(wav_output_path, "wb") as f:
            f.write(wav_bytes)
        
        print(f"Saved to: {wav_output_path}")
        
        # Get audio info
        audio_array, sr = sf.read(wav_output_path)
        duration = len(audio_array) / sr
        print(f"\nAudio Details:")
        print(f"  - Duration: {duration:.2f}s")
        print(f"  - Sample rate: {sr}Hz")
        print(f"  - Samples: {len(audio_array):,}")
        
        # Step 5: Generate speech (WebRTC)
        print_step(5, "Generating Speech - WebRTC Stream")
        print("\nEnter text for WebRTC stream (or press Enter for default):")
        webrtc_text = input("> ").strip()
        
        if not webrtc_text:
            webrtc_text = "Testing WebRTC streaming with real-time audio synthesis."
            print(f"Using default text: '{webrtc_text}'")
        
        print(f"\nGenerating WebRTC stream for: '{webrtc_text}'")
        print("Creating mock SDP offer...")
        
        # Create a proper SDP offer with ICE credentials
        mock_sdp_offer = """v=0
o=- 4611731400430051336 2 IN IP4 127.0.0.1
s=-
t=0 0
a=group:BUNDLE 0
a=msid-semantic: WMS
m=audio 9 UDP/TLS/RTP/SAVPF 111
c=IN IP4 0.0.0.0
a=rtcp:9 IN IP4 0.0.0.0
a=ice-ufrag:testufrag123
a=ice-pwd:testpassword1234567890123456
a=ice-options:trickle
a=fingerprint:sha-256 00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00
a=setup:actpass
a=mid:0
a=sendrecv
a=rtcp-mux
a=rtpmap:111 opus/48000/2
a=fmtp:111 minptime=10;useinbandfec=1
"""
        
        print("Generating WebRTC answer...")
        
        webrtc_response = await tts.generate_webrtc_stream(
            voice_id=voice_id,
            reference_audio=processed_audio,
            text=webrtc_text,
            client_sdp_offer=mock_sdp_offer,
            client_sdp_type="offer"
        )
        
        print(f"WebRTC stream created successfully!")
        print(f"\nWebRTC Response Details:")
        print(f"  - Connection ID: {webrtc_response.connection_id}")
        print(f"  - Duration: {webrtc_response.duration:.2f}s")
        print(f"  - SDP Type: {webrtc_response.type}")
        print(f"  - SDP Length: {len(webrtc_response.sdp)} characters")
        
        # Save WebRTC response details
        webrtc_output_path = OUTPUT_DIR / f"output_webrtc_{voice_id[:8]}.txt"
        with open(webrtc_output_path, "w") as f:
            f.write(f"Connection ID: {webrtc_response.connection_id}\n")
            f.write(f"Duration: {webrtc_response.duration}s\n")
            f.write(f"Type: {webrtc_response.type}\n")
            f.write(f"\nSDP Answer:\n")
            f.write("-" * 70 + "\n")
            f.write(webrtc_response.sdp)
        
        print(f"WebRTC details saved to: {webrtc_output_path}")
        
        # Note: To actually capture the WebRTC audio stream, we'd need a real WebRTC client
        # For now, we'll generate the same audio as WAV for reference
        print("\nGenerating reference audio (same content as WebRTC stream)...")
        
        webrtc_wav_bytes = await tts.generate_wav(
            voice_id=voice_id,
            reference_audio=processed_audio,
            text=webrtc_text
        )
        
        webrtc_wav_path = OUTPUT_DIR / f"output_webrtc_reference_{voice_id[:8]}.wav"
        with open(webrtc_wav_path, "wb") as f:
            f.write(webrtc_wav_bytes)
        
        print(f"WebRTC reference audio saved to: {webrtc_wav_path}")
        
        # Step 6: System status
        print_step(6, "System Status")
        print(f"Queue size: {tts.get_queue_size()}")
        print(f"Active workers: {tts.get_active_workers()}")
        print(f"WebRTC connections: {tts.get_active_webrtc_connections()}")
        print(f"Model loaded: {tts.is_model_loaded()}")
        
        # Step 7: Test voice management
        print_step(7, "Testing Voice Management")
        
        print(f"\nChecking if voice exists: {voice_id[:16]}...")
        exists = await tts.voice_exists(voice_id)
        print(f"Voice exists: {exists}")
        
        print("\nDo you want to delete this test voice? (y/N): ", end="")
        delete_choice = input().strip().lower()
        
        if delete_choice == 'y':
            print(f"Deleting voice: {voice_id}")
            deleted = await tts.delete_voice(voice_id)
            print(f"Voice deleted: {deleted}")
            
            # Verify deletion
            exists_after = await tts.voice_exists(voice_id)
            print(f"Voice exists after deletion: {exists_after}")
        else:
            print("Keeping voice registered")
        
        # Final summary
        print_header("TEST SUMMARY")
        
        print("All tests completed successfully!")
        print("\nGenerated Files:")
        print(f"  1. {REGISTERED_VOICE_OUTPUT}")
        print(f"  2. {wav_output_path}")
        print(f"  3. {webrtc_output_path}")
        print(f"  4. {webrtc_wav_path}")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR during test: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # Step 8: Cleanup
        print_step(8, "Cleanup")
        print("Shutting down TTS Library...")
        await tts.shutdown()
        print("Library shutdown complete")
        print("\n" + "="*70)


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
