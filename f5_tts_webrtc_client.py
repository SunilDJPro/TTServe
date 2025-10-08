"""
F5-TTS WebRTC Client
Python client for testing the F5-TTS WebRTC server
"""

import asyncio
import json
import base64
import sys
from pathlib import Path
import numpy as np

import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription
from av import AudioFrame
import soundfile as sf


class F5TTSWebRTCClient:
    """WebRTC client for F5-TTS server"""
    
    def __init__(self, server_url: str = "http://localhost:8766"):
        self.server_url = server_url
        self.session = None
    
    async def connect(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession()
        print(f"Connected to {self.server_url}")
    
    async def disconnect(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            print("Disconnected")
    
    async def register_voice(self, audio_path: str, reference_text: str) -> str:
        """Register a voice profile"""
        print(f"\nRegistering voice from: {audio_path}")
        print(f"Reference text: {reference_text}")
        
        # Read audio file
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        
        # Encode to base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        # Send request
        async with self.session.post(
            f"{self.server_url}/api/register_voice",
            json={
                "audio": audio_base64,
                "reference_text": reference_text
            }
        ) as response:
            result = await response.json()
            
            if result.get("status") == "success":
                voice_id = result.get("voice_id")
                print(f"✓ Voice registered successfully!")
                print(f"  Voice ID: {voice_id}")
                return voice_id
            else:
                print(f"✗ Error: {result.get('message')}")
                return None
    
    async def generate_speech_webrtc(
        self,
        voice_id: str,
        text: str,
        output_file: str = None
    ):
        """Generate speech using WebRTC and optionally save to file"""
        print(f"\nGenerating speech via WebRTC...")
        print(f"Voice ID: {voice_id}")
        print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        
        # Create peer connection
        pc = RTCPeerConnection()
        
        # Store received audio
        audio_frames = []
        
        @pc.on("track")
        async def on_track(track):
            print(f"✓ Receiving audio track: {track.kind}")
            
            if track.kind == "audio":
                try:
                    frame_count = 0
                    while True:
                        frame = await track.recv()
                        
                        # Use to_ndarray() which handles decoded format correctly
                        # For stereo: returns shape (1, samples) where samples are interleaved [L,R,L,R,...]
                        audio_data = frame.to_ndarray()
                        
                        # Debug first frame
                        if frame_count == 0:
                            print(f"  First frame shape: {audio_data.shape}, dtype: {audio_data.dtype}")
                            print(f"  Sample rate: {frame.sample_rate}, format: {frame.format.name}, layout: {frame.layout.name}")
                        
                        # Flatten to 1D
                        audio_data = audio_data.flatten()
                        
                        # De-interleave stereo: take only left channel (every 2nd sample starting at 0)
                        if frame.layout.name == 'stereo':
                            audio_data = audio_data[::2]  # [L0, L1, L2, ...]
                        
                        # Convert from int16 to float32
                        if audio_data.dtype == np.int16:
                            audio_data = audio_data.astype(np.float32) / 32768.0
                        
                        audio_frames.append(audio_data)
                        frame_count += 1
                        
                        # Progress indicator
                        if frame_count % 50 == 0:
                            print(f"  Received {frame_count} frames...")
                        
                except Exception as e:
                    if "StopIteration" not in str(type(e).__name__):
                        print(f"Track ended: {e}")
                    print(f"  Total frames: {frame_count}")
        
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"Connection state: {pc.connectionState}")
        
        # Add transceiver to receive audio (CRITICAL)
        pc.addTransceiver("audio", direction="recvonly")
        
        # Create offer
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        
        # Send offer to server
        print("✓ Sending WebRTC offer to server...")
        
        try:
            async with self.session.post(
                f"{self.server_url}/api/offer",
                json={
                    "sdp": pc.localDescription.sdp,
                    "type": pc.localDescription.type,
                    "voice_id": voice_id,
                    "text": text
                },
                timeout=aiohttp.ClientTimeout(total=300)  # 5 minute timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Server error (HTTP {response.status}): {error_text}")
                
                answer_data = await response.json()
                duration = answer_data.get("duration", 0)
                print(f"✓ Server accepted offer (audio duration: {duration:.2f}s)")
        
        except asyncio.TimeoutError:
            raise Exception("Server timeout - request took too long")
        except aiohttp.ClientError as e:
            raise Exception(f"Connection error: {str(e)}")
        except Exception as e:
            raise Exception(f"Request failed: {str(e)}")
        
        # Set remote description
        answer = RTCSessionDescription(
            sdp=answer_data["sdp"],
            type=answer_data["type"]
        )
        await pc.setRemoteDescription(answer)
        
        # Wait for audio to complete
        print("✓ Streaming audio...")
        await asyncio.sleep(duration + 2)  # Wait for full stream + buffer
        
        # Close connection
        await pc.close()
        
        # Process received audio
        if audio_frames:
            print(f"\n✓ Received {len(audio_frames)} audio frames")
            
            # Concatenate all frames directly (already 1D arrays)
            audio_array = np.concatenate(audio_frames)
            
            print(f"✓ Combined audio: {len(audio_array)} samples @ 48kHz")
            print(f"✓ Duration: {len(audio_array) / 48000:.2f} seconds")
            
            # Save to file if requested
            if output_file:
                # Save at 48kHz (WebRTC rate)
                sf.write(output_file, audio_array, 48000)
                print(f"✓ Audio saved to: {output_file}")
            
            return audio_array
        else:
            raise Exception("No audio frames received")
    
    async def delete_voice(self, voice_id: str) -> bool:
        """Delete a voice profile"""
        print(f"\nDeleting voice: {voice_id}")
        
        async with self.session.delete(
            f"{self.server_url}/api/voices/{voice_id}"
        ) as response:
            if response.status == 200:
                print(f"✓ Voice deleted successfully!")
                return True
            else:
                result = await response.json()
                print(f"✗ Error: {result.get('detail')}")
                return False
    
    async def list_voices(self) -> list:
        """List all voice profiles"""
        print("\nListing all voices...")
        
        async with self.session.get(
            f"{self.server_url}/api/voices"
        ) as response:
            result = await response.json()
            
            voices = result.get("voices", [])
            count = result.get("count", 0)
            
            print(f"✓ Found {count} voice(s):")
            for voice_id in voices:
                print(f"  - {voice_id}")
            
            return voices


async def test_full_workflow():
    """Test complete workflow: register voice, generate speech via WebRTC"""
    
    client = F5TTSWebRTCClient()
    
    try:
        # Connect
        await client.connect()
        
        # Example 1: Register a voice
        print("\n" + "="*60)
        print("TEST 1: Register Voice")
        print("="*60)
        
        # You need to provide a reference audio file
        reference_audio = "path/to/your/reference.wav"  # CHANGE THIS
        reference_text = "This is the reference text that matches the audio."  # CHANGE THIS
        
        if not Path(reference_audio).exists():
            print(f"\n⚠ Please update reference_audio path in the script!")
            print(f"Current path: {reference_audio}")
            print(f"\nUsage: python f5_tts_webrtc_client.py <reference_audio.wav> <reference_text>")
            return
        
        voice_id = await client.register_voice(reference_audio, reference_text)
        
        if not voice_id:
            print("Failed to register voice. Exiting.")
            return
        
        # Example 2: Generate speech via WebRTC
        print("\n" + "="*60)
        print("TEST 2: Generate Speech via WebRTC")
        print("="*60)
        
        test_texts = [
            "Hello, this is a test of the F5 TTS system using WebRTC streaming.",
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world in amazing ways."
        ]
        
        for i, text in enumerate(test_texts, 1):
            output_file = f"webrtc_output_{i}.wav"
            await client.generate_speech_webrtc(
                voice_id=voice_id,
                text=text,
                output_file=output_file
            )
            await asyncio.sleep(1)  # Brief pause between requests
        
        # Example 3: List voices
        print("\n" + "="*60)
        print("TEST 3: List All Voices")
        print("="*60)
        
        await client.list_voices()
        
        # Example 4: Delete voice (optional)
        print("\n" + "="*60)
        print("TEST 4: Delete Voice (optional)")
        print("="*60)
        
        user_input = input("\nDelete this voice profile? (y/n): ")
        if user_input.lower() == 'y':
            await client.delete_voice(voice_id)
        else:
            print("Voice profile kept on server")
        
        print("\n" + "="*60)
        print("All tests completed!")
        print("="*60)
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await client.disconnect()


async def interactive_mode():
    """Interactive mode for manual testing"""
    
    client = F5TTSWebRTCClient()
    
    try:
        await client.connect()
        
        current_voice_id = None
        
        while True:
            print("\n" + "="*60)
            print("F5-TTS WebRTC Client - Interactive Mode")
            print("="*60)
            print("1. Register a new voice")
            print("2. Generate speech (WebRTC)")
            print("3. List all voices")
            print("4. Delete a voice")
            print("5. Exit")
            print("="*60)
            
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == "1":
                audio_path = input("Enter reference audio path (.wav): ").strip()
                ref_text = input("Enter reference text: ").strip()
                
                if Path(audio_path).exists():
                    voice_id = await client.register_voice(audio_path, ref_text)
                    if voice_id:
                        current_voice_id = voice_id
                else:
                    print(f"✗ File not found: {audio_path}")
            
            elif choice == "2":
                if not current_voice_id:
                    voice_id = input("Enter voice ID: ").strip()
                else:
                    use_current = input(f"Use current voice ({current_voice_id})? (y/n): ").strip()
                    voice_id = current_voice_id if use_current.lower() == 'y' else input("Enter voice ID: ").strip()
                
                text = input("Enter text to synthesize: ").strip()
                output = input("Save to file (enter filename or press Enter to skip): ").strip()
                
                await client.generate_speech_webrtc(
                    voice_id=voice_id,
                    text=text,
                    output_file=output if output else None
                )
            
            elif choice == "3":
                await client.list_voices()
            
            elif choice == "4":
                voice_id = input("Enter voice ID to delete: ").strip()
                await client.delete_voice(voice_id)
                
                if voice_id == current_voice_id:
                    current_voice_id = None
            
            elif choice == "5":
                print("Exiting...")
                break
            
            else:
                print("Invalid option")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await client.disconnect()


async def simple_test():
    """Simple quick test"""
    
    # Check if reference audio path is provided as argument
    if len(sys.argv) < 3:
        print("Usage: python f5_tts_webrtc_client.py <reference_audio.wav> <reference_text>")
        print("\nOr run in interactive mode:")
        print("python f5_tts_webrtc_client.py --interactive")
        return
    
    reference_audio = sys.argv[1]
    reference_text = sys.argv[2]
    
    if not Path(reference_audio).exists():
        print(f"Error: File not found: {reference_audio}")
        return
    
    client = F5TTSWebRTCClient()
    
    try:
        await client.connect()
        
        # Register voice
        voice_id = await client.register_voice(reference_audio, reference_text)
        
        if voice_id:
            # Generate test speech via WebRTC
            test_text = "Hello! This is a test of the F5 TTS system using WebRTC streaming."
            await client.generate_speech_webrtc(
                voice_id=voice_id,
                text=test_text,
                output_file="webrtc_test_output.wav"
            )
    
    finally:
        await client.disconnect()


def main():
    """Main entry point"""
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        asyncio.run(interactive_mode())
    elif len(sys.argv) > 1 and sys.argv[1] == "--test":
        asyncio.run(test_full_workflow())
    else:
        asyncio.run(simple_test())


if __name__ == "__main__":
    main()