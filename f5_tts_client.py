"""
F5-TTS WebSocket Client
Example client for testing the F5-TTS WebSocket server
"""

import asyncio
import json
import base64
import wave
import sys
from pathlib import Path
import websockets


class F5TTSClient:
    """WebSocket client for F5-TTS server"""
    
    def __init__(self, server_url: str = "ws://localhost:8765/ws"):
        self.server_url = server_url
        self.websocket = None
    
    async def connect(self):
        """Connect to WebSocket server"""
        print(f"Connecting to {self.server_url}...")
        self.websocket = await websockets.connect(self.server_url)
        print("Connected!")
    
    async def disconnect(self):
        """Disconnect from server"""
        if self.websocket:
            await self.websocket.close()
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
        request = {
            "action": "register_voice",
            "audio": audio_base64,
            "reference_text": reference_text
        }
        
        await self.websocket.send(json.dumps(request))
        
        # Receive response
        response = await self.websocket.recv()
        result = json.loads(response)
        
        if result.get("status") == "success":
            voice_id = result.get("voice_id")
            print(f"✓ Voice registered successfully!")
            print(f"  Voice ID: {voice_id}")
            return voice_id
        else:
            print(f"✗ Error: {result.get('message')}")
            return None
    
    async def generate_speech(self, voice_id: str, text: str, output_file: str = None):
        """Generate speech from text"""
        await self.websocket.send(json.dumps({
            "action": "tts",
            "voice_id": voice_id,
            "text": text,
            "save_file": bool(output_file)
        }))
        
        response = await self.websocket.recv()
        data = json.loads(response)
        
        # Handle chunked response
        if data.get("status") == "streaming":
            print(f"Receiving large audio in {data['total_chunks']} chunks...")
            audio_chunks = []
            
            # Receive all chunks
            for i in range(data['total_chunks']):
                chunk_msg = await self.websocket.recv()
                chunk_data = json.loads(chunk_msg)
                
                if chunk_data.get("status") == "chunk":
                    audio_chunks.append(base64.b64decode(chunk_data["chunk"]))
                    print(f"Received chunk {chunk_data['chunk_index'] + 1}/{data['total_chunks']}")
            
            # Wait for completion message
            complete_msg = await self.websocket.recv()
            complete_data = json.loads(complete_msg)
            
            if complete_data.get("status") == "complete":
                # Combine all chunks
                audio_data = b''.join(audio_chunks)
                
                if output_file:
                    with open(output_file, 'wb') as f:
                        f.write(audio_data)
                    print(f"✓ Audio saved to: {output_file}")
                
                return audio_data
        
        # Handle single message response
        elif data.get("status") == "success":
            audio_data = base64.b64decode(data["audio"])
            
            if output_file:
                with open(output_file, 'wb') as f:
                    f.write(audio_data)
                print(f"✓ Audio saved to: {output_file}")
            
            return audio_data
        
        else:
            raise Exception(data.get("message", "Unknown error"))
    
    async def delete_voice(self, voice_id: str) -> bool:
        """Delete a voice profile"""
        print(f"\nDeleting voice: {voice_id}")
        
        request = {
            "action": "delete_voice",
            "voice_id": voice_id
        }
        
        await self.websocket.send(json.dumps(request))
        
        # Receive response
        response = await self.websocket.recv()
        result = json.loads(response)
        
        if result.get("status") == "success":
            print(f"✓ Voice deleted successfully!")
            return True
        else:
            print(f"✗ Error: {result.get('message')}")
            return False
    
    async def list_voices(self) -> list:
        """List all voice profiles"""
        print("\nListing all voices...")
        
        request = {
            "action": "list_voices"
        }
        
        await self.websocket.send(json.dumps(request))
        
        # Receive response
        response = await self.websocket.recv()
        result = json.loads(response)
        
        if result.get("status") == "success":
            voices = result.get("voices", [])
            count = result.get("count", 0)
            
            print(f"✓ Found {count} voice(s):")
            for voice_id in voices:
                print(f"  - {voice_id}")
            
            return voices
        else:
            print(f"✗ Error: {result.get('message')}")
            return []


async def test_full_workflow():
    """Test complete workflow: register voice, generate speech, cleanup"""
    
    client = F5TTSClient()
    
    try:
        # Connect
        await client.connect()
        
        # Example 1: Register a voice
        print("\n" + "="*60)
        print("TEST 1: Register Voice")
        print("="*60)
        
        # You need to provide a reference audio file (10 seconds)
        reference_audio = "path/to/your/reference.wav"  # CHANGE THIS
        reference_text = "This is the reference text that matches the audio."  # CHANGE THIS
        
        if not Path(reference_audio).exists():
            print(f"\n⚠ Please update reference_audio path in the script!")
            print(f"Current path: {reference_audio}")
            return
        
        voice_id = await client.register_voice(reference_audio, reference_text)
        
        if not voice_id:
            print("Failed to register voice. Exiting.")
            return
        
        # Example 2: Generate speech
        print("\n" + "="*60)
        print("TEST 2: Generate Speech")
        print("="*60)
        
        test_texts = [
            "Hello, this is a test of the F5 TTS system.",
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world in amazing ways."
        ]
        
        for i, text in enumerate(test_texts, 1):
            output_file = f"output_test_{i}.wav"
            await client.generate_speech(
                voice_id=voice_id,
                text=text,
                output_file=output_file,
                save_on_server=True
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
    
    client = F5TTSClient()
    
    try:
        await client.connect()
        
        current_voice_id = None
        
        while True:
            print("\n" + "="*60)
            print("F5-TTS WebSocket Client - Interactive Mode")
            print("="*60)
            print("1. Register a new voice")
            print("2. Generate speech")
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
                
                await client.generate_speech(
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
        print("Usage: python f5_tts_client.py <reference_audio.wav> <reference_text>")
        print("\nOr run in interactive mode:")
        print("python f5_tts_client.py --interactive")
        return
    
    reference_audio = sys.argv[1]
    reference_text = sys.argv[2]
    
    if not Path(reference_audio).exists():
        print(f"Error: File not found: {reference_audio}")
        return
    
    client = F5TTSClient()
    
    try:
        await client.connect()
        
        # Register voice
        voice_id = await client.register_voice(reference_audio, reference_text)
        
        if voice_id:
            # Generate test speech
            test_text = "Hello! This is a test of the F5 TTS system."
            await client.generate_speech(
                voice_id=voice_id,
                text=test_text,
                output_file="test_output.wav"
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