import os
import torch
import torchaudio
import numpy as np
import asyncio
import time
import logging
import sounddevice as sd
import wave
import edge_tts
import playsound
from scipy.spatial.distance import cosine
import noisereduce as nr  
from model_loader import model  # To import the preloaded model

logging.basicConfig(level=logging.INFO)

class VoiceRecognition:
    def __init__(self, dataset_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dataset_path = dataset_path
        self.known_speakers, self.speaker_thresholds = self.load_known_speakers()
        for speaker, thresh in self.speaker_thresholds.items():
            print(f"Speaker: {speaker}, Max Distance Threshold: {thresh:.4f}")

    def record_audio(self, filename="live_input.wav", duration=3, samplerate=16000):
        print("🎙 Recording... Speak now!")
        audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
        sd.wait()
        
        # Convert audio_data to float32 
        audio_data_float = audio_data.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
        
        # Apply noise reduction
        try:
            reduced_noise = nr.reduce_noise(y=audio_data_float.flatten(), sr=samplerate)
            # Convert back to int16 for WAV file
            audio_data_reduced = (reduced_noise * 32768).astype(np.int16)
        except Exception as e:
            logging.error(f"Failed to apply noise reduction: {e}")
            audio_data_reduced = audio_data  
        
        # Save the processed audio to WAV file
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(samplerate)
            wf.writeframes(audio_data_reduced.tobytes())
    
        return filename

    def extract_embedding(self, audio_file):
        try:
            signal, fs = torchaudio.load(audio_file, normalize=True)
            signal = signal.numpy().flatten()
            
            # Apply noise reduction to the loaded audio
            try:
                reduced_signal = nr.reduce_noise(y=signal, sr=fs)
            except Exception as e:
                logging.error(f"Failed to apply noise reduction to {audio_file}: {e}")
                reduced_signal = signal
            
            # Convert back to torch tensor
            signal = torch.from_numpy(reduced_signal).float().reshape(1, -1)
            signal = signal.to(self.device)
            
            if fs != 16000:
                transform = torchaudio.transforms.Resample(fs, 16000).to(self.device)
                signal = transform(signal)
            with torch.no_grad():
                embedding = self.model.encode_batch(signal).squeeze().cpu().numpy()
            embedding = embedding.flatten()
            # No padding/truncation for  (192-dim embeddings)
            embedding /= np.linalg.norm(embedding)
            if np.isnan(embedding).any() or np.isinf(embedding).any():
                raise ValueError(f"Invalid embedding extracted from {audio_file}")
            return embedding
        except Exception as e:
            logging.error(f"Failed to extract embedding from {audio_file}: {e}")
            raise

    def load_known_speakers(self):
        speaker_embeddings = {}  # {speaker: [embedding1, embedding2, ...]}
        speaker_thresholds = {}  # {speaker: max_distance}
        
        for speaker in os.listdir(self.dataset_path):
            speaker_path = os.path.join(self.dataset_path, speaker)
            if os.path.isdir(speaker_path):
                embeddings_list = []
                audio_files = [f for f in os.listdir(speaker_path) if f.endswith(".wav")]
                if len(audio_files) < 3:
                    logging.warning(f"Skipping {speaker}: insufficient samples ({len(audio_files)} < 3)")
                    continue
                for file in audio_files:
                    file_path = os.path.join(speaker_path, file)
                    try:
                        embedding = self.extract_embedding(file_path)
                        embeddings_list.append(embedding)
                    except Exception as e:
                        logging.error(f"Error processing {file_path}: {e}")
                        continue
                if len(embeddings_list) < 3:
                    logging.warning(f"Skipping {speaker}: too few valid samples after processing")
                    continue
                
                speaker_embeddings[speaker] = embeddings_list
                
                distances = []
                for i, emb1 in enumerate(embeddings_list):
                    for j, emb2 in enumerate(embeddings_list):
                        if i < j:  # Avoiding self-comparison 
                            dist = cosine(emb1, emb2)
                            distances.append(dist)
                if distances:
                    max_distance = np.max(distances) * 1.5 
                    speaker_thresholds[speaker] = min(max_distance, 0.5)  # Cap at 0.5
                else:
                    speaker_thresholds[speaker] = 0.5  # Default threshold

        if not speaker_embeddings:
            print("Error:No valid speakers loaded. Check dataset.")
            return {}, {}
        
        return speaker_embeddings, speaker_thresholds

    async def speak(self, text):
        communicate = edge_tts.Communicate(text, "en-US-JennyNeural")
        await communicate.save("aiva_response.mp3")
        playsound.playsound("aiva_response.mp3")

    async def identify_speaker(self, audio_file):
        if not self.known_speakers:
            print("No known speakers registered.")
            await self.speak("Sorry, I don’t recognize this voice.")
            return "Unknown"

        start_time = time.time()

        try:
            live_embedding = self.extract_embedding(audio_file)
        except Exception as e:
            print(f"Error extracting live embedding: {e}")
            await self.speak("Sorry, I couldn’t process your voice.")
            return "Unknown"

        best_speaker = None
        best_distance = float('inf')
        
        # Compare to all known embeddings
        for speaker, embeddings in self.known_speakers.items():
            for emb in embeddings:
                distance = cosine(live_embedding, emb)
                if distance < best_distance:
                    best_distance = distance
                    best_speaker = speaker

        threshold = self.speaker_thresholds.get(best_speaker, 0.5)  # Default tight threshold

        #logging.info(f"Best match: {best_speaker} (distance: {best_distance:.4f}, threshold: {threshold:.4f})")

        if best_distance < threshold:
            recognized_speaker = best_speaker
            print(f" Recognized as {recognized_speaker}")
        else:
            recognized_speaker = "Unknown"
            print(f" Unknown (distance {best_distance:.4f} >= threshold {threshold:.4f})")

        end_time = time.time()
        recognition_time = end_time - start_time  # Time in seconds
        print(f"Execution_time: {recognition_time:.2f} seconds")

        await self.speak(f"Hello {recognized_speaker}, How can I help you?" 
                        if recognized_speaker != "Unknown" else "Sorry, I don’t recognize this voice.")
        #print("\n🎤 Recognized Speaker:", recognized_speaker)
        
    
        user_preferences = {
                "Charvina": { 
                    "AC Temperature": "21°C",
                    "Lighting": "Bright Cool",
                    "Music": "Lo-fi Focus",
                    "Blinds": "Quarter Open",
                    "TV Screen Mode": "Off",
                    "Water Temperature": "80°C" 
                    
                },
                "Vidhi": { 
                    "AC Temperature": "24°C", 
                    "Lighting": "Cool White", 
                    "Music": "Rock Classics", 
                    "Blinds": "Fully Open" 

                },
                "Isabella": { 
                    "AC Temperature": "20°C", 
                    "Lighting": "Dimmed Warm", 
                    "Music": "Classic Indie", 
                    "Blinds": "Half Open" 
                    
                },
                "Noah": {
                    "AC Temperature": "20°C", 
                    "Lighting": "Dimmed Warm", 
                    "Music": "Classic Indie", 
                    "Blinds": "Half Open"

                },
                "Oscar":{
                    "AC Temperature": "20°C", 
                    "Lighting": "Dimmed Warm", 
                    "Music": "Classic Indie", 
                    "Blinds": "Half Open",
                    "TV Screen": "Cinema",
                    "Water Temperature": "60°C"
                    
                }
            }
            
        if recognized_speaker in user_preferences:
                prefs = user_preferences[recognized_speaker]
                print("\n🔧 AIVA: Applying Smart Home Preferences for", recognized_speaker)
                for key, value in prefs.items():
                    print(f" - {key}: {value}")
        else:
                print("\n⚠️ AIVA: No stored preferences for this user.")

if __name__ == "__main__":
    DATASET_PATH = r"D:\aiml_project\aiml_aiva\known_audios" 
    vr = VoiceRecognition(DATASET_PATH)

    async def test_identification():
        test_audio = vr.record_audio(duration=3)
        await vr.identify_speaker(test_audio)

    asyncio.run(test_identification())