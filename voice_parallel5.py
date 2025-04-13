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
from model_loader import model  # Import the preloaded model

logging.basicConfig(level=logging.INFO)

class VoiceRecognition:
    def __init__(self, dataset_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dataset_path = dataset_path
        self.known_speakers, self.speaker_thresholds = self.load_known_speakers()
        #print(f"Using device: {self.device}")
        for speaker, thresh in self.speaker_thresholds.items():
            print(f"Speaker: {speaker}, Max Distance Threshold: {thresh:.4f}")

    def record_audio(self, filename="live_input.wav", duration=5, samplerate=16000):
        print("🎙 Recording... Speak now!")
        audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
        sd.wait()
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(audio_data.tobytes())
        #print("🎤 Recording saved as", filename)
        return filename

    def extract_embedding(self, audio_file):
        try:
            signal, fs = torchaudio.load(audio_file, normalize=True)
            signal = signal.to(self.device)
            if fs != 16000:
                transform = torchaudio.transforms.Resample(fs, 16000).to(self.device)
                signal = transform(signal)
            with torch.no_grad():
                embedding = self.model.encode_batch(signal).squeeze().cpu().numpy()
            embedding = embedding.flatten()
            embedding = np.pad(embedding, (0, 512 - embedding.shape[0]), 'constant') if embedding.shape[0] < 512 else embedding[:512]
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
                        if i < j:  # Avoid self-comparison and duplicates
                            dist = cosine(emb1, emb2)
                            distances.append(dist)
                if distances:
                    max_distance = np.max(distances) * 1.5  # 20% buffer beyond max observed distance
                    speaker_thresholds[speaker] = min(max_distance, 0.5)  # Cap at 0.35
                else:
                    speaker_thresholds[speaker] = 0.5  # Default threshold

        if not speaker_embeddings:
            logging.error("No valid speakers loaded. Check dataset.")
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
            recognition_result = best_speaker
            logging.info(f" Recognized as {recognition_result}")
        else:
            recognition_result = "Unknown"
            logging.info(f"Decision: Unknown (distance {best_distance:.4f} >= threshold {threshold:.4f})")

        end_time = time.time()
        recognition_time = (end_time - start_time) * 1000
        logging.info(f"Time taken to recognize speaker: {recognition_time:.2f} ms")

        await self.speak(f"Hello {recognition_result}, How can I help you?" if recognition_result != "Unknown" else "Sorry, I don’t recognize this voice.")
        #print("\n🎤 Recognized Speaker:", recognition_result)
        return recognition_result

if __name__ == "__main__":
    DATASET_PATH = r"D:\aiml_project\aiml_aiva\known_audios"  # Replace with your actual path
    vr = VoiceRecognition(DATASET_PATH)

    async def test_identification():
        test_audio = vr.record_audio(duration=5)
        await vr.identify_speaker(test_audio)

    asyncio.run(test_identification())

    