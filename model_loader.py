# model_loader.py
import torch
import os
import warnings
import logging
from speechbrain.inference import SpeakerRecognition

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("speechbrain").setLevel(logging.WARNING)
os.environ["SB_DISABLE_QUIRKS"] = "disable_jit_profiling,allow_tf32"
torch.autograd.set_detect_anomaly(False)

class ModelLoader:
    def __init__(self):
        print("📥 Loading Speaker Recognition Model...")
        self.model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb", 
            savedir="pretrained_models/spkrec-xvect-voxceleb"
        )
        print("✅ Model Loaded Successfully!")

model_loader = ModelLoader()
model = model_loader.model
