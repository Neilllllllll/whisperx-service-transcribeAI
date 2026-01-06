import os
import torch
# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
MODEL_DIR = os.getenv("ASR_MODEL_PATH", "/models")
MODEL_NAME = os.getenv("ASR_MODEL_NAME", "base")