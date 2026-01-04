import os
import torch
import functools
from fastapi import FastAPI, UploadFile, File, HTTPException
import whisperx
from whisperx.diarize import DiarizationPipeline
import shutil
import uuid

# --- PATCHES OBLIGATOIRES pour éviter les warnings ---
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

app = FastAPI(title="WhisperX Diarization API")

# --- CONFIGURATION ---
HF_TOKEN = os.getenv("HF_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16"
BATCH_SIZE = 16
MODEL_DIR = os.getenv("ASR_MODEL_PATH", "/models")
MODEL_NAME = os.getenv("ASR_MODEL_NAME", "base")

# --- CHARGEMENT DES MODÈLES AU DÉMARRAGE ---
# On utilise un dictionnaire global pour stocker les modèles
models = {}

@app.on_event("startup")
async def load_models():
    print("Chargement des modèles WhisperX...")
    models["asr"] = whisperx.load_model(MODEL_NAME, DEVICE, compute_type=COMPUTE_TYPE, download_root=MODEL_DIR)
    models["diarize"] = DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
    # On chargera le modèle d'alignement dynamiquement selon la langue détectée
    models["align"] = {} 
    print("Modèles prêts !")
    app.state.is_processing = False

@app.post("/diarize")
async def do_diarization(file: UploadFile = File(...)):
    if app.state.is_processing:
        raise HTTPException(409, "Service occupé")
    
    app.state.is_processing = True
    temp_filename = f"temp_{uuid.uuid4()}.mp3"
    
    try:
        # Sauvegarde
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Transcription (ASR)
        audio = whisperx.load_audio(temp_filename)
        result = models["asr"].transcribe(audio, batch_size=BATCH_SIZE)
        language = result["language"]

        # 2. Alignement avec sécurité
        try:
            if language not in models["align"]:
                # On tente de charger, sinon on replie sur l'alignement par défaut ou on skip
                model_a, metadata = whisperx.load_align_model(language_code=language, device=DEVICE)
                models["align"][language] = (model_a, metadata)
            
            model_a, metadata = models["align"][language]
            result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)
        except Exception as align_error:
            print(f"Warning: Alignment failed for language {language}: {align_error}")
            # On continue sans alignement précis si ça échoue

        # 3. Diarization
        # On s'assure que l'audio est libéré de la mémoire si possible après ça
        diarize_segments = models["diarize"](audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # 4. Nettoyage VRAM agressif (Optionnel mais recommandé pour RTX 50)
        # torch.cuda.empty_cache() 

        output = [{
            "start": round(seg["start"], 2) if "start" in seg else 0,
            "end": round(seg["end"], 2) if "end" in seg else 0,
            "speaker": seg.get("speaker", "UNKNOWN"),
            "text": seg["text"].strip()
        } for seg in result["segments"]]

        return {"language": language, "segments": output}

    except Exception as e:
        print(f"Global Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        app.state.is_processing = False

@app.get("/busy")
async def is_busy():
    return {"is_processing": app.state.is_processing}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)