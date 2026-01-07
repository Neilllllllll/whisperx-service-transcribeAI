import config
import utils
import os
import torch
import functools
from fastapi import FastAPI, UploadFile, File, HTTPException
import whisperx
from whisperx.diarize import DiarizationPipeline
import shutil
import uuid

# --- PATCHES OBLIGATOIRES pour éviter les warnings ---
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

app = FastAPI(title="WhisperX Diarization API")

@app.on_event("startup")
async def load_models():
    app.state.models = {}
    app.state.is_processing = True

    try:
        # Vérifie si les modèles sont présents (si non les télécharges)
        if(os.getenv("HF_TOKEN")):
            utils.ensure_models_downloaded()

        # Passage en mode offline
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        print("⏳ Chargement des modèles pour la diarization !")
        print(f"⏳ Chargement du modèle {config.MODEL_NAME}...")
        app.state.models["asr"] = whisperx.load_model(config.MODEL_NAME, config.DEVICE, compute_type=config.COMPUTE_TYPE, download_root=config.MODEL_DIR)
        print(f"✅ Modèle {config.MODEL_NAME} prêt sur {config.DEVICE}")

        print(f"⏳ Chargement du modèle pyannote pour la diarization ")
        app.state.models["diarize"] = DiarizationPipeline(use_auth_token=config.HF_TOKEN, device=config.DEVICE)
        print(f"✅ Modèle pyannote prêt sur {config.DEVICE}")

        print(f"⏳ Chargement du modèle pour l'alignement ")
        model_a, metadata = whisperx.load_align_model(
            language_code="fr", 
            device=config.DEVICE
        )
        app.state.models["align"] = {"fr": (model_a, metadata)}
        print(f"✅ Modèle d'alignement prêt sur {config.DEVICE}")

        print("✅ Les modèles pour la diarization ont étés chargées avec succès !")
    except Exception as e:
        print(f"Erreur au start : {e}")
        raise e
    finally:
        app.state.is_processing = False

@app.post("/diarize")
async def do_diarization(audioFile: UploadFile = File(...)):
    if app.state.is_processing:
        raise HTTPException(409, "Service occupé")
    
    app.state.is_processing = True

    original_extension = os.path.splitext(audioFile.filename)[1]
    if not original_extension:
        original_extension = ".tmp" # repli par défaut
        
    temp_filename = f"temp_{uuid.uuid4()}{original_extension}"
    
    try:
        # Sauvegarde
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(audioFile.file, buffer)

        # 1. Transcription (ASR)
        audio = whisperx.load_audio(temp_filename)
        result = app.state.models["asr"].transcribe(audio, batch_size=config.BATCH_SIZE)
        language = result["language"]

        # 2. Alignement avec sécurité
        try:
            model_a, metadata = app.state.models["align"][language]
            result = whisperx.align(result["segments"], model_a, metadata, audio, config.DEVICE, return_char_alignments=False)
        except Exception as align_error:
            print(f"Warning: Alignment failed for language {language}: {align_error}")
            # On continue sans alignement précis si ça échoue

        # 3. Diarization
        diarize_segments = app.state.models["diarize"](audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

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
    uvicorn.run(app, host="0.0.0.0", port=5001)