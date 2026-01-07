import config
import os
from huggingface_hub import snapshot_download
# Vérifie et télécharge les modèles si nécessaire avant le démarrage.
def ensure_models_downloaded():
    print(f"📂 Vérification du volume de modèles dans : {config.ASR_MODEL_PATH}")
    
    # 1. On écrase le nom du dossier de cache hugging face
    hf_cache = os.path.join(config.ASR_MODEL_PATH, "hf_cache")
    os.environ["HF_HOME"] = hf_cache
    
    # Liste des modèles Hugging Face requis
    hf_models = [
        "pyannote/speaker-diarization-3.1",
        "pyannote/segmentation-3.0",
        "jonatasgrosman/wav2vec2-large-xlsr-53-french"
    ]

    for model_id in hf_models:
        print(f"🔍 Vérification locale de {model_id}...")
        # snapshot_download ne télécharge que si les fichiers sont manquants ou corrompus
        snapshot_download(
            repo_id=model_id,
            cache_dir=hf_cache,
            token=config.HF_TOKEN if config.HF_TOKEN else None,
            local_files_only=False # Sera mis à True globalement APRÈS ce loop
        )