"""Rutas y constantes del pipeline de fine-tuning.

Todas las rutas se pueden sobrescribir con variables de entorno para ejecutar
el pipeline en otra máquina (por ejemplo, con la carpeta de Google Drive
sincronizada localmente mediante Google Drive para escritorio):

    ECO_DATA_ROOT         Carpeta raíz "AudioMoth - CONDESAN" (o su copia local).
    ECO_WORK_DIR          Carpeta de trabajo para embeddings y tablas intermedias.
    ECO_BIRDNET_MODEL_DIR Carpeta con BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite
                          y BirdNET_GLOBAL_6K_V2.4_Labels.txt.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Copia local (espejo) de la carpeta de Drive "AudioMoth - CONDESAN".
# data_cache/ esta excluido de git: contiene datos privados del proyecto.
DATA_ROOT = Path(os.environ.get("ECO_DATA_ROOT", REPO_ROOT / "data_cache" / "drive_copy"))
TABLA_FINAL = DATA_ROOT / "Resultados" / "TablaFinal.xlsx"
LISTA_ESPECIES = DATA_ROOT / "Resultados" / "Lista_Especies.xlsx"

# Artefactos intermedios (embeddings, manifiestos). Tambien excluidos de git.
WORK_DIR = Path(os.environ.get("ECO_WORK_DIR", REPO_ROOT / "data_cache" / "work"))

# Resultados agregados (metricas, figuras, tablas LaTeX) que si se versionan.
RESULTS_DIR = REPO_ROOT / "finetuning" / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# Clasificador local exportado para la aplicacion. No usar "models/" en la raiz:
# build_exe.py interpreta esa carpeta como los checkpoints de BirdNET.
MODELS_DIR = REPO_ROOT / "local_models" / "condesan"

# Pesos de BirdNET V2.4 (licencia CC BY-NC-SA 4.0, no se versionan).
BIRDNET_MODEL_DIR = Path(os.environ.get("ECO_BIRDNET_MODEL_DIR", Path.home() / "birdnet_models" / "V2.4"))
BIRDNET_MODEL_FILE = "BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"
BIRDNET_LABELS_FILE = "BirdNET_GLOBAL_6K_V2.4_Labels.txt"

# Parametros de senal de BirdNET.
SAMPLE_RATE = 48_000
SIG_LENGTH = 3.0
RECORDING_LENGTH = 55.0  # AudioMoth: 55 s de grabacion + 5 s de pausa por minuto.

# Especies objetivo: minimo de detecciones validadas en TablaFinal para entrenar.
MIN_POSITIVES_TARGET = 15

RANDOM_SEED = 42


def ensure_dirs() -> None:
    for path in (WORK_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)
