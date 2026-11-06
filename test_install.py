import sys
import torch
import torchaudio
import birdnet_analyzer
from PySide6.QtWidgets import QApplication

print("--- REPORTE DE SISTEMA ---")
print(f"Python: {sys.version}")
print(f"PyTorch (CPU): {torch.__version__}")
print(f"Audio Backend: {torchaudio.get_audio_backend()}")
print(f"BirdNET Analyzer: Importado correctamente")
print(f"PySide6 (UI): {QApplication}")
print("--- TODO LISTO PARA DESPEGAR ---")