"""Clasificador local CONDESAN para EcoAcoustic Sentinel.

Aplica la capa de clasificacion de BirdNET V2.4 (opcionalmente afinada con
datos locales) a los embeddings que la app ya calcula, restringida a las
especies del bosque seco validadas por el proyecto CONDESAN y con umbrales
calibrados por especie a partir de las validaciones expertas.

Archivos (``local_models/condesan``):

    condesan_local.json  especies, nombres, umbrales, metadatos y metricas.
    condesan_local.npz   (opcional) pesos afinados W (1024, S) y b (S). Si no
                         existe, los pesos se leen del propio modelo BirdNET
                         instalado (capa ``CLASS_DENSE_LAYER``), por lo que no
                         se redistribuyen pesos de BirdNET.

Uso basico::

    clf = LocalSpeciesClassifier.load()
    probs = clf.probabilities(raw_embeddings)          # (n, S)
    rows = clf.detections(raw_embedding, start, end, file_path)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _app_root() -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS"))
    return Path(__file__).resolve().parent


DEFAULT_MODEL_DIR = _app_root() / "local_models" / "condesan"
META_FILE = "condesan_local.json"
WEIGHTS_FILE = "condesan_local.npz"


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))


def _birdnet_paths() -> tuple[Path, Path]:
    import birdnet_analyzer.config as cfg

    return Path(cfg.BIRDNET_MODEL_PATH), Path(cfg.BIRDNET_LABELS_FILE)


def read_birdnet_dense_layer(model_path: Path | str, labels_path: Path | str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Lee W (1024, C) y b (C,) de la capa CLASS_DENSE_LAYER de un TFLite de BirdNET."""
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore
    except Exception:
        from tensorflow import lite as _tflite

        Interpreter = _tflite.Interpreter

    labels = [line.strip() for line in Path(labels_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    weights = bias = None
    for tensor in interpreter.get_tensor_details():
        if "CLASS_DENSE_LAYER" not in tensor["name"]:
            continue
        shape = tuple(int(x) for x in tensor["shape"])
        if shape == (len(labels), 1024):
            weights = np.array(interpreter.get_tensor(tensor["index"]), dtype=np.float32)
        elif shape == (len(labels),):
            bias = np.array(interpreter.get_tensor(tensor["index"]), dtype=np.float32)
    if weights is None or bias is None:
        raise RuntimeError("No se encontro la capa CLASS_DENSE_LAYER en el modelo BirdNET.")
    return weights.T.copy(), bias, labels


@dataclass
class LocalSpecies:
    scientific: str
    common_es: str
    common_en: str
    birdnet_label: str
    threshold: float


class LocalSpeciesClassifier:
    def __init__(self, meta: dict, W: np.ndarray, b: np.ndarray, *, model_dir: Path | None = None) -> None:
        self.meta = meta
        self.model_dir = model_dir
        self.species = [
            LocalSpecies(
                scientific=str(s["scientific"]),
                common_es=str(s.get("common_es") or s["scientific"]),
                common_en=str(s.get("common_en") or ""),
                birdnet_label=str(s.get("birdnet_label", "")),
                threshold=float(s.get("threshold", 0.5)),
            )
            for s in meta["species"]
        ]
        self.W = np.asarray(W, dtype=np.float32)
        self.b = np.asarray(b, dtype=np.float32)
        if self.W.shape != (1024, len(self.species)) or self.b.shape != (len(self.species),):
            raise ValueError(f"Pesos incompatibles: W{self.W.shape}, b{self.b.shape}, especies={len(self.species)}")
        self.thresholds = np.array([s.threshold for s in self.species], dtype=np.float32)

    # ------------------------------------------------------------------ carga
    @classmethod
    def available(cls, model_dir: Path | str = DEFAULT_MODEL_DIR) -> bool:
        return (Path(model_dir) / META_FILE).exists()

    @classmethod
    def load(
        cls,
        model_dir: Path | str = DEFAULT_MODEL_DIR,
        *,
        birdnet_model_path: Path | str | None = None,
        birdnet_labels_path: Path | str | None = None,
    ) -> "LocalSpeciesClassifier":
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / META_FILE).read_text(encoding="utf-8"))
        weights_file = model_dir / WEIGHTS_FILE
        if meta.get("weights") == WEIGHTS_FILE and weights_file.exists():
            data = np.load(weights_file)
            return cls(meta, data["W"], data["b"], model_dir=model_dir)

        if birdnet_model_path is None or birdnet_labels_path is None:
            default_model, default_labels = _birdnet_paths()
            birdnet_model_path = birdnet_model_path or default_model
            birdnet_labels_path = birdnet_labels_path or default_labels
        W_all, b_all, labels = read_birdnet_dense_layer(birdnet_model_path, birdnet_labels_path)
        index = {lab: i for i, lab in enumerate(labels)}
        missing = [s["birdnet_label"] for s in meta["species"] if s["birdnet_label"] not in index]
        if missing:
            raise ValueError(f"Etiquetas BirdNET no encontradas: {missing[:3]}")
        cols = [index[s["birdnet_label"]] for s in meta["species"]]
        return cls(meta, W_all[:, cols], b_all[cols], model_dir=model_dir)

    # ------------------------------------------------------------ inferencia
    def probabilities(self, raw_embeddings: np.ndarray) -> np.ndarray:
        """Probabilidades (n, S) a partir de embeddings BirdNET crudos (sin normalizar)."""
        x = np.asarray(raw_embeddings, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return _sigmoid(x @ self.W + self.b)

    def detections(self, raw_embedding: np.ndarray, start: float, end: float, file_path: str, *, threshold_scale: float = 1.0) -> list[dict]:
        """Detecciones (formato de la app) para una ventana: especies sobre su umbral."""
        probs = self.probabilities(raw_embedding)[0]
        thr = np.clip(self.thresholds * float(threshold_scale), 0.01, 0.99)
        rows = []
        for k in np.where(probs >= thr)[0]:
            sp = self.species[int(k)]
            rows.append(
                {
                    "start": float(start),
                    "end": float(end),
                    "scientific": sp.scientific,
                    "common": sp.common_es,
                    "confidence": float(probs[k]),
                    "file": str(file_path),
                    "source": "local",
                }
            )
        return rows

    def describe(self) -> str:
        weights = "afinados" if self.meta.get("weights") == WEIGHTS_FILE else "BirdNET original"
        return f"{len(self.species)} especies locales, pesos {weights}, version {self.meta.get('version', '?')}"
