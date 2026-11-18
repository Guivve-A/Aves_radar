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
    verificador_local.npz (opcional) sonda lineal entrenada con las detecciones
                         validadas y rechazadas del proyecto. No cambia que se
                         detecta: BirdNET propone y la sonda marca como dudosas
                         ("revisar") las detecciones que no se parecen a las
                         validadas de su especie. Lo genera
                         ``python -m finetuning.deep.probes --export``.

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
VERIFIER_FILE = "verificador_local.npz"
SOURCE_LOCAL = "local"
SOURCE_REVIEW = "local_revisar"       # deteccion local que el verificador considera dudosa


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
        self.verifier: dict | None = None

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
            clf = cls(meta, data["W"], data["b"], model_dir=model_dir)
            clf.load_verifier(model_dir / VERIFIER_FILE)
            return clf

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
        clf = cls(meta, W_all[:, cols], b_all[cols], model_dir=model_dir)
        clf.load_verifier(model_dir / VERIFIER_FILE)
        return clf

    def load_verifier(self, path: Path | str) -> bool:
        """Carga el verificador opcional. Devuelve False (y no cambia nada) si no existe o no es utilizable."""
        path = Path(path)
        if not path.exists():
            return False
        try:
            with np.load(path, allow_pickle=False) as data:
                meta = json.loads(str(data["meta"]))
                W, b, cutoff = (np.asarray(data[k], dtype=np.float32) for k in ("W", "b", "cutoff"))
            names = [str(n) for n in meta["species"]]
            if W.shape != (1024, len(names)) or b.shape != (len(names),) or cutoff.shape != (len(names),):
                return False
        except Exception:
            return False
        index = {name: i for i, name in enumerate(names)}
        cols = np.array([index.get(sp.scientific, -1) for sp in self.species])
        known = cols >= 0
        if not known.any():
            return False
        safe = np.where(known, cols, 0)
        self.verifier = {
            "W": W[:, safe],
            "b": b[safe],
            # una especie que el verificador no conoce nunca se marca: su corte queda en 0
            "cutoff": np.where(known, cutoff[safe], 0.0).astype(np.float32),
            "l2_input": bool(meta.get("l2_input", False)),
            "meta": meta,
        }
        return True

    # ------------------------------------------------------------ inferencia
    def probabilities(self, raw_embeddings: np.ndarray) -> np.ndarray:
        """Probabilidades (n, S) a partir de embeddings BirdNET crudos (sin normalizar)."""
        x = np.asarray(raw_embeddings, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return _sigmoid(x @ self.W + self.b)

    def verification(self, raw_embeddings: np.ndarray) -> np.ndarray | None:
        """Puntuacion de la sonda local (n, S), o None si no hay verificador instalado."""
        if self.verifier is None:
            return None
        x = np.asarray(raw_embeddings, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if self.verifier["l2_input"]:
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        return _sigmoid(x @ self.verifier["W"] + self.verifier["b"])

    def detections(self, raw_embedding: np.ndarray, start: float, end: float, file_path: str, *, threshold_scale: float = 1.0) -> list[dict]:
        """Detecciones (formato de la app) para una ventana: especies sobre su umbral.

        Con verificador, cada deteccion lleva ademas ``verification`` (puntuacion de la sonda local) y, si queda
        por debajo del corte de su especie, la fuente pasa a ser ``local_revisar``. El conjunto de detecciones
        no cambia: solo se marca cuales conviene escuchar antes de aceptarlas.
        """
        probs = self.probabilities(raw_embedding)[0]
        thr = np.clip(self.thresholds * float(threshold_scale), 0.01, 0.99)
        hits = np.where(probs >= thr)[0]
        verif = self.verification(raw_embedding)[0] if self.verifier is not None and len(hits) else None
        rows = []
        for k in hits:
            sp = self.species[int(k)]
            row = {
                "start": float(start),
                "end": float(end),
                "scientific": sp.scientific,
                "common": sp.common_es,
                "confidence": float(probs[k]),
                "file": str(file_path),
                "source": SOURCE_LOCAL,
            }
            if verif is not None:
                row["verification"] = float(verif[k])
                if verif[k] < self.verifier["cutoff"][k]:
                    row["source"] = SOURCE_REVIEW
            rows.append(row)
        return rows

    def describe(self) -> str:
        weights = "afinados" if self.meta.get("weights") == WEIGHTS_FILE else "BirdNET original"
        verifier = ", con verificador local" if self.verifier is not None else ""
        return f"{len(self.species)} especies locales, pesos {weights}{verifier}, version {self.meta.get('version', '?')}"
