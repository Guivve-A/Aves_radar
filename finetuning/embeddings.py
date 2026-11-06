"""Extraccion de embeddings y scores originales de BirdNET V2.4.

Reproduce el mismo camino de calculo que usa la aplicacion (birdnet_analyzer):

    audio.open_audio_file   librosa, 48 kHz, mono, remuestreo kaiser_fast
    (pasa-banda 0-15 kHz: identidad con la configuracion por defecto)
    audio.split_signal      ventanas de 3 s, solape 0 s, longitud minima 1 s
    model.embeddings        penultima capa de BirdNET (vector de 1024 dimensiones)
    model.predict           logits de las 6 522 clases de BirdNET V2.4

Asi, la cabeza afinada que se entrena sobre estos embeddings es directamente
aplicable a los embeddings que la app ya calcula con
``iterate_audio_chunks(..., embeddings=True)``.

Uso:
    python -m finetuning.embeddings            # procesa todo el indice del espejo
    python -m finetuning.embeddings --limit 3  # prueba rapida
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

EMB_DIR = config.WORK_DIR / "embeddings"
ALL_NPZ = config.WORK_DIR / "embeddings_all.npz"


def configure_birdnet(model_dir: Path | str | None = None, *, threads: int = 4, batch_size: int = 32) -> list[str]:
    """Configura birdnet_analyzer para usar el modelo TFLite local y devuelve las etiquetas."""
    import birdnet_analyzer.config as cfg

    model_dir = Path(model_dir or config.BIRDNET_MODEL_DIR)
    model_path = model_dir / config.BIRDNET_MODEL_FILE
    labels_path = model_dir / config.BIRDNET_LABELS_FILE
    if not model_path.exists():
        # Instalacion estandar de la app: checkpoints descargados por birdnet_analyzer.
        model_path = Path(cfg.BIRDNET_MODEL_PATH)
        labels_path = Path(cfg.BIRDNET_LABELS_FILE)
    if not model_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "No se encontro el modelo BirdNET V2.4. Define ECO_BIRDNET_MODEL_DIR o ejecuta la app una vez "
            "para que birdnet_analyzer descargue sus checkpoints."
        )

    cfg.MODEL_PATH = str(model_path)
    cfg.LABELS_FILE = str(labels_path)
    cfg.SAMPLE_RATE = config.SAMPLE_RATE
    cfg.SIG_LENGTH = config.SIG_LENGTH
    cfg.SIG_OVERLAP = 0.0
    cfg.SIG_MINLEN = 1.0
    cfg.AUDIO_SPEED = 1.0
    cfg.BANDPASS_FMIN = 0
    cfg.BANDPASS_FMAX = 15000
    cfg.TFLITE_THREADS = max(1, int(threads))
    cfg.CPU_THREADS = 1
    cfg.BATCH_SIZE = max(1, int(batch_size))
    cfg.CUSTOM_CLASSIFIER = None
    cfg.USE_PERCH = False
    cfg.APPLY_SIGMOID = True
    cfg.SIGMOID_SENSITIVITY = 1.0
    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cfg.LABELS = labels
    return labels


def extract_file(path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (inicios_s, embeddings[n,1024], logits[n,6522]) para un archivo."""
    import birdnet_analyzer.config as cfg
    from birdnet_analyzer import audio, model

    sig, rate = audio.open_audio_file(
        str(path), cfg.SAMPLE_RATE, 0.0, None, cfg.BANDPASS_FMIN, cfg.BANDPASS_FMAX, cfg.AUDIO_SPEED
    )
    chunks = audio.split_signal(sig, rate, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN)
    if not chunks:
        return np.zeros(0), np.zeros((0, 1024), np.float32), np.zeros((0, len(cfg.LABELS)), np.float32)
    starts = np.round(np.arange(len(chunks)) * (cfg.SIG_LENGTH - cfg.SIG_OVERLAP), 2)
    embs, logits = [], []
    for i in range(0, len(chunks), cfg.BATCH_SIZE):
        batch = chunks[i : i + cfg.BATCH_SIZE]
        logit = np.asarray(model.predict(batch), dtype=np.float32)
        emb = _embeddings_after_predict(batch)
        embs.append(emb)
        logits.append(logit)
    return starts, np.vstack(embs), np.vstack(logits)


_EMB_CHECKED = False


def _embeddings_after_predict(batch) -> np.ndarray:
    """Lee el tensor de embeddings (penultima capa) tras la inferencia de clasificacion.

    birdnet_analyzer obtiene los embeddings leyendo el tensor ``salida - 1`` del
    interprete TFLite. Tras ``model.predict`` ese tensor ya contiene los mismos
    valores, lo que evita una segunda inferencia. La primera vez se verifica
    contra ``model.embeddings`` y, si no coincidiera, se usa siempre la llamada
    explicita.
    """
    global _EMB_CHECKED
    from birdnet_analyzer import model

    emb = None
    try:
        emb = np.array(model.INTERPRETER.get_tensor(model.OUTPUT_DETAILS[0]["index"] - 1), dtype=np.float32)
    except Exception:
        emb = None
    if emb is None or emb.shape[0] != len(batch) or not _EMB_CHECKED:
        reference = np.asarray(model.embeddings(batch), dtype=np.float32)
        if emb is not None and emb.shape == reference.shape and np.allclose(emb, reference, atol=1e-5):
            _EMB_CHECKED = True
        return reference
    return emb


def birdnet_classifier_weights(model_dir: Path | str | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pesos exactos de la capa de clasificacion de BirdNET V2.4.

    La red termina en ``GLOBAL_AVG_POOL`` (embedding de 1024 dimensiones) seguida
    de ``CLASS_DENSE_LAYER`` (lineal, 6522 salidas): logit = W @ e + b. Se leen
    los tensores constantes del archivo TFLite. Devuelve W (1024, 6522), b (6522,)
    y las etiquetas.
    """
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore
    except Exception:  # pragma: no cover - mismo interprete que usa birdnet_analyzer
        from tensorflow import lite as _tflite

        Interpreter = _tflite.Interpreter

    labels = configure_birdnet(model_dir)
    import birdnet_analyzer.config as cfg

    it = Interpreter(model_path=cfg.MODEL_PATH)
    it.allocate_tensors()
    weights = bias = None
    for t in it.get_tensor_details():
        name = t["name"]
        if "CLASS_DENSE_LAYER" not in name:
            continue
        shape = tuple(int(x) for x in t["shape"])
        if shape == (len(labels), 1024):
            weights = np.array(it.get_tensor(t["index"]), dtype=np.float32)
        elif shape == (len(labels),):
            bias = np.array(it.get_tensor(t["index"]), dtype=np.float32)
    if weights is None or bias is None:
        raise RuntimeError("No se encontro CLASS_DENSE_LAYER en el modelo TFLite.")
    return weights.T.copy(), bias, labels


def flat_sigmoid(logits: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
    """Misma activacion que BirdNET (birdnet_analyzer.model.flat_sigmoid)."""
    bias = (sensitivity - 1.0) * 10.0
    return 1.0 / (1.0 + np.exp(-np.clip(logits + bias, -20, 20)))


def _cache_path(file_key: str) -> Path:
    return EMB_DIR / (file_key.replace("/", "__") + ".npz")


def run(index: pd.DataFrame, *, threads: int = 4, limit: int | None = None) -> pd.DataFrame:
    labels = configure_birdnet(threads=threads)
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    todo = index if limit is None else index.head(limit)
    t0 = time.time()
    errors = []
    created = 0
    for i, rec in enumerate(todo.itertuples(index=False), start=1):
        cache = _cache_path(rec.file_key)
        if not cache.exists():
            try:
                starts, emb, logit = extract_file(config.DATA_ROOT / rec.rel_path)
            except Exception as ex:  # archivo corrupto, vacio o inaccesible: se registra y se sigue
                print(f"  aviso: se omite {rec.rel_path}: {ex}", flush=True)
                errors.append({"file_key": rec.file_key, "rel_path": rec.rel_path, "error": str(ex)})
                continue
            if len(starts) == 0:
                errors.append({"file_key": rec.file_key, "rel_path": rec.rel_path, "error": "sin ventanas"})
                continue
            tmp = cache.with_name(cache.stem + ".tmp.npz")
            np.savez_compressed(tmp, starts=starts, emb=emb, logits=logit.astype(np.float16))
            tmp.replace(cache)  # escritura atomica: una desconexion no deja caches corruptos
            created += 1
        rows.append({"file_key": rec.file_key, "cache": str(cache)})
        if i % 10 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} archivos ({time.time() - t0:.1f} s)", flush=True)
    if errors:
        pd.DataFrame(errors).to_csv(config.WORK_DIR / "embeddings_errors.csv", index=False)
        print(f"  {len(errors)} archivos omitidos (ver embeddings_errors.csv)", flush=True)

    if limit is not None:
        # Prueba rapida: no se toca el consolidado (puede contener el conjunto completo).
        print(f"  prueba: {len(rows)} grabaciones listas; {ALL_NPZ.name} no se modifica", flush=True)
        return pd.DataFrame(rows)
    if created == 0 and ALL_NPZ.exists():
        with np.load(ALL_NPZ) as prev:
            if set(prev["file_key"].astype(str)) == {r["file_key"] for r in rows}:
                print(f"  sin grabaciones nuevas: {ALL_NPZ.name} ya esta al dia", flush=True)
                return pd.DataFrame(rows)

    # Consolidado
    keys, starts_all, embs, logits = [], [], [], []
    for row in rows:
        data = np.load(row["cache"])
        n = len(data["starts"])
        keys.extend([row["file_key"]] * n)
        starts_all.append(data["starts"])
        embs.append(data["emb"])
        logits.append(data["logits"])
    np.savez_compressed(
        ALL_NPZ,
        file_key=np.array(keys),
        start=np.concatenate(starts_all) if starts_all else np.zeros(0),
        emb=np.vstack(embs) if embs else np.zeros((0, 1024), np.float32),
        logits=np.vstack(logits) if logits else np.zeros((0, len(labels)), np.float16),
        labels=np.array(labels),
    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--todas", action="store_true", help="Incluye grabaciones fuera de las sesiones de TablaFinal.")
    args = parser.parse_args()
    from .drive_mirror import INDEX_CSV, build_index, labeled_scope

    index = pd.read_csv(INDEX_CSV) if INDEX_CSV.exists() else build_index()
    if not args.todas:
        index = labeled_scope(index)
    print(f"Extrayendo embeddings de {len(index) if args.limit is None else args.limit} grabaciones...")
    run(index, threads=args.threads, limit=args.limit)
    if args.limit is None:
        with np.load(ALL_NPZ) as data:
            print(f"Consolidado: {data['start'].shape[0]} ventanas -> {ALL_NPZ}")


if __name__ == "__main__":
    main()
