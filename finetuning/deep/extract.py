"""Audio -> ventanas de 3 s -> logits, embeddings y mapas de activacion de BirdNET V2.4.

Replica el camino de birdnet_analyzer 2.4.0 (el mismo de la app y de
``finetuning.embeddings``): lectura a 48 kHz mono, ventanas de 3 s sin solape,
la ultima se conserva si tiene al menos 1 s y se rellena con ceros. Se comprobo
que los embeddings coinciden con los calculados en Colab (diferencia maxima
3e-6) y los logits dentro del redondeo de float16.

Uso:
    python -m finetuning.deep.extract                 # procesa los WAV disponibles del manifiesto
    python -m finetuning.deep.extract --watch 1800    # sigue esperando archivos nuevos hasta 30 min
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from .tflite_tools import MultiOutputBirdNET

FEAT_DIR = config.WORK_DIR / "features"
MANIFEST = config.WORK_DIR / "wav_manifest.csv"
CHUNK = int(config.SAMPLE_RATE * config.SIG_LENGTH)


def read_audio(path: Path | str) -> np.ndarray:
    import soundfile as sf

    sig, rate = sf.read(str(path), dtype="float32", always_2d=True)
    sig = sig.mean(axis=1) if sig.shape[1] > 1 else sig[:, 0]
    if rate != config.SAMPLE_RATE:
        from scipy.signal import resample_poly

        sig = resample_poly(sig, config.SAMPLE_RATE, rate).astype(np.float32)
    return np.ascontiguousarray(sig, dtype=np.float32)


def split_signal(sig: np.ndarray, seconds: float = config.SIG_LENGTH, overlap: float = 0.0, minlen: float = 1.0) -> np.ndarray:
    """Misma particion que ``birdnet_analyzer.audio.split_signal`` (relleno con ceros)."""
    rate = config.SAMPLE_RATE
    chunksize = int(rate * seconds)
    stepsize = int(rate * (seconds - overlap))
    minsize = int(rate * minlen)
    last = int((sig.size - chunksize + stepsize - 1) / stepsize) * stepsize
    if last < 0:
        last = 0
    elif sig.size - last < minsize:
        last -= stepsize
    if last < 0 or sig.size == 0:
        return np.zeros((0, chunksize), dtype=np.float32)
    data = np.concatenate((sig, np.zeros(chunksize, dtype=sig.dtype)))
    return np.stack([data[i : i + chunksize] for i in range(0, last + 1, stepsize)])


def cache_path(file_key: str) -> Path:
    return FEAT_DIR / (file_key.replace("/", "__") + ".npz")


def extract_file(net: MultiOutputBirdNET, wav: Path | str, batch: int = 32) -> dict[str, np.ndarray] | None:
    chunks = split_signal(read_audio(wav))
    if len(chunks) == 0:
        return None
    outs = [net(chunks[i : i + batch]) for i in range(0, len(chunks), batch)]
    return {
        "starts": (np.arange(len(chunks)) * config.SIG_LENGTH).astype(np.float32),
        "emb": np.concatenate([o["emb"] for o in outs]).astype(np.float32),
        "logits": np.concatenate([o["logits"] for o in outs]).astype(np.float16),
        "f3": np.concatenate([o["f3"] for o in outs]).astype(np.float16),
        "f2": np.concatenate([o["f2"] for o in outs]).astype(np.float16),
    }


def run(manifest: Path = MANIFEST, threads: int = 4, watch: float = 0.0) -> None:
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    net = MultiOutputBirdNET(threads=threads)
    deadline = time.time() + watch
    done = failed = 0
    t0 = time.time()
    while True:
        man = pd.read_csv(manifest, dtype=str).fillna("")
        pending = [(r.file_key, config.DATA_ROOT / r.rel_path) for r in man.itertuples() if not cache_path(r.file_key).exists()]
        ready = [(k, p) for k, p in pending if p.exists()]
        for key, wav in ready:
            try:
                out = extract_file(net, wav)
            except Exception as ex:  # archivo corrupto o a medio copiar
                print(f"  aviso: {wav.name}: {ex}", flush=True)
                failed += 1
                continue
            if out is None:
                failed += 1
                continue
            tmp = cache_path(key).with_suffix(".tmp.npz")
            np.savez(tmp, **out)
            tmp.replace(cache_path(key))
            done += 1
            if done % 50 == 0:
                print(f"  {done} grabaciones ({time.time() - t0:.0f} s), pendientes {len(pending) - len(ready)} sin audio", flush=True)
        if not pending or (time.time() > deadline and not ready):
            break
        if not ready:
            time.sleep(15)
    total = sum(1 for _ in FEAT_DIR.glob("*.npz"))
    print(f"Extraccion: {done} nuevas, {failed} con error, {total} en cache ({time.time() - t0:.0f} s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--watch", type=float, default=0.0, help="Segundos durante los que se esperan audios nuevos.")
    args = parser.parse_args()
    run(threads=args.threads, watch=args.watch)


if __name__ == "__main__":
    main()
