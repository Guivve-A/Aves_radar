import csv
import json
import logging
import math
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

import numpy as np
import soundfile as sf
import torch
from PySide6.QtCore import QObject, Signal, Slot
from sklearn.cluster import KMeans, MiniBatchKMeans

import birdnet_analyzer.config as cfg
from birdnet_analyzer import analyze as birdnet_run
from birdnet_analyzer.analyze.utils import iterate_audio_chunks
from birdnet_analyzer.utils import ensure_model_exists

try:
    import noisereduce as nr
except Exception:  # pragma: no cover - optional dependency fallback
    nr = None

try:
    from local_classifier import LocalSpeciesClassifier
except Exception:  # pragma: no cover - modulo opcional
    LocalSpeciesClassifier = None


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".wma", ".aiff", ".aif"}
ANOMALY_COMMON_LABEL = "Anomalia Acustica / Posible Nueva Especie"
log = logging.getLogger(__name__)


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    rows = np.asarray(matrix, dtype=np.float32)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    out = np.zeros_like(rows, dtype=np.float32)
    for i in range(rows.shape[0]):
        out[i] = _l2_normalize(rows[i])
    return out


@dataclass
class ReferenceProfile:
    name: str
    safe_name: str
    source_audio: str
    vector_file: str
    created_at: str
    embedding_dim: list[int]
    num_vectors: int
    vector_dim: int


class ProfileManager:
    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.storage_dir / "profiles.json"

    def _load_metadata(self) -> list[dict]:
        if not self.meta_path.exists():
            return []
        try:
            data = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return []

    def _save_metadata(self, items: list[dict]) -> None:
        self.meta_path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")

    def _safe_name(self, name: str) -> str:
        text = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip())
        text = re.sub(r"_+", "_", text).strip("_")
        return text.lower() or "species"

    def _legacy_shape_to_list(self, embedding_dim: object) -> list[int]:
        if isinstance(embedding_dim, list) and len(embedding_dim) == 2:
            return [int(embedding_dim[0]), int(embedding_dim[1])]
        if isinstance(embedding_dim, tuple) and len(embedding_dim) == 2:
            return [int(embedding_dim[0]), int(embedding_dim[1])]
        if isinstance(embedding_dim, int):
            return [1, int(embedding_dim)]
        return [1, 0]

    def _to_profile(self, raw: dict) -> ReferenceProfile:
        shape = self._legacy_shape_to_list(raw.get("embedding_dim"))
        num_vectors = int(raw.get("num_vectors", shape[0]))
        vector_dim = int(raw.get("vector_dim", shape[1]))
        return ReferenceProfile(
            name=str(raw.get("name", "")).strip(),
            safe_name=str(raw.get("safe_name", "")).strip(),
            source_audio=str(raw.get("source_audio", "")),
            vector_file=str(raw.get("vector_file", "")),
            created_at=str(raw.get("created_at", "")),
            embedding_dim=[num_vectors, vector_dim],
            num_vectors=num_vectors,
            vector_dim=vector_dim,
        )

    def list_profiles(self) -> list[ReferenceProfile]:
        return [self._to_profile(item) for item in self._load_metadata()]

    def _find_profile_index(self, profiles: list[dict], profile_name: str) -> int:
        target = profile_name.strip().lower()
        for idx, item in enumerate(profiles):
            if str(item.get("name", "")).strip().lower() == target:
                return idx
        return -1

    def save_profile(self, name: str, source_audio: str, vectors: np.ndarray) -> ReferenceProfile:
        profiles = self._load_metadata()
        safe_name = self._safe_name(name)
        taken = {
            str(item.get("safe_name", ""))
            for item in profiles
            if str(item.get("name", "")).strip().lower() != name.strip().lower()
        }
        base, suffix = safe_name, 2
        while safe_name in taken:
            safe_name = f"{base}_{suffix}"
            suffix += 1
        vector_file = f"{safe_name}.npy"
        vector_path = self.storage_dir / vector_file

        matrix = _normalize_rows(np.asarray(vectors, dtype=np.float32))
        np.save(vector_path, matrix.astype(np.float32))
        num_vectors, vector_dim = int(matrix.shape[0]), int(matrix.shape[1])

        entry = ReferenceProfile(
            name=name.strip(),
            safe_name=safe_name,
            source_audio=str(source_audio),
            vector_file=vector_file,
            created_at=datetime.utcnow().isoformat(),
            embedding_dim=[num_vectors, vector_dim],
            num_vectors=num_vectors,
            vector_dim=vector_dim,
        )

        updated = False
        for idx, item in enumerate(profiles):
            if str(item.get("name", "")).strip().lower() == name.strip().lower():
                profiles[idx] = asdict(entry)
                updated = True
                break
        if not updated:
            profiles.append(asdict(entry))
        self._save_metadata(profiles)
        return entry

    def update_profile_vectors(
        self,
        profile_name: str,
        vectors: np.ndarray,
        *,
        source_audio: str | None = None,
    ) -> ReferenceProfile:
        profiles = self._load_metadata()
        idx = self._find_profile_index(profiles, profile_name)
        if idx < 0:
            raise ValueError(f"Perfil no encontrado: {profile_name}")

        raw = profiles[idx]
        vector_file = str(raw.get("vector_file", ""))
        if not vector_file:
            raise ValueError(f"Perfil invalido (sin vector_file): {profile_name}")
        vector_path = self.storage_dir / vector_file

        matrix = _normalize_rows(np.asarray(vectors, dtype=np.float32))
        np.save(vector_path, matrix.astype(np.float32))
        num_vectors, vector_dim = int(matrix.shape[0]), int(matrix.shape[1])

        raw["embedding_dim"] = [num_vectors, vector_dim]
        raw["num_vectors"] = num_vectors
        raw["vector_dim"] = vector_dim
        raw["created_at"] = datetime.utcnow().isoformat()
        if source_audio is not None:
            raw["source_audio"] = source_audio
        profiles[idx] = raw
        self._save_metadata(profiles)
        return self._to_profile(raw)

    def remove_profile(self, profile_name: str) -> bool:
        profiles = self._load_metadata()
        kept: list[dict] = []
        removed = False
        target = profile_name.strip().lower()

        for item in profiles:
            if str(item.get("name", "")).strip().lower() == target:
                vector_path = self.storage_dir / str(item.get("vector_file", ""))
                if vector_path.exists():
                    vector_path.unlink()
                removed = True
                continue
            kept.append(item)

        if removed:
            self._save_metadata(kept)
        return removed

    def load_profile_vectors(self) -> dict[str, np.ndarray]:
        vectors: dict[str, np.ndarray] = {}
        for profile in self.list_profiles():
            vector_path = self.storage_dir / profile.vector_file
            if not vector_path.exists():
                continue
            vec = np.load(vector_path)
            matrix = _normalize_rows(np.asarray(vec, dtype=np.float32))
            vectors[profile.name] = matrix
        return vectors

    def load_single_profile_vectors(self, profile_name: str) -> np.ndarray:
        target = profile_name.strip().lower()
        for profile in self.list_profiles():
            if profile.name.strip().lower() != target:
                continue
            vector_path = self.storage_dir / profile.vector_file
            if not vector_path.exists():
                break
            arr = np.load(vector_path)
            return _normalize_rows(np.asarray(arr, dtype=np.float32))
        raise ValueError(f"Perfil no encontrado: {profile_name}")


class AudioAnalyzer:
    def __init__(
        self,
        profile_manager: ProfileManager,
        similarity_threshold: float = 0.75,
        anomaly_threshold: float = 0.45,
        overlap_seconds: float = 0.3,
        chunk_seconds: float = 3.0,
        batch_size: int = 1,
        threads: int = 4,
        use_noise_reduction: bool = False,
        sensitivity: float = 1.0,
        global_min_conf: float = 0.25,
    ) -> None:
        self.profile_manager = profile_manager
        self.similarity_threshold = float(similarity_threshold)
        self.anomaly_threshold = float(anomaly_threshold)
        self.overlap_seconds = float(overlap_seconds)
        self.chunk_seconds = float(chunk_seconds)
        self.batch_size = int(batch_size)
        self.threads = int(threads)
        self.use_noise_reduction = bool(use_noise_reduction)
        # Solo afectan al Modo Descubrimiento (BirdNET global): el clasificador local usa
        # sus propios umbrales calibrados y los perfiles, la similitud coseno.
        self.sensitivity = float(sensitivity)
        self.global_min_conf = float(global_min_conf)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cosine = torch.nn.CosineSimilarity(dim=1)
        self.local_classifier = None
        self.local_classifier_error = ""
        ensure_model_exists()

    def local_classifier_available(self) -> bool:
        return LocalSpeciesClassifier is not None and LocalSpeciesClassifier.available()

    def get_local_classifier(self):
        """Carga perezosa del clasificador local CONDESAN (None si no esta disponible)."""
        if self.local_classifier is not None:
            return self.local_classifier
        if not self.local_classifier_available():
            self.local_classifier_error = "No se encontro local_models/condesan/condesan_local.json."
            return None
        try:
            self.local_classifier = LocalSpeciesClassifier.load()
            self.local_classifier_error = ""
        except Exception as ex:  # pragma: no cover - depende del entorno
            self.local_classifier_error = str(ex)
            self.local_classifier = None
        return self.local_classifier

    def _collect_audio_files(self, input_path: str) -> list[Path]:
        p = Path(input_path)
        if not p.exists():
            return []
        if p.is_file():
            return [p] if p.suffix.lower() in AUDIO_EXTENSIONS else []

        return sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS)

    def _configure_birdnet(self, audio_input: str) -> None:
        cfg.MODEL_PATH = cfg.BIRDNET_MODEL_PATH
        cfg.LABELS_FILE = cfg.BIRDNET_LABELS_FILE
        cfg.SAMPLE_RATE = cfg.BIRDNET_SAMPLE_RATE
        cfg.SIG_LENGTH = cfg.BIRDNET_SIG_LENGTH

        cfg.INPUT_PATH = audio_input
        cfg.SIG_OVERLAP = max(0.0, min(2.9, float(self.overlap_seconds)))
        cfg.AUDIO_SPEED = 1.0
        cfg.BANDPASS_FMIN = 0
        cfg.BANDPASS_FMAX = 15000
        cfg.BATCH_SIZE = max(1, int(self.batch_size))

        input_obj = Path(audio_input)
        if input_obj.is_dir():
            cfg.CPU_THREADS = max(1, int(self.threads))
            cfg.TFLITE_THREADS = 1
        else:
            cfg.CPU_THREADS = 1
            cfg.TFLITE_THREADS = max(1, int(self.threads))

    def _estimate_total_segments(self, files: list[Path]) -> int:
        total = 0
        step = max(0.1, self.chunk_seconds - self.overlap_seconds)
        for file_path in files:
            try:
                duration = float(sf.info(str(file_path)).duration)
                segments = max(1, int(math.ceil(max(duration - self.chunk_seconds, 0.0) / step)) + 1)
            except Exception:
                segments = 1
            total += segments
        return max(total, 1)

    def _reduce_noise(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        arr = np.asarray(audio, dtype=np.float32)
        if nr is None:
            return arr

        if arr.ndim == 1:
            cleaned = nr.reduce_noise(y=arr, sr=sample_rate, stationary=True, prop_decrease=0.9)
            return np.asarray(cleaned, dtype=np.float32)

        if arr.ndim == 2:
            cleaned_channels = []
            for ch in range(arr.shape[1]):
                cleaned_ch = nr.reduce_noise(y=arr[:, ch], sr=sample_rate, stationary=True, prop_decrease=0.9)
                cleaned_channels.append(np.asarray(cleaned_ch, dtype=np.float32))
            return np.stack(cleaned_channels, axis=1).astype(np.float32)

        return arr

    def _prepare_file_for_embeddings(self, file_path: Path) -> tuple[Path, TemporaryDirectory | None]:
        if not self.use_noise_reduction:
            return file_path, None

        if nr is None:
            log.warning("Noise reduction requested but noisereduce is unavailable. Using raw audio.")
            return file_path, None

        tmp_dir = TemporaryDirectory(prefix="ecoacoustic_nr_")
        try:
            audio, sample_rate = sf.read(str(file_path), always_2d=False)
            cleaned = self._reduce_noise(np.asarray(audio), int(sample_rate))
            denoised_path = Path(tmp_dir.name) / f"{file_path.stem}_nr.wav"
            sf.write(str(denoised_path), cleaned, int(sample_rate))
            return denoised_path, tmp_dir
        except Exception:
            tmp_dir.cleanup()
            raise

    def _extract_file_embeddings(self, file_path: Path):
        prepared_path, tmp_dir = self._prepare_file_for_embeddings(file_path)
        try:
            self._configure_birdnet(str(prepared_path))
            for start, end, emb in iterate_audio_chunks(str(prepared_path), embeddings=True):
                vec = np.asarray(emb, dtype=np.float32).reshape(-1)
                if np.linalg.norm(vec) <= 1e-12:
                    continue
                yield float(start), float(end), _l2_normalize(vec)
        finally:
            if tmp_dir is not None:
                tmp_dir.cleanup()

    def _extract_file_raw_embeddings(self, file_path: Path):
        """Como _extract_file_embeddings pero sin normalizar (requerido por el clasificador local)."""
        prepared_path, tmp_dir = self._prepare_file_for_embeddings(file_path)
        try:
            self._configure_birdnet(str(prepared_path))
            for start, end, emb in iterate_audio_chunks(str(prepared_path), embeddings=True):
                vec = np.asarray(emb, dtype=np.float32).reshape(-1)
                if np.linalg.norm(vec) <= 1e-12:
                    continue
                yield float(start), float(end), vec
        finally:
            if tmp_dir is not None:
                tmp_dir.cleanup()

    def _cluster_profile_vectors(
        self,
        vectors: np.ndarray,
        *,
        use_multi_vector: bool = True,
        min_chunks_for_cluster: int = 10,
        max_prototypes: int = 5,
    ) -> np.ndarray:
        matrix = _normalize_rows(np.asarray(vectors, dtype=np.float32))
        if matrix.shape[0] == 0:
            raise ValueError("No hay vectores para crear perfil.")

        if (not use_multi_vector) or matrix.shape[0] <= min_chunks_for_cluster:
            return _normalize_rows(matrix.mean(axis=0, keepdims=True))

        n_clusters = min(max_prototypes, matrix.shape[0])
        if n_clusters <= 1:
            return _normalize_rows(matrix.mean(axis=0, keepdims=True))

        if matrix.shape[0] > 1000:
            model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256, n_init="auto")
        else:
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        model.fit(matrix)
        centers = np.asarray(model.cluster_centers_, dtype=np.float32)
        return _normalize_rows(centers)

    def build_reference_vectors(self, reference_audio: str, *, use_multi_vector: bool = True) -> tuple[np.ndarray, int]:
        vectors: list[np.ndarray] = []
        for _, _, emb in self._extract_file_embeddings(Path(reference_audio)):
            vectors.append(emb)
        if not vectors:
            raise ValueError("No se pudieron extraer embeddings validos del audio de referencia.")

        matrix = np.vstack(vectors).astype(np.float32)
        profile_vectors = self._cluster_profile_vectors(matrix, use_multi_vector=use_multi_vector)
        return profile_vectors, len(vectors)

    def add_reference_profile(
        self,
        profile_name: str,
        reference_audio: str,
        *,
        use_multi_vector: bool = True,
    ) -> tuple[ReferenceProfile, int]:
        vectors, used_chunks = self.build_reference_vectors(reference_audio, use_multi_vector=use_multi_vector)
        profile = self.profile_manager.save_profile(profile_name, reference_audio, vectors)
        return profile, used_chunks

    def extract_segment_embedding(
        self,
        audio_path: str,
        start_sec: float,
        end_sec: float,
    ) -> np.ndarray:
        target_start = float(start_sec)
        target_end = float(end_sec)
        best_delta = float("inf")
        best_vec: np.ndarray | None = None

        for start, end, emb in self._extract_file_embeddings(Path(audio_path)):
            delta = abs(start - target_start) + abs(end - target_end)
            if delta < best_delta:
                best_delta = delta
                best_vec = emb
            if abs(start - target_start) <= 0.06 and abs(end - target_end) <= 0.06:
                best_vec = emb
                break

        if best_vec is None:
            raise ValueError("No se pudo extraer embedding para el segmento seleccionado.")
        return _l2_normalize(best_vec)

    def refine_profile_with_embedding(
        self,
        profile_name: str,
        new_embedding: np.ndarray,
        *,
        use_multi_vector: bool = True,
    ) -> tuple[ReferenceProfile, int]:
        existing = self.profile_manager.load_single_profile_vectors(profile_name)
        new_vec = _l2_normalize(np.asarray(new_embedding, dtype=np.float32).reshape(-1))
        combined = np.vstack([existing, new_vec.reshape(1, -1)]).astype(np.float32)

        if use_multi_vector:
            updated = self._cluster_profile_vectors(combined, use_multi_vector=True)
        else:
            updated = _normalize_rows(combined)

        profile = self.profile_manager.update_profile_vectors(profile_name, updated)
        return profile, int(combined.shape[0])

    def _score_all_species(
        self,
        embedding: np.ndarray,
        profile_vectors: dict[str, np.ndarray],
        *,
        use_multi_vector: bool = True,
    ) -> tuple[str, float]:
        emb_t = torch.from_numpy(_l2_normalize(embedding).astype(np.float32)).to(self.device)

        best_name = ""
        best_score = -1.0
        for name, vectors in profile_vectors.items():
            matrix = _normalize_rows(vectors)
            if not use_multi_vector and matrix.shape[0] > 1:
                matrix = _normalize_rows(matrix.mean(axis=0, keepdims=True))

            vectors_t = torch.from_numpy(matrix).to(self.device)
            emb_rep = emb_t.unsqueeze(0).expand(vectors_t.shape[0], -1)
            scores = self.cosine(vectors_t, emb_rep)
            species_score = float(torch.max(scores).item())
            if species_score > best_score:
                best_score = species_score
                best_name = name
        return best_name, best_score

    def _apply_temporal_smoothing(
        self,
        detections: list[dict],
        *,
        window_size: int = 3,
        threshold: float | None = None,
    ) -> list[dict]:
        if not detections:
            return detections

        win = max(3, int(window_size))
        half = win // 2
        th = float(threshold if threshold is not None else self.similarity_threshold)
        high_conf = min(0.99, th + 0.12)
        medium_low = max(0.0, th - 0.20)

        by_file: dict[str, list[dict]] = {}
        for d in detections:
            by_file.setdefault(d["file"], []).append(dict(d))

        smoothed_all: list[dict] = []
        for file_path, rows in by_file.items():
            rows = sorted(rows, key=lambda x: (x["start"], x["end"]))

            boosted = [dict(r) for r in rows]
            for i, center in enumerate(rows):
                left = max(0, i - half)
                right = min(len(rows), i + half + 1)
                center_species = center["common"]
                center_conf = float(center["confidence"])
                if center_conf < high_conf:
                    continue

                for j in range(left, right):
                    if j == i:
                        continue
                    if boosted[j]["common"] != center_species:
                        continue
                    neighbor_conf = float(boosted[j]["confidence"])
                    if medium_low <= neighbor_conf < th:
                        boosted[j]["confidence"] = min(0.99, max(neighbor_conf, (neighbor_conf + center_conf) * 0.5 + 0.05))

            cleaned: list[dict] = []
            for i, row in enumerate(boosted):
                duration = float(row["end"]) - float(row["start"])
                conf = float(row["confidence"])
                prev_same = i > 0 and boosted[i - 1]["common"] == row["common"]
                next_same = i < len(boosted) - 1 and boosted[i + 1]["common"] == row["common"]
                isolated = not prev_same and not next_same
                marginal = conf < (th + 0.05)

                if duration < 0.5 and isolated and marginal:
                    continue
                cleaned.append(row)

            smoothed_all.extend(cleaned)

        smoothed_all.sort(key=lambda x: (x["file"], x["start"], x["end"]))
        return smoothed_all

    def _csv_get(self, row: dict, *candidates: str) -> str:
        normalized = {str(k).strip().lower(): str(v).strip() for k, v in row.items()}
        for key in candidates:
            value = normalized.get(key.lower())
            if value is not None:
                return value
        return ""

    def _safe_float(self, value: str) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    def _parse_global_csv(self, csv_path: Path, *, default_file: str) -> list[dict]:
        rows: list[dict] = []
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    start = self._safe_float(self._csv_get(row, "start (s)", "start"))
                    end = self._safe_float(self._csv_get(row, "end (s)", "end"))
                    if end <= start:
                        end = start + max(0.1, self.chunk_seconds)

                    scientific = self._csv_get(row, "scientific name", "scientific")
                    common = self._csv_get(row, "common name", "common")
                    confidence = self._safe_float(self._csv_get(row, "confidence"))
                    file_path = self._csv_get(row, "file", "audio file") or default_file
                    rows.append(
                        {
                            "start": float(start),
                            "end": float(end),
                            "scientific": scientific or common or "Unknown",
                            "common": common or scientific or "Unknown",
                            "confidence": float(confidence),
                            "file": file_path,
                            "source": "birdnet_global",
                        }
                    )
        except Exception:
            return []
        return rows

    def _invoke_birdnet_module(
        self,
        file_path: Path,
        output_dir: Path,
        *,
        latitude: float | None = None,
        longitude: float | None = None,
        log_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        """Ejecuta BirdNET sobre un archivo y deja el CSV de resultados en ``output_dir``."""
        kwargs = {
            "rtype": "csv",
            "overlap": float(self.overlap_seconds),
            "threads": max(1, int(self.threads)),
            "sensitivity": float(self.sensitivity),
            "min_conf": float(self.global_min_conf),
        }
        if latitude is not None and longitude is not None:
            kwargs["lat"] = float(latitude)
            kwargs["lon"] = float(longitude)
        try:
            birdnet_run(str(file_path), str(output_dir), **kwargs)
            return
        except TypeError as ex:
            # Otra version de birdnet_analyzer con una firma distinta.
            if getattr(sys, "frozen", False):
                raise RuntimeError(f"La version instalada de BirdNET no admite estos parametros: {ex}") from ex
            if log_callback:
                log_callback("Warning", f"API de BirdNET incompatible ({ex}). Usando la linea de comandos...")

        cmd = [
            sys.executable,
            "-m",
            "birdnet_analyzer.analyze",
            str(file_path),
            "-o",
            str(output_dir),
            "--rtype",
            "csv",
            "--overlap",
            f"{self.overlap_seconds:.2f}",
            "--sensitivity",
            f"{self.sensitivity:.2f}",
            "--min_conf",
            f"{self.global_min_conf:.2f}",
            "-t",
            str(max(1, int(self.threads))),
        ]
        if latitude is not None and longitude is not None:
            cmd.extend(["--lat", f"{float(latitude):.6f}", "--lon", f"{float(longitude):.6f}"])
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()[-600:]
            raise RuntimeError(f"BirdNET termino con error ({result.returncode}): {detail}")

    def _run_global_discovery_for_file(
        self,
        file_path: Path,
        *,
        latitude: float | None = None,
        longitude: float | None = None,
        log_callback: Callable[[str, str], None] | None = None,
    ) -> list[dict]:
        prepared_path, tmp_nr = self._prepare_file_for_embeddings(file_path)
        try:
            with TemporaryDirectory(prefix="ecoacoustic_global_") as tmp:
                out_dir = Path(tmp)
                self._configure_birdnet(str(prepared_path))
                cfg.OUTPUT_PATH = str(out_dir)
                cfg.RESULT_TYPES = ["csv"]
                if latitude is not None and longitude is not None:
                    cfg.LATITUDE = float(latitude)
                    cfg.LONGITUDE = float(longitude)
                else:
                    cfg.LATITUDE = None
                    cfg.LONGITUDE = None

                self._invoke_birdnet_module(
                    prepared_path, out_dir, latitude=latitude, longitude=longitude, log_callback=log_callback
                )

                csv_paths = sorted(out_dir.rglob("*.BirdNET.results.csv"))
                if not csv_paths:
                    csv_paths = sorted(out_dir.rglob("*.csv"))

                detections: list[dict] = []
                for csv_path in csv_paths:
                    detections.extend(self._parse_global_csv(csv_path, default_file=str(file_path)))
                return detections
        finally:
            if tmp_nr is not None:
                tmp_nr.cleanup()

    def _is_time_overlap(self, a: dict, b: dict, *, min_overlap: float = 0.0) -> bool:
        if str(a.get("file", "")) != str(b.get("file", "")):
            return False
        start = max(float(a.get("start", 0.0)), float(b.get("start", 0.0)))
        end = min(float(a.get("end", 0.0)), float(b.get("end", 0.0)))
        return (end - start) > float(min_overlap)

    def _merge_custom_and_global(self, custom_rows: list[dict], global_rows: list[dict]) -> list[dict]:
        if not custom_rows:
            out = list(global_rows)
            out.sort(key=lambda x: (x.get("file", ""), x.get("start", 0.0), x.get("end", 0.0)))
            return out

        merged = list(custom_rows)
        for row in global_rows:
            if any(self._is_time_overlap(row, c, min_overlap=0.01) for c in custom_rows):
                continue
            merged.append(row)
        merged.sort(key=lambda x: (x.get("file", ""), x.get("start", 0.0), x.get("end", 0.0)))
        return merged

    def _build_anomaly_rows(
        self,
        global_rows: list[dict],
        final_custom_rows: list[dict],
        *,
        anomaly_threshold: float,
    ) -> list[dict]:
        anomalies: list[dict] = []
        for row in global_rows:
            conf = float(row.get("confidence", 0.0))
            if conf < anomaly_threshold:
                continue
            if any(self._is_time_overlap(row, known, min_overlap=0.05) for known in final_custom_rows):
                continue

            anomalies.append(
                {
                    "start": float(row.get("start", 0.0)),
                    "end": float(row.get("end", 0.0)),
                    "scientific": row.get("scientific", "Unknown") or "Unknown",
                    "common": ANOMALY_COMMON_LABEL,
                    "confidence": conf,
                    "file": row.get("file", ""),
                    "source": "anomaly",
                    "birdnet_common": row.get("common", ""),
                    "birdnet_scientific": row.get("scientific", ""),
                }
            )
        anomalies.sort(key=lambda x: (x.get("file", ""), x.get("start", 0.0), x.get("end", 0.0)))
        return anomalies

    def _compose_final_detections(
        self,
        final_custom_rows: list[dict],
        global_rows: list[dict],
        *,
        has_custom_profiles: bool,
        use_global_discovery: bool,
        anomaly_threshold: float,
    ) -> list[dict]:
        if not use_global_discovery:
            out = list(final_custom_rows)
            out.sort(key=lambda x: (x.get("file", ""), x.get("start", 0.0), x.get("end", 0.0)))
            return out

        if not has_custom_profiles:
            return self._merge_custom_and_global(final_custom_rows, global_rows)

        anomalies = self._build_anomaly_rows(
            global_rows,
            final_custom_rows,
            anomaly_threshold=anomaly_threshold,
        )
        merged = list(final_custom_rows) + anomalies
        merged.sort(key=lambda x: (x.get("file", ""), x.get("start", 0.0), x.get("end", 0.0)))
        return merged

    def analyze(
        self,
        input_path: str,
        *,
        similarity_threshold: float | None = None,
        anomaly_threshold: float | None = None,
        use_noise_reduction: bool | None = None,
        use_multi_vector: bool = True,
        smoothing_window: int = 3,
        use_global_discovery: bool = False,
        latitude: float | None = None,
        longitude: float | None = None,
        progress_callback: Callable[[int, int, str, int, float], None] | None = None,
        log_callback: Callable[[str, str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        use_local_classifier: bool = False,
        local_threshold_scale: float = 1.0,
    ) -> tuple[list[dict], dict]:
        threshold = float(similarity_threshold if similarity_threshold is not None else self.similarity_threshold)
        anomaly_th = float(anomaly_threshold if anomaly_threshold is not None else self.anomaly_threshold)
        anomaly_th = max(0.01, min(0.99, anomaly_th))
        if use_noise_reduction is not None:
            self.use_noise_reduction = bool(use_noise_reduction)
        profile_vectors = self.profile_manager.load_profile_vectors()
        has_custom_profiles = bool(profile_vectors)

        # Clasificador local CONDESAN (opcional): capa de BirdNET restringida a las
        # especies locales con umbrales calibrados por especie.
        local_clf = None
        if use_local_classifier:
            local_clf = self.get_local_classifier()
            if log_callback:
                if local_clf is None:
                    log_callback("Warning", f"Clasificador local no disponible: {self.local_classifier_error}")
                else:
                    log_callback("Info", f"Clasificador local CONDESAN activo ({local_clf.describe()}).")
        use_local = local_clf is not None

        if not has_custom_profiles and not use_global_discovery and not use_local:
            raise ValueError("No hay perfiles de referencia cargados. Agrega especies en 'Biblioteca de Especies'.")

        files = self._collect_audio_files(input_path)
        if not files:
            raise ValueError("No hay archivos de audio validos para analizar.")
        if self.use_noise_reduction and log_callback:
            log_callback("Info", "Reduccion de ruido DSP activada para extraccion y descubrimiento global.")

        uses_embeddings = has_custom_profiles or use_local
        total_segments = self._estimate_total_segments(files) if uses_embeddings else max(len(files), 1)
        processed = 0
        candidates: list[dict] = []
        local_rows: list[dict] = []
        global_candidates: list[dict] = []
        provisional_hits = 0
        stopped_by_user = False

        def known_rows(final_custom: list[dict]) -> list[dict]:
            rows = list(final_custom) + list(local_rows)
            rows.sort(key=lambda x: (x.get("file", ""), x.get("start", 0.0), x.get("end", 0.0)))
            return rows

        for file_path in files:
            if log_callback:
                log_callback("Info", f"Analizando: {file_path.name}")

            file_custom_hits = 0
            file_best_custom_conf = 0.0
            file_local_hits = 0
            if uses_embeddings:
                for start, end, raw in self._extract_file_raw_embeddings(file_path):
                    if should_stop and should_stop():
                        smoothed = (
                            self._apply_temporal_smoothing(candidates, window_size=smoothing_window, threshold=threshold)
                            if has_custom_profiles
                            else []
                        )
                        final = [d for d in smoothed if float(d["confidence"]) >= threshold]
                        final = self._compose_final_detections(
                            known_rows(final),
                            global_candidates,
                            has_custom_profiles=has_custom_profiles,
                            use_global_discovery=use_global_discovery,
                            anomaly_threshold=anomaly_th,
                        )
                        summary = {
                            "processed_segments": processed,
                            "total_segments": total_segments,
                            "detections": len(final),
                            "avg_confidence": float(np.mean([d["confidence"] for d in final])) if final else 0.0,
                            "stopped": True,
                        }
                        return final, summary

                    processed += 1
                    if has_custom_profiles:
                        emb = _l2_normalize(raw)
                        matched_name, best_score = self._score_all_species(emb, profile_vectors, use_multi_vector=use_multi_vector)
                        if best_score >= threshold:
                            provisional_hits += 1
                            file_custom_hits += 1
                        if best_score > file_best_custom_conf:
                            file_best_custom_conf = float(best_score)

                        candidates.append(
                            {
                                "start": float(start),
                                "end": float(end),
                                "scientific": matched_name,
                                "common": matched_name,
                                "confidence": float(best_score),
                                "file": str(file_path),
                                "source": "custom",
                            }
                        )

                    if use_local:
                        rows = local_clf.detections(raw, start, end, str(file_path), threshold_scale=local_threshold_scale)
                        if rows:
                            local_rows.extend(rows)
                            provisional_hits += len(rows)
                            file_local_hits += len(rows)

                    if progress_callback:
                        confs = [d["confidence"] for d in candidates if d["confidence"] >= threshold]
                        confs.extend(d["confidence"] for d in local_rows)
                        avg_conf = float(np.mean(confs)) if provisional_hits and confs else 0.0
                        progress_callback(processed, total_segments, str(file_path), provisional_hits, avg_conf)

                if use_local and log_callback:
                    log_callback("Info", f"Clasificador local ({file_path.name}): {file_local_hits} detecciones.")

            if use_global_discovery:
                if should_stop and should_stop():
                    stopped_by_user = True
                    break
                if log_callback:
                    if has_custom_profiles:
                        log_callback(
                            "Info",
                            f"BirdNET global/anomalias en {file_path.name} "
                            f"(hits custom={file_custom_hits}, max={file_best_custom_conf:.3f}).",
                        )
                    else:
                        log_callback("Info", f"Descubrimiento global BirdNET: {file_path.name}")

                global_rows = self._run_global_discovery_for_file(
                    file_path,
                    latitude=latitude,
                    longitude=longitude,
                    log_callback=log_callback,
                )
                global_candidates.extend(global_rows)
                if log_callback:
                    log_callback("Info", f"Detecciones globales ({file_path.name}): {len(global_rows)}")

                if not uses_embeddings and progress_callback:
                    processed += 1
                    avg_global = float(np.mean([d["confidence"] for d in global_candidates])) if global_candidates else 0.0
                    progress_callback(processed, total_segments, str(file_path), len(global_candidates), avg_global)

        smoothed = self._apply_temporal_smoothing(candidates, window_size=smoothing_window, threshold=threshold) if has_custom_profiles else []
        final_custom = [d for d in smoothed if float(d["confidence"]) >= threshold]
        final_detections = self._compose_final_detections(
            known_rows(final_custom),
            global_candidates,
            has_custom_profiles=has_custom_profiles,
            use_global_discovery=use_global_discovery,
            anomaly_threshold=anomaly_th,
        )
        processed_summary = processed if uses_embeddings else total_segments

        summary = {
            "processed_segments": processed_summary,
            "total_segments": total_segments,
            "detections": len(final_detections),
            "avg_confidence": float(np.mean([d["confidence"] for d in final_detections])) if final_detections else 0.0,
            "stopped": stopped_by_user,
        }
        return final_detections, summary

    def export_detections(
        self,
        output_dir: str,
        detections: list[dict],
        formats: list[str],
        zone_metadata: dict | None = None,
    ) -> list[Path]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exported: list[Path] = []

        if "csv" in formats:
            csv_path = out_dir / f"EmbeddingDetections_{timestamp}.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                header = ["Start (s)", "End (s)", "Scientific name", "Common name", "Confidence", "File", "Source"]
                if zone_metadata:
                    header.extend(["Country", "Province", "Canton", "Latitude", "Longitude", "RadiusKm"])
                writer.writerow(header)

                for row in detections:
                    line = [
                        f"{row['start']:.2f}",
                        f"{row['end']:.2f}",
                        row["scientific"],
                        row["common"],
                        f"{row['confidence']:.4f}",
                        row["file"],
                        row.get("source", "custom"),
                    ]
                    if zone_metadata:
                        line.extend(
                            [
                                zone_metadata.get("country", ""),
                                zone_metadata.get("province", ""),
                                zone_metadata.get("canton", ""),
                                zone_metadata.get("lat", ""),
                                zone_metadata.get("lon", ""),
                                zone_metadata.get("radius_km", ""),
                            ]
                        )
                    writer.writerow(line)
            exported.append(csv_path)

        if "table" in formats:
            table_path = out_dir / f"EmbeddingDetections_{timestamp}.BirdNET.selection.table.txt"
            with table_path.open("w", encoding="utf-8") as f:
                f.write(
                    "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\t"
                    "Species Code\tCommon Name\tScientific Name\tConfidence\n"
                )
                for idx, row in enumerate(detections, start=1):
                    sp = row["common"]
                    code = re.sub(r"[^A-Z]", "", sp.upper())[:6] or "CUSTOM"
                    f.write(
                        f"{idx}\tSpectrogram 1\t1\t{row['start']:.2f}\t{row['end']:.2f}\t0\t15000\t{code}\t"
                        f"{row['common']}\t{row['scientific']}\t{row['confidence']:.4f}\n"
                    )
            exported.append(table_path)

        if "audacity" in formats:
            aud_path = out_dir / f"EmbeddingDetections_{timestamp}.Audacity.labels.txt"
            with aud_path.open("w", encoding="utf-8") as f:
                for row in detections:
                    f.write(f"{row['start']:.2f}\t{row['end']:.2f}\t{row['common']} ({row['confidence']:.2f})\n")
            exported.append(aud_path)

        if "kaleidoscope" in formats:
            kal_path = out_dir / f"EmbeddingDetections_{timestamp}.kaleidoscope.csv"
            with kal_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["FOLDER", "IN FILE", "OFFSET", "DURATION", "SCIENTIFIC NAME", "COMMON NAME", "CONFIDENCE"])
                for row in detections:
                    file_path = Path(row["file"])
                    writer.writerow(
                        [
                            str(file_path.parent),
                            file_path.name,
                            f"{row['start']:.2f}",
                            f"{max(0.1, row['end'] - row['start']):.2f}",
                            row["scientific"],
                            row["common"],
                            f"{row['confidence']:.4f}",
                        ]
                    )
            exported.append(kal_path)

        return exported


class AudioWorker(QObject):
    progress = Signal(int, int, str, int, float)
    log = Signal(str, str)
    finished = Signal(object, object, object)  # detections, summary, exported_files
    failed = Signal(str)

    def __init__(
        self,
        analyzer: AudioAnalyzer,
        input_path: str,
        output_dir: str,
        export_formats: list[str],
        *,
        zone_metadata: dict | None = None,
        similarity_threshold: float | None = None,
        anomaly_threshold: float | None = None,
        overlap_seconds: float | None = None,
        threads: int | None = None,
        use_noise_reduction: bool = False,
        use_multi_vector: bool = True,
        smoothing_window: int = 3,
        use_global_discovery: bool = False,
        latitude: float | None = None,
        longitude: float | None = None,
        use_local_classifier: bool = False,
        sensitivity: float = 1.0,
    ) -> None:
        super().__init__()
        self.analyzer = analyzer
        self.input_path = input_path
        self.output_dir = output_dir
        self.export_formats = export_formats
        self.zone_metadata = zone_metadata
        self.similarity_threshold = similarity_threshold
        self.anomaly_threshold = anomaly_threshold
        self.overlap_seconds = overlap_seconds
        self.threads = threads
        self.use_noise_reduction = bool(use_noise_reduction)
        self.use_multi_vector = bool(use_multi_vector)
        self.smoothing_window = int(smoothing_window)
        self.use_global_discovery = bool(use_global_discovery)
        self.latitude = latitude
        self.longitude = longitude
        self.use_local_classifier = bool(use_local_classifier)
        self.sensitivity = float(sensitivity)
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def _should_stop(self) -> bool:
        return self._stop_requested

    @Slot()
    def run(self) -> None:
        try:
            if self.overlap_seconds is not None:
                self.analyzer.overlap_seconds = float(self.overlap_seconds)
            if self.threads is not None:
                self.analyzer.threads = int(self.threads)
            self.analyzer.sensitivity = self.sensitivity

            detections, summary = self.analyzer.analyze(
                self.input_path,
                similarity_threshold=self.similarity_threshold,
                anomaly_threshold=self.anomaly_threshold,
                use_noise_reduction=self.use_noise_reduction,
                use_multi_vector=self.use_multi_vector,
                smoothing_window=self.smoothing_window,
                use_global_discovery=self.use_global_discovery,
                latitude=self.latitude,
                longitude=self.longitude,
                use_local_classifier=self.use_local_classifier,
                progress_callback=lambda done, total, current_file, det_count, avg_conf: self.progress.emit(
                    done, total, current_file, det_count, avg_conf
                ),
                log_callback=lambda level, text: self.log.emit(level, text),
                should_stop=self._should_stop,
            )
            exported = self.analyzer.export_detections(
                self.output_dir,
                detections,
                self.export_formats,
                zone_metadata=self.zone_metadata,
            )
            self.finished.emit(detections, summary, [str(p) for p in exported])
        except Exception as ex:
            self.failed.emit(str(ex))
