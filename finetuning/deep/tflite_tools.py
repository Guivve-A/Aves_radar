"""Utilidades sobre el archivo TFLite de BirdNET V2.4.

BirdNET se distribuye como TFLite (no entrenable) y el delegado XNNPACK no deja
leer tensores intermedios. Aqui se reescribe el archivo anadiendo como salidas
del grafo los tensores que interesan (embedding y mapas de activacion a la
salida de los bloques 2 y 3). El calculo no cambia: solo se exponen tensores
que ya existen, de modo que la inferencia sigue siendo rapida.

    Identity                        (N, 6522)        logits de las 6 522 clases
    model/GLOBAL_AVG_POOL/Mean      (N, 1024)        embedding
    model/BLOCK_3-5_ADD/add         (N, 6, 16, 108)  entrada del bloque 4
    model/BLOCK_2-4_ADD/add         (N, 12, 32, 72)  entrada del bloque 3
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import config

EMB_TENSOR = "model/GLOBAL_AVG_POOL/Mean"
F3_TENSOR = "model/BLOCK_3-5_ADD/add"
F2_TENSOR = "model/BLOCK_2-4_ADD/add"
MULTI_OUT_FILE = "BirdNET_V2.4_multiout.tflite"


def _schema():
    try:
        from ai_edge_litert import schema_py_generated as schema  # type: ignore
    except Exception:  # pragma: no cover - instalacion con TensorFlow completo
        from tensorflow.lite.python import schema_py_generated as schema  # type: ignore
    return schema


def interpreter_class():
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore
    except Exception:  # pragma: no cover
        from tensorflow import lite as _tflite  # type: ignore

        Interpreter = _tflite.Interpreter
    return Interpreter


def source_model_path(model_dir: Path | str | None = None) -> Path:
    path = Path(model_dir or config.BIRDNET_MODEL_DIR) / config.BIRDNET_MODEL_FILE
    if not path.exists():
        raise FileNotFoundError(f"No se encontro {path}. Define ECO_BIRDNET_MODEL_DIR.")
    return path


def load_labels(model_dir: Path | str | None = None) -> list[str]:
    path = Path(model_dir or config.BIRDNET_MODEL_DIR) / config.BIRDNET_LABELS_FILE
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_multi_output_model(
    src: Path | str | None = None,
    dst: Path | str | None = None,
    extra: tuple[str, ...] = (EMB_TENSOR, F3_TENSOR, F2_TENSOR),
) -> Path:
    """Escribe una copia del TFLite cuyas salidas son [logits, *extra]."""
    import flatbuffers

    schema = _schema()
    src = Path(src or source_model_path())
    dst = Path(dst or src.with_name(MULTI_OUT_FILE))
    buf = bytearray(src.read_bytes())
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(buf, 0))
    graph = model.subgraphs[0]
    names = [t.name.decode() for t in graph.tensors]
    outputs = [int(i) for i in graph.outputs]
    for name in extra:
        if name not in names:
            raise KeyError(f"Tensor no encontrado en el modelo: {name}")
        idx = names.index(name)
        if idx not in outputs:
            outputs.append(idx)
    graph.outputs = np.array(outputs, dtype=np.int32)
    builder = flatbuffers.Builder(len(buf) + 4096)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    dst.write_bytes(bytes(builder.Output()))
    return dst


class MultiOutputBirdNET:
    """Inferencia por lotes con todas las salidas (logits, embedding y mapas intermedios)."""

    def __init__(self, model_dir: Path | str | None = None, threads: int = 4):
        src = source_model_path(model_dir)
        self.path = src.with_name(MULTI_OUT_FILE)
        if not self.path.exists() or self.path.stat().st_mtime < src.stat().st_mtime:
            build_multi_output_model(src, self.path)
        self.interpreter = interpreter_class()(model_path=str(self.path), num_threads=max(1, int(threads)))
        self._input = self.interpreter.get_input_details()[0]["index"]
        self._outputs = {d["name"]: d["index"] for d in self.interpreter.get_output_details()}
        self._batch = None

    def __call__(self, chunks: np.ndarray) -> dict[str, np.ndarray]:
        chunks = np.ascontiguousarray(chunks, dtype=np.float32)
        if self._batch != len(chunks):
            self.interpreter.resize_tensor_input(self._input, list(chunks.shape))
            self.interpreter.allocate_tensors()
            self._batch = len(chunks)
        self.interpreter.set_tensor(self._input, chunks)
        self.interpreter.invoke()
        get = self.interpreter.get_tensor
        return {
            "logits": get(self._outputs["Identity"]).copy(),
            "emb": get(self._outputs[EMB_TENSOR]).copy(),
            "f3": get(self._outputs[F3_TENSOR]).copy(),
            "f2": get(self._outputs[F2_TENSOR]).copy(),
        }


def graph_ops(model_path: Path | str | None = None):
    """Operadores y tensores del grafo original (sin delegados), para construir la replica en PyTorch."""
    schema = _schema()
    path = Path(model_path or source_model_path())
    buf = bytearray(path.read_bytes())
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(buf, 0))
    graph = model.subgraphs[0]
    opnames = {v: k for k, v in vars(schema.BuiltinOperator).items() if not k.startswith("_")}
    codes = [max(int(c.builtinCode), int(c.deprecatedBuiltinCode)) for c in model.operatorCodes]
    tensors = []
    for t in graph.tensors:
        data = model.buffers[t.buffer].data if t.buffer < len(model.buffers) else None
        tensors.append({"name": t.name.decode(), "shape": [int(x) for x in (t.shape if t.shape is not None else [])], "type": int(t.type), "has_data": data is not None and len(data) > 0, "buffer": int(t.buffer)})
    ops = []
    for i, op in enumerate(graph.operators):
        ops.append(
            {
                "index": i,
                "op": opnames[codes[op.opcodeIndex]],
                "inputs": [int(x) for x in op.inputs],
                "outputs": [int(x) for x in op.outputs],
                "options": op.builtinOptions,
            }
        )
    return model, graph, tensors, ops


def constant(model, graph, index: int) -> np.ndarray:
    """Valor de un tensor constante (pesos) del archivo TFLite."""
    schema = _schema()
    t = graph.tensors[index]
    raw = model.buffers[t.buffer].data
    dtype = {schema.TensorType.FLOAT32: np.float32, schema.TensorType.INT32: np.int32, schema.TensorType.INT64: np.int64, schema.TensorType.FLOAT16: np.float16}[t.type]
    arr = np.frombuffer(bytes(raw) if not isinstance(raw, np.ndarray) else raw.tobytes(), dtype=dtype)
    shape = [int(x) for x in (t.shape if t.shape is not None else [])]
    return arr.reshape(shape).copy() if shape else arr.copy()
