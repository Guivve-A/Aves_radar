"""Replica entrenable (PyTorch) de los bloques superiores de BirdNET V2.4.

BirdNET solo se distribuye como TFLite, que no se puede entrenar. Este modulo
lee el grafo y los pesos del archivo TFLite y construye un ``torch.nn.Module``
que ejecuta exactamente los mismos operadores desde un mapa de activacion
intermedio hasta el embedding y los logits:

    corte "g44"  salida del bloque 4 (3 x 8 x 192)   -> BN+ReLU -> POST_CONV 3x3 -> embedding -> logits
    corte "g42"  salida del sub-bloque 4-2           -> sub-bloques 4-3 y 4-4 -> ...
    corte "f3"   salida del bloque 3 (6 x 16 x 108)  -> bloque 4 completo -> ...
    corte "f2"   salida del bloque 2 (12 x 32 x 72)  -> bloques 3 y 4 -> ...

Las capas inferiores quedan congeladas (sus activaciones se guardan en disco
con ``finetuning.deep.extract``), de modo que afinar las capas profundas es
viable en CPU. ``verify`` comprueba que la replica reproduce las salidas del
interprete TFLite.

Convenciones: TFLite usa NHWC; aqui todo se ejecuta en NCHW. La normalizacion
por lotes ya viene fusionada en los pesos de las convoluciones.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import tflite_tools as T

CUT_TENSORS = {"f3": T.F3_TENSOR, "f2": T.F2_TENSOR, "g42": "model/BLOCK_4-2_ADD/add", "g44": "model/BLOCK_4-4_ADD/add"}
CUT_CHANNELS = {"f3": 108, "f2": 72, "g42": 192, "g44": 192}
GROUP_ORDER = ("block3", "b41", "b42", "b43", "b44", "post", "classifier")
_SKIP = {"SHAPE", "STRIDED_SLICE", "PACK"}  # solo calculan la forma del RESHAPE de los bloques SE


def _pname(name: str) -> str:
    """Nombre de parametro legible a partir del nombre del tensor TFLite."""
    part = name.split(";")[-1] if ";" in name else name
    part = part.replace("model/", "")
    return re.sub(r"[^0-9a-zA-Z]+", "_", part).strip("_")


def _same_pad(size: int, kernel: int, stride: int, dilation: int = 1) -> tuple[int, int]:
    eff = (kernel - 1) * dilation + 1
    total = max((-(-size // stride) - 1) * stride + eff - size, 0)
    return total // 2, total - total // 2


class BirdNETTop(nn.Module):
    """Operadores de BirdNET desde el corte indicado hasta los logits.

    Parametros:
        cut            "f3" o "f2".
        class_indices  columnas de la capa de clasificacion que se conservan
                       (por ejemplo, las 33 especies objetivo). ``None`` = las 6 522.
    """

    def __init__(self, cut: str = "f3", class_indices: list[int] | None = None, model_path: Path | str | None = None):
        super().__init__()
        if cut not in CUT_TENSORS:
            raise ValueError(f"corte desconocido: {cut}")
        model, graph, tensors, ops = T.graph_ops(model_path)
        names = [t["name"] for t in tensors]
        start = names.index(CUT_TENSORS[cut])
        first = next(i for i, op in enumerate(ops) if start in op["outputs"]) + 1
        self.cut = cut
        self.start_tensor = start
        self.emb_tensor = names.index(T.EMB_TENSOR)
        self.out_tensor = int(graph.outputs[0])
        self.params = nn.ParameterDict()
        self.steps: list[dict] = []
        self.class_indices = None if class_indices is None else [int(i) for i in class_indices]
        self.drop_path = 0.0
        self.head_dropout = 0.0
        self.relative: set[str] = set()
        self.name_to_tensor = {n: i for i, n in enumerate(names)}

        def add_param(tensor_index: int, array: np.ndarray, role: str) -> str:
            key = f"{_pname(names[tensor_index])}__{role}"
            if key not in self.params:
                self.params[key] = nn.Parameter(torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)))
            return key

        for op in ops[first:]:
            kind = op["op"]
            if kind in _SKIP:
                continue
            ins, outs, opt = op["inputs"], op["outputs"], op["options"]
            step = {"op": kind, "in": ins, "out": outs[0], "name": names[outs[0]]}
            act = int(getattr(opt, "fusedActivationFunction", 0) or 0) if opt is not None else 0
            step["act"] = act
            if kind in ("CONV_2D", "DEPTHWISE_CONV_2D"):
                w = T.constant(model, graph, ins[1])
                b = T.constant(model, graph, ins[2])
                if kind == "CONV_2D":
                    w = w.transpose(0, 3, 1, 2)  # (O, H, W, I) -> (O, I, H, W)
                    step["groups"] = 1
                else:
                    w = w.transpose(3, 0, 1, 2)  # (1, H, W, C) -> (C, 1, H, W)
                    step["groups"] = int(w.shape[0])
                step["w"] = add_param(outs[0], w, "w")
                step["b"] = add_param(outs[0], b, "b")
                step["stride"] = (int(opt.strideH), int(opt.strideW))
                step["dilation"] = (int(opt.dilationHFactor), int(opt.dilationWFactor))
                step["same"] = int(opt.padding) == 0
                step["x"] = ins[0]
            elif kind in ("MUL", "ADD"):
                const = [i for i in ins if tensors[i]["has_data"]]
                step["x"] = [i for i in ins if not tensors[i]["has_data"]]
                if const:
                    c = T.constant(model, graph, const[0]).astype(np.float32)
                    c = c.reshape(1, -1, 1, 1) if c.ndim == 1 and c.size > 1 else c.reshape(-1)
                    step["c"] = add_param(outs[0], c, "scale" if kind == "MUL" else "shift")
            elif kind == "LOGISTIC":
                step["x"] = ins[0]
            elif kind == "MEAN":
                axes = [int(a) for a in T.constant(model, graph, ins[1]).reshape(-1)]
                if sorted(axes) != [1, 2]:
                    raise NotImplementedError(f"MEAN sobre ejes {axes}")
                step["x"] = ins[0]
                step["keep"] = bool(getattr(opt, "keepDims", False))
            elif kind == "RESHAPE":
                step["x"] = ins[0]
            elif kind == "PAD":
                pads = T.constant(model, graph, ins[1]).reshape(-1, 2)
                step["x"] = ins[0]
                step["pad"] = (int(pads[2][0]), int(pads[2][1]), int(pads[1][0]), int(pads[1][1]))  # W antes/despues, H antes/despues
            elif kind == "FULLY_CONNECTED":
                w = T.constant(model, graph, ins[1])
                b = T.constant(model, graph, ins[2])
                if self.class_indices is not None:
                    w, b = w[self.class_indices], b[self.class_indices]
                step["w"] = add_param(outs[0], w, "w")
                step["b"] = add_param(outs[0], b, "b")
                step["x"] = ins[0]
            else:
                raise NotImplementedError(f"Operador TFLite no soportado: {kind}")
            self.steps.append(step)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _activate(x: torch.Tensor, act: int) -> torch.Tensor:
        if act == 0:
            return x
        if act == 1:
            return F.relu(x)
        if act == 3:
            return F.relu6(x)
        raise NotImplementedError(f"activacion fusionada {act}")

    def weight(self, key: str) -> torch.Tensor:
        """Peso efectivo: W0 + s * V en los tensores con parametrizacion relativa; el parametro tal cual en el resto."""
        if key in self.relative:
            return getattr(self, f"w0__{key}") + getattr(self, f"sc__{key}") * self.params[key]
        return self.params[key]

    def make_relative(self, groups: tuple[str, ...]) -> list[str]:
        """Reparametriza los grupos indicados como W = W0 + s * V, con V = 0 al inicio.

        ``s`` es la magnitud tipica (RMS) de cada canal de salida de W0. Con la normalizacion por lotes
        fusionada, los pesos de capas distintas difieren en dos ordenes de magnitud; un mismo paso absoluto
        de Adam moveria unas capas cien veces mas que otras. Entrenar V hace que el paso sea relativo, y un
        decaimiento desacoplado sobre V tira hacia los pesos de BirdNET (equivale a L2-SP).
        Los demas grupos quedan congelados. Devuelve las claves entrenables.
        """
        trainable = []
        by_group = self.groups()
        for g, keys in by_group.items():
            for key in keys:
                p = self.params[key]
                if g not in groups:
                    p.requires_grad_(False)
                    continue
                w0 = p.detach().clone()
                if float(w0.abs().max()) == 0.0:
                    p.requires_grad_(False)                       # sesgos que son cero en el archivo: se dejan asi
                    continue
                if w0.dim() == 4 and w0.shape[0] > 1:
                    rms = w0.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt()
                elif w0.dim() == 2:
                    rms = w0.pow(2).mean(dim=1, keepdim=True).sqrt()
                else:
                    rms = w0.pow(2).mean().sqrt().reshape([1] * w0.dim())
                if rms.numel() > 1:
                    rms = rms.clamp(min=0.1 * float(rms.median()))
                self.register_buffer(f"w0__{key}", w0)
                self.register_buffer(f"sc__{key}", rms.clamp(min=1e-8))
                self.params[key] = nn.Parameter(torch.zeros_like(w0))
                self.relative.add(key)
                trainable.append(key)
        return trainable

    def relative_drift(self) -> dict[str, float]:
        """Desviacion relativa ||W - W0|| / ||W0|| por grupo (0 = BirdNET sin cambios)."""
        out = {}
        with torch.no_grad():
            for g, keys in self.groups().items():
                num = sum(float((getattr(self, f"sc__{k}") * self.params[k]).pow(2).sum()) for k in keys if k in self.relative)
                den = sum(float(getattr(self, f"w0__{k}").pow(2).sum()) for k in keys if k in self.relative)
                if den > 0:
                    out[g] = (num / den) ** 0.5
        return out

    def forward(self, x: torch.Tensor, return_embedding: bool = False, stop_at: str | None = None):
        """``x``: activaciones del corte en NHWC (como se guardan) o NCHW.

        ``stop_at``: nombre de un tensor intermedio; si se indica, se devuelve ese tensor (en NHWC).
        """
        c = CUT_CHANNELS[self.cut]
        if x.shape[-1] == c and x.shape[1] != c:
            x = x.permute(0, 3, 1, 2)
        vals: dict[int, torch.Tensor] = {self.start_tensor: x.float()}
        stop = None if stop_at is None else self.name_to_tensor[stop_at]
        emb = None
        for s in self.steps:
            kind = s["op"]
            if kind in ("CONV_2D", "DEPTHWISE_CONV_2D"):
                h = vals[s["x"]]
                w = self.weight(s["w"])
                if s["same"]:
                    ph = _same_pad(h.shape[2], w.shape[2], s["stride"][0], s["dilation"][0])
                    pw = _same_pad(h.shape[3], w.shape[3], s["stride"][1], s["dilation"][1])
                    if any(ph) or any(pw):
                        h = F.pad(h, (pw[0], pw[1], ph[0], ph[1]))
                y = F.conv2d(h, w, self.weight(s["b"]), stride=s["stride"], dilation=s["dilation"], groups=s["groups"])
            elif kind == "MUL":
                a = vals[s["x"][0]]
                y = a * (self.weight(s["c"]) if "c" in s else vals[s["x"][1]])
            elif kind == "ADD":
                a = vals[s["x"][0]]
                if "c" in s:
                    y = a + self.weight(s["c"])
                else:
                    b = vals[s["x"][1]]
                    if self.training and self.drop_path > 0 and "_ADD" in s["name"]:
                        # profundidad estocastica sobre la rama residual (la ultima calculada)
                        keep = (torch.rand(a.shape[0], 1, 1, 1, device=a.device) >= self.drop_path).float() / (1.0 - self.drop_path)
                        branch, skip = (b, a) if s["x"][1] > s["x"][0] else (a, b)
                        y = skip + branch * keep
                    else:
                        y = a + b
            elif kind == "LOGISTIC":
                y = torch.sigmoid(vals[s["x"]])
            elif kind == "MEAN":
                y = vals[s["x"]].mean(dim=(2, 3), keepdim=s["keep"])
            elif kind == "RESHAPE":
                h = vals[s["x"]]
                y = h.reshape(h.shape[0], -1, 1, 1)
            elif kind == "PAD":
                y = F.pad(vals[s["x"]], s["pad"])
            elif kind == "FULLY_CONNECTED":
                h = vals[s["x"]]
                emb = h
                if self.training and self.head_dropout > 0:
                    h = F.dropout(h, self.head_dropout)
                y = F.linear(h, self.weight(s["w"]), self.weight(s["b"]))
            y = self._activate(y, s.get("act", 0))
            vals[s["out"]] = y
            if stop is not None and s["out"] == stop:
                return y.permute(0, 2, 3, 1)
        logits = vals[self.out_tensor]
        return (logits, emb) if return_embedding else logits

    # ------------------------------------------------------------------ #
    def groups(self) -> dict[str, list[str]]:
        """Nombres de parametros por bloque (para tasas de aprendizaje discriminativas)."""
        out: dict[str, list[str]] = {g: [] for g in GROUP_ORDER}
        for key in self.params.keys():
            if key.startswith("BLOCK_3"):
                out["block3"].append(key)
            elif key.startswith("BLOCK_4"):
                out[f"b4{key[8]}"].append(key)                # BLOCK_4_<n>_... -> b41 ... b44
            elif key.startswith("CLASS_DENSE") or key.startswith("Identity"):
                out["classifier"].append(key)
            else:
                out["post"].append(key)
        return out

    def num_parameters(self) -> dict[str, int]:
        return {g: int(sum(self.params[k].numel() for k in keys)) for g, keys in self.groups().items()}


def verify(cut: str = "f3", n: int = 8, seed: int = 0, audio: np.ndarray | None = None) -> dict[str, float]:
    """Compara la replica con el interprete TFLite sobre ``n`` ventanas."""
    net = T.MultiOutputBirdNET()
    if audio is None:
        rng = np.random.default_rng(seed)
        t = np.arange(144_000) / 48_000.0
        audio = np.stack([0.05 * rng.normal(size=144_000) + 0.1 * np.sin(2 * np.pi * rng.uniform(500, 8000) * t * (1 + 0.2 * t)) for _ in range(n)]).astype(np.float32)
    ref = net(audio)
    top = BirdNETTop(cut).eval()
    with torch.no_grad():
        logits, emb = top(torch.from_numpy(ref[cut]), return_embedding=True)
    return {
        "max_abs_logit_error": float((logits.numpy() - ref["logits"]).__abs__().max()),
        "max_abs_embedding_error": float((emb.numpy() - ref["emb"]).__abs__().max()),
        "logit_scale": float(np.abs(ref["logits"]).mean()),
        "embedding_scale": float(np.abs(ref["emb"]).mean()),
    }
