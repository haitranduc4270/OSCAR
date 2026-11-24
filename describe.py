"""Print the layout of a nested value without dumping its numbers."""

from __future__ import annotations

import numpy as np
import torch


def describe(x) -> None:
    print()
    print(schema(x))


def schema(x) -> str:
    if isinstance(x, dict):
        if not x:
            return "dict (0)"
        sigs = [schema(v) for v in x.values()]
        if len(set(sigs)) == 1:
            keys = ", ".join(repr(k) for k in x)
            return f"dict ({len(x)}) -> {sigs[0]}\n  keys: {keys}"
        lines = [f"dict ({len(x)})"]
        for k, v in x.items():
            lines.append(f"  {k!r}:")
            lines.append(_indent(schema(v), 2))
        return "\n".join(lines)

    if isinstance(x, (list, tuple)):
        name = type(x).__name__
        if not x:
            return f"{name} (0)"
        sigs = [schema(v) for v in x]
        if len(set(sigs)) == 1:
            return f"{name} ({len(x)}) -> {sigs[0]}"
        lines = [f"{name} ({len(x)})"]
        for i, v in enumerate(x):
            lines.append(f"  [{i}]:")
            lines.append(_indent(schema(v), 2))
        return "\n".join(lines)

    if isinstance(x, torch.Tensor):
        return f"Tensor shape={tuple(x.shape)} dtype={x.dtype}"
    if isinstance(x, np.ndarray):
        return f"ndarray shape={x.shape} dtype={x.dtype}"
    if x is None or isinstance(x, (bool, int, float, str)):
        return type(x).__name__
    return type(x).__name__


def _indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())
