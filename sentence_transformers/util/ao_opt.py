
from __future__ import annotations

import os

__all__ = ["flag", "opt_13", "opt_16", "apply_backend_flags", "DEFAULTS"]

DEFAULTS = {
    # On by default: numerically transparent, measured 2.0x on their own.
    "ST_OPT_3": True,
    "ST_OPT_2": True,
    "ST_OPT_1": True,
    "ST_OPT_4": True,
    "ST_OPT_7": False,
    "ST_OPT_8": False,
}

OPT_17 = 2

OPT_18: str | None = "bf16"

#: encode() runs at this batch size when the caller leaves `batch_size` at its
#: documented default of 32. An explicitly different value is always honoured.
DEFAULT_BATCH_SIZE = 384
STOCK_BATCH_SIZE = 32


def flag(name: str, default: bool | None = None) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return DEFAULTS.get(name, False) if default is None else default
    return raw.strip().lower() not in ("", "0", "false", "no", "off")


def opt_13() -> str | None:
    raw = os.environ.get("ST_OPT_6")
    if raw is None:
        return OPT_18
    raw = raw.strip().lower()
    if raw in ("bf16", "bfloat16"):
        return "bf16"
    if raw in ("fp16", "float16", "half"):
        return "fp16"
    return None


def apply_backend_flags() -> None:
    import torch

    if flag("ST_OPT_7"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def batch_size_override() -> int:
    try:
        return max(0, int(os.environ.get("ST_BATCH_SIZE") or 0))
    except ValueError:
        return 0


def opt_14() -> int:
    raw = os.environ.get("ST_OPT_5")
    if raw is None:
        return OPT_17
    try:
        return max(0, int(raw))
    except ValueError:
        return OPT_17


def resolved_batch_size(requested: int) -> int:
    """The batch size encode() should actually run at.

    `ST_BATCH_SIZE` overrides everything. Otherwise a caller who left
    `batch_size` at its documented default of 32 gets `DEFAULT_BATCH_SIZE`
    (`ST_DEFAULT_BATCH_SIZE` to change it, `=32` to restore the old default);
    a caller who asked for any other value keeps exactly what they asked for,
    because that is usually a memory decision.
    """
    override = batch_size_override()
    if override:
        return override
    raw = os.environ.get("ST_DEFAULT_BATCH_SIZE")
    try:
        default = DEFAULT_BATCH_SIZE if raw is None else int(raw)
    except ValueError:
        default = DEFAULT_BATCH_SIZE
    if requested == STOCK_BATCH_SIZE and default > 0:
        return default
    return requested


def opt_15(preprocess, inputs_sorted, batch_size, depth):
    import queue
    import threading

    out: "queue.Queue" = queue.Queue(maxsize=max(1, depth))
    done = object()

    def _worker():
        try:
            for start in range(0, len(inputs_sorted), batch_size):
                out.put(preprocess(inputs_sorted[start : start + batch_size]))
        except BaseException as exc:                 # surfaced on the consumer side
            out.put(exc)
        else:
            out.put(done)

    thread = threading.Thread(target=_worker, daemon=True, name="ao-preprocess")
    thread.start()

    def _opt_19():
        while True:
            item = out.get()
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    return _opt_19()


_OPT_20: set[int] = set()


def opt_16(model) -> str | None:
    import torch
    from torch import nn

    if not flag("ST_OPT_8"):
        return None

    best = None
    for module in model.modules():
        if not isinstance(module, nn.ModuleList) or len(module) < 2:
            continue
        if len({type(m) for m in module}) != 1:
            continue
        params = sum(p.numel() for p in module[0].parameters())
        if best is None or params > best[1]:
            best = (type(module[0]), params, len(module))
    if best is None:
        return None

    cls, _params, n = best
    if id(cls) not in _OPT_20:
        cls.forward = torch.compile(cls.forward, dynamic=True)
        _OPT_20.add(id(cls))
    return f"{cls.__name__} x{n}"
