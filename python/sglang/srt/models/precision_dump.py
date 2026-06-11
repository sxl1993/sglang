"""Static precision-diff dump bypass (sglang side). Activated by MILES_DSV4_PRECISION_DUMP=1.

Self-contained copy of the train-side util so the sglang process needs no
miles_plugins dependency. Writes one .pt per (side, layer, boundary, rank) with
the tensor and its global token positions, for offline alignment with the train side.

Only the prefill (extend) forward is dumped: it processes the whole prompt in a
single forward pass, so each shard already carries the full sequence and can be
written immediately. Decode steps are skipped -- they emit one token at a time and
would need cross-step accumulation that cannot survive the inference process being
killed by ray at rollout end. Use dump_prefill() at each dump site to gate on this.
"""

import os
from pathlib import Path

import torch

_ENABLED = None
_ROOT = None

# Mirror of sglang.srt.constants.HEALTH_CHECK_RID_PREFIX (kept inline to stay
# self-contained); /health_generate tags its probe rid with this prefix.
HEALTH_CHECK_RID_PREFIX = "HEALTH_CHECK"


def dump_enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.environ.get("MILES_DSV4_PRECISION_DUMP", "0") == "1"
    return _ENABLED


def dump_prefill(forward_batch) -> bool:
    """True only when dumping is on AND this is a prefill (extend) forward pass."""
    return dump_enabled() and forward_batch.forward_mode.is_extend()


def dump_decode(forward_batch) -> bool:
    """True only when dumping is on AND this is a real decode forward pass.

    Decode emits one token per step, so each boundary tensor is accumulated into
    a CPU buffer via append_decode() and flushed once per request (flush_decode).

    Skips /health_generate probes (rid prefixed HEALTH_CHECK): they send
    input_ids=[0], max_new_tokens=1, so their single decode step lands at pos=1
    and would pollute the dump (aligns onto train prompt pos=1 offline).
    """
    if not (dump_enabled() and forward_batch.forward_mode.is_decode()):
        return False
    rids = getattr(forward_batch, "rids", None)
    if rids and any(isinstance(r, str) and r.startswith(HEALTH_CHECK_RID_PREFIX) for r in rids):
        return False
    return True


# (layer, boundary) -> {"rows": list[Tensor on CPU], "pos": list[int]}.
# Single-request assumption (rollout-batch-size 1, n-samples 1): no request-id key.
_DECODE_BUF: dict = {}


def append_decode(side: str, layer: int, boundary: str, tensor, global_pos=None):
    """Append one decode step's boundary tensor to the CPU buffer. No-op unless enabled.

    tensor is moved to CPU immediately: colocate releases GPU memory after rollout
    (release_memory_occupation), which would drop any buffer left on GPU.
    global_pos: the scalar global token position of this decode step (prompt_len +
        step_idx), aligned to the train side's global_token_indices coordinate system.
    """
    if not dump_enabled():
        return
    # rank0 alone represents pure-TP replicas; see dump_tensor for rationale.
    if _rank() != 0:
        return
    key = (layer, boundary)
    entry = _DECODE_BUF.setdefault(key, {"rows": [], "pos": [], "side": side})
    # tensor is (1, ...) for a single decode token -> squeeze the token axis.
    row = tensor.detach().to(torch.float32).cpu()
    if row.shape[0] == 1:
        row = row[0]
    entry["rows"].append(row)
    if global_pos is not None:
        gp = global_pos.detach().cpu().reshape(-1)
        entry["pos"].append(int(gp[-1]))


def flush_decode(side: str):
    """Write accumulated decode buffer to one .pt per (layer, boundary). No-op if empty.

    Each file holds tensor (n_steps, ...) and global_pos (n_steps,), mirroring the
    prefill dump payload but with a decode_ filename prefix.
    """
    if not dump_enabled() or not _DECODE_BUF:
        return
    out_dir = _root() / side
    out_dir.mkdir(parents=True, exist_ok=True)
    rank = _rank()
    for (layer, boundary), entry in _DECODE_BUF.items():
        rows = entry["rows"]
        if not rows:
            continue
        tensor = torch.stack(rows, dim=0)
        pos = entry["pos"]
        global_pos = torch.tensor(pos, dtype=torch.long) if len(pos) == len(rows) else None
        payload = {
            "tensor": tensor,
            "global_pos": global_pos,
            "side": side,
            "layer": layer,
            "boundary": boundary,
            "rank": rank,
        }
        path = out_dir / f"decode_{layer}_{boundary}_rank{rank}.pt"
        torch.save(payload, path)
    _DECODE_BUF.clear()


def _root() -> Path:
    global _ROOT
    if _ROOT is None:
        base = os.environ.get(
            "MILES_DSV4_PRECISION_DUMP_DIR",
            "/mnt/amed-s1/common/ckpt/muchen/precision_dump",
        )
        _ROOT = Path(base)
    return _ROOT


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def dump_tensor(side: str, layer: int, boundary: str, tensor, global_pos=None):
    """Write one dump shard. No-op unless MILES_DSV4_PRECISION_DUMP=1.

    side: "rollout" for the sglang inference side.
    layer: layer index (0-based).
    boundary: semantic name of the dump point (e.g. "hidden_in", "kv_proj",
        "fuse_out", "ape_raw"). Used verbatim in the filename.
    tensor: the intermediate tensor to compare (detached, moved to CPU).
    global_pos: 1-D LongTensor of global token positions aligned to tensor's
        token axis, or None if positional alignment is not applicable.
    """
    if not dump_enabled():
        return
    # rollout is pure TP=8 with DP attention off -> all ranks hold bit-exact
    # replicas, so rank0 alone represents the full data. Gating here avoids 8-way
    # concurrent torch.save to alluxio-fuse, which intermittently fails with EIO.
    if _rank() != 0:
        return
    out_dir = _root() / side
    out_dir.mkdir(parents=True, exist_ok=True)
    rank = _rank()
    payload = {
        "tensor": tensor.detach().to(torch.float32).cpu(),
        "global_pos": (global_pos.detach().cpu() if global_pos is not None else None),
        "side": side,
        "layer": layer,
        "boundary": boundary,
        "rank": rank,
    }
    path = out_dir / f"prefill_{layer}_{boundary}_rank{rank}.pt"
    torch.save(payload, path)
