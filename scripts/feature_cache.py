"""
Versioned on-disk cache for precomputed encoder features (e.g. CLIP embeddings).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from common_multilabel import write_json

# ViT patch caches are large; fp16 halves disk vs float32.
PATCH_CACHE_VERSION = "patch_v2_fp16"


def _hash_row_ids(row_ids: List[str]) -> str:
    payload = "\n".join(row_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_nbytes(tensor: Any, dtype_name: str = "float32") -> int:
    import torch

    if not hasattr(tensor, "numel"):
        return 0
    if dtype_name == "float16":
        return int(tensor.numel()) * 2
    if dtype_name == "float32":
        return int(tensor.numel()) * 4
    return int(tensor.numel()) * tensor.element_size()


def _object_nbytes(obj: Any) -> int:
    import torch

    if isinstance(obj, torch.Tensor):
        return int(obj.numel()) * obj.element_size()
    if isinstance(obj, dict):
        return sum(_object_nbytes(v) for v in obj.values())
    return 0


def _disk_usage_target(path: Path) -> Path:
    """Directory on the target volume (works for relative paths on Windows)."""
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved.parent


def _check_disk_space(path: Path, nbytes: int, label: str) -> None:
    """Raise with a clear message if the target volume lacks free space."""
    check_at = _disk_usage_target(path)
    usage = shutil.disk_usage(str(check_at))
    free = usage.free
    # Headroom for zip container overhead and manifest sidecars.
    required = int(nbytes * 1.05) + 256 * 1024 * 1024
    if free < required:
        free_gb = free / (1024**3)
        need_gb = required / (1024**3)
        volume = check_at.drive or str(check_at.anchor) or str(check_at)
        raise OSError(
            f"Not enough disk space to write {label} at {path} "
            f"(need ~{need_gb:.1f} GB free, have {free_gb:.1f} GB on {volume}). "
            f"Free space or set --embeddings_cache_dir to a drive with room."
        )


def atomic_torch_save(obj: Any, path: Path, storage_dtype: Optional[str] = None) -> Any:
    """Save a tensor or picklable object atomically."""
    import torch

    to_store = obj
    if isinstance(obj, torch.Tensor):
        if storage_dtype == "float16":
            to_store = obj.detach().cpu().to(torch.float16)
        elif storage_dtype == "float32":
            to_store = obj.detach().cpu().float()
        nbytes = _tensor_nbytes(to_store, storage_dtype or "float32")
    else:
        nbytes = _object_nbytes(to_store)

    _check_disk_space(path, nbytes, path.name)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        torch.save(to_store, tmp)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        if path.exists():
            path.unlink()
        raise
    return to_store


class FeatureCache:
    """
    Stores tensors under ``{cache_dir}/{dataset_id}_{encoder_id}_{version}.pt``
    with a manifest recording row-order hashes for integrity checks.
    """

    def __init__(self, cache_dir: Union[str, Path] = "data/processed/embeddings"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.cache_dir / "manifest.json"
        self._manifest: Dict[str, Any] = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        with self.manifest_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save_manifest(self) -> None:
        write_json(self.manifest_path, self._manifest)

    def cache_key(self, dataset_id: str, encoder_id: str, version: str) -> str:
        return f"{dataset_id}_{encoder_id}_{version}"

    def cache_path(self, dataset_id: str, encoder_id: str, version: str) -> Path:
        key = self.cache_key(dataset_id, encoder_id, version)
        return self.cache_dir / f"{key}.pt"

    def get_or_compute(
        self,
        dataset_id: str,
        encoder_id: str,
        version: str,
        row_ids: List[str],
        compute_fn: Callable[[], Any],
        storage_dtype: Optional[str] = None,
    ) -> Any:
        """
        Return cached tensor if present and row order matches; else compute and store.
        """
        import torch

        key = self.cache_key(dataset_id, encoder_id, version)
        path = self.cache_path(dataset_id, encoder_id, version)
        row_hash = _hash_row_ids(row_ids)

        if key in self._manifest:
            entry = self._manifest[key]
            if entry.get("row_id_hash") != row_hash:
                raise ValueError(
                    f"Feature cache row order mismatch for {key}: "
                    f"expected hash {entry.get('row_id_hash')}, got {row_hash}. "
                    "Delete cache file and recompute."
                )
            if path.exists():
                return torch.load(path, map_location="cpu")

        if path.exists():
            stored = torch.load(path, map_location="cpu")
            meta_path = path.with_suffix(".meta.json")
            if meta_path.exists():
                with meta_path.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("row_id_hash") != row_hash:
                    raise ValueError(
                        f"Feature cache on disk has stale row order for {path}; delete and recompute."
                    )
            return stored

        tensor = compute_fn()
        stored = atomic_torch_save(tensor, path, storage_dtype=storage_dtype)
        shape = list(stored.shape) if hasattr(stored, "shape") else None
        meta = {
            "file": str(path),
            "dataset_id": dataset_id,
            "encoder_id": encoder_id,
            "version": version,
            "row_id_hash": row_hash,
            "shape": shape,
            "storage_dtype": storage_dtype,
        }
        write_json(path.with_suffix(".meta.json"), meta)
        self._manifest[key] = meta
        self._save_manifest()
        return stored

    def invalidate(self, dataset_id: str, encoder_id: str, version: str) -> None:
        key = self.cache_key(dataset_id, encoder_id, version)
        path = self.cache_path(dataset_id, encoder_id, version)
        if path.exists():
            path.unlink()
        meta = path.with_suffix(".meta.json")
        if meta.exists():
            meta.unlink()
        self._manifest.pop(key, None)
        self._save_manifest()


def clip_cache_dataset_id(protocol: str) -> str:
    """Map protocol name to dataset_id for CLIP caches."""
    if protocol == "calibrated4way":
        return "chexpert_calibrated4way"
    return "chexpert_default"
