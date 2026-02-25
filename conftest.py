"""
conftest.py
-----------
Shared pytest fixtures for the Mini GACS test suite.
"""

import os
import sys
import tempfile
from typing import List
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Torch-free test infrastructure
# ---------------------------------------------------------------------------
# PyTorch is an optional 2 GB+ dependency.  All tests that exercise CLIP-
# dependent code mock out the model; their mock helpers construct fake tensors
# via `import torch; torch.zeros(...)`.  Instead of requiring torch to be
# installed we register a minimal numpy-backed shim in sys.modules so that
# every `import torch` inside a test function returns the shim rather than
# raising ModuleNotFoundError.
#
# The shim supports exactly the operations used in the test mocks:
#   torch.zeros / torch.ones / torch.randn / torch.tensor
#   MockTensor: .cpu() .numpy() .norm() .expand() / (truediv) [] .shape
# ---------------------------------------------------------------------------

class _MockTensor:
    """Numpy-backed tensor that satisfies the interface used in test mocks."""

    def __init__(self, data, dtype=None):
        arr = np.asarray(data)
        if dtype is not None:
            arr = arr.astype(np.dtype(dtype))
        elif arr.dtype.kind == "f":
            arr = arr.astype(np.float32)
        self._data = arr
        self.shape = arr.shape

    # ---- tensor-like methods called by production code -------------------

    def cpu(self):
        return self

    def detach(self):
        return self

    def numpy(self):
        return self._data

    def to(self, device):  # .to("cpu") / .to("cuda") → no-op for mock
        return self

    def norm(self, dim=-1, keepdim=False):
        norms = np.linalg.norm(self._data, axis=dim, keepdims=keepdim)
        return _MockTensor(norms.astype(np.float32))

    def expand(self, *shape):
        """Broadcast to *shape*, treating -1 as "keep existing size"."""
        new_shape = tuple(
            self._data.shape[i] if s == -1 else s
            for i, s in enumerate(shape)
        )
        return _MockTensor(np.broadcast_to(self._data, new_shape).copy())

    # ---- arithmetic -------------------------------------------------------

    def __truediv__(self, other):
        d = other._data if isinstance(other, _MockTensor) else other
        return _MockTensor(self._data / d)

    def __rtruediv__(self, other):
        return _MockTensor(other / self._data)

    # ---- item access ------------------------------------------------------

    def __getitem__(self, key):
        r = self._data[key]
        return _MockTensor(r) if isinstance(r, np.ndarray) else float(r)

    def __setitem__(self, key, value):
        self._data[key] = value

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"MockTensor(shape={self.shape}, dtype={self._data.dtype})"


class _NoGrad:
    """Null context manager replacing ``torch.no_grad()``."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _MockCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class _MockTorchModule:
    """Minimal stand-in for the ``torch`` module used by test mock helpers."""

    # dtype constants — values equal to the corresponding numpy dtype so that
    # `dtype=torch.long` passes cleanly to np.zeros / np.ones.
    long = np.int64
    float32 = np.float32

    # Type alias required by scipy's array_api_compat
    Tensor = _MockTensor

    # Sub-module stubs
    cuda = _MockCuda()

    @staticmethod
    def no_grad():
        return _NoGrad()

    @staticmethod
    def zeros(*shape, dtype=None):
        dt = np.dtype(dtype) if dtype is not None else np.float32
        return _MockTensor(np.zeros(shape, dtype=dt))

    @staticmethod
    def ones(*shape, dtype=None):
        dt = np.dtype(dtype) if dtype is not None else np.float32
        return _MockTensor(np.ones(shape, dtype=dt))

    @staticmethod
    def randn(*shape):
        return _MockTensor(np.random.randn(*shape).astype(np.float32))

    @staticmethod
    def tensor(data, dtype=None):
        arr = data._data if isinstance(data, _MockTensor) else np.asarray(data)
        dt = np.dtype(dtype) if dtype is not None else np.float32
        return _MockTensor(arr.astype(dt))


# Register the mock only when torch is not actually installed so that
# environments with a real torch installation are unaffected.
if "torch" not in sys.modules:
    sys.modules["torch"] = _MockTorchModule()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Synthetic video / image factories
# ---------------------------------------------------------------------------

def _write_video(path: str, n_frames: int = 30, fps: int = 10,
                 width: int = 64, height: int = 64) -> None:
    """Write a short synthetic colour video using OpenCV."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(n_frames):
        colour = (i * 8 % 256, (255 - i * 8) % 256, (i * 4) % 256)
        frame = np.full((height, width, 3), colour, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _write_image(path: str, r: int = 128, g: int = 64, b: int = 32) -> None:
    """Save a small solid-colour JPEG image."""
    img = Image.new("RGB", (32, 32), (r, g, b))
    img.save(path, "JPEG")


# ---------------------------------------------------------------------------
# Session-scoped temp directory
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tmp_session(tmp_path_factory):
    """A single temporary directory shared across the whole test session."""
    return tmp_path_factory.mktemp("session")


# ---------------------------------------------------------------------------
# Synthetic video fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_video(tmp_path):
    """Yield the path to a freshly written 3-second synthetic video."""
    path = str(tmp_path / "synth.mp4")
    _write_video(path, n_frames=30, fps=10)
    return path


@pytest.fixture()
def synthetic_video_dir(tmp_path):
    """Yield a directory containing 2 synthetic videos."""
    d = tmp_path / "videos"
    d.mkdir()
    for i in range(2):
        _write_video(str(d / f"video_{i}.mp4"), n_frames=20, fps=10)
    return str(d)


# ---------------------------------------------------------------------------
# Synthetic frame images fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_frames(tmp_path):
    """Yield a list of 6 JPEG image paths with varying solid colours."""
    paths = []
    for i in range(6):
        p = str(tmp_path / f"frame_{i:04d}.jpg")
        _write_image(p, r=i * 40 % 256, g=i * 20 % 256, b=i * 10 % 256)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# L2-normalised random embeddings fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def unit_embeddings():
    """Return a factory that produces L2-normalised random embeddings."""
    def _make(n: int, dim: int = 8, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        raw = rng.standard_normal((n, dim)).astype(np.float32)
        return raw / np.linalg.norm(raw, axis=1, keepdims=True)
    return _make


# ---------------------------------------------------------------------------
# Mock CLIP model factory fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_clip_factory():
    """
    Return a factory that builds (MockModel, MockProcessor) pairs.

    The processor respects the actual batch size of the input so tests that
    examine embedding shapes work correctly.
    """
    def _make(embed_dim: int = 16, fixed_output: bool = False):
        import torch

        mock_model = MagicMock()
        if fixed_output:
            # Always return the same unit vector – determinism test
            fixed = torch.zeros(1, embed_dim)
            fixed[0, 0] = 1.0

            def _get_features(**kwargs):
                n = kwargs["pixel_values"].shape[0]
                return fixed.expand(n, -1)
        else:
            def _get_features(**kwargs):
                n = kwargs["pixel_values"].shape[0]
                feats = torch.randn(n, embed_dim)
                return feats / feats.norm(dim=-1, keepdim=True)

        mock_model.get_image_features.side_effect = _get_features
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        # Text features mirror image features
        def _get_text_features(**kwargs):
            n = kwargs.get("input_ids", torch.zeros(1)).shape[0]
            feats = torch.randn(n, embed_dim)
            return feats / feats.norm(dim=-1, keepdim=True)

        mock_model.get_text_features.side_effect = _get_text_features

        mock_processor = MagicMock()

        def _proc_call(*args, images=None, text=None,
                       return_tensors=None, padding=None,
                       truncation=None, **kwargs):
            batch = images if images is not None else (text or [])
            n = len(batch)
            return {"pixel_values": torch.zeros(n, 3, 224, 224),
                    "input_ids": torch.zeros(n, 77, dtype=torch.long)}

        mock_processor.side_effect = _proc_call

        return mock_model, mock_processor

    return _make
