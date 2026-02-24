"""
conftest.py
-----------
Shared pytest fixtures for the Mini GACS test suite.
"""

import os
import tempfile
from typing import List
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from PIL import Image


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
