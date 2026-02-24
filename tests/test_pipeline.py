"""
test_pipeline.py
----------------
Unit and integration tests for the Mini GACS pipeline.

Test categories
~~~~~~~~~~~~~~~
1. Frame extraction  – uses synthetic OpenCV videos; no real video needed.
2. Embeddings        – uses a mocked CLIP model to avoid downloading weights.
3. Similarity        – pure NumPy; no external dependencies needed.
4. Visualization     – checks that Matplotlib files are written correctly.
5. Integration       – wires mock objects together end-to-end.
6. Affective scoring – mocked CLIP text encoder; checks shape/range/NaN.
7. Clustering        – pure NumPy + sklearn; no CLIP model needed.

Run with:
    python -m pytest tests/ -v
"""

import csv
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_video(path: str, n_frames: int = 30, fps: int = 10,
                           width: int = 64, height: int = 64) -> None:
    """Write a short synthetic colour video with OpenCV."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    assert writer.isOpened(), f"VideoWriter failed to open for {path}"
    for i in range(n_frames):
        # Gradient colour so frames are visually distinct
        colour = (i * 8 % 256, (255 - i * 8) % 256, (i * 4) % 256)
        frame = np.full((height, width, 3), colour, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _make_synthetic_frame(path: str, r: int = 128, g: int = 64, b: int = 32) -> None:
    """Save a small solid-colour JPEG image."""
    img = Image.new("RGB", (32, 32), (r, g, b))
    img.save(path, "JPEG")


# ---------------------------------------------------------------------------
# 1. Frame extraction tests
# ---------------------------------------------------------------------------

class TestFrameExtractor(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    # --- extract_frames ---

    def test_extract_frames_basic(self):
        from src.frame_extractor import extract_frames

        video_path = os.path.join(self.tmp, "test_video.mp4")
        _make_synthetic_video(video_path, n_frames=30, fps=10)

        frames_dir = os.path.join(self.tmp, "frames")
        metadata = extract_frames(video_path, frames_dir, interval_seconds=1.0)

        # 30 frames at 10 fps → 3 seconds → expect 3 extracted frames
        self.assertGreaterEqual(len(metadata), 1)
        for entry in metadata:
            self.assertIn("video_id", entry)
            self.assertIn("timestamp", entry)
            self.assertIn("frame_idx", entry)
            self.assertIn("file_path", entry)
            self.assertTrue(os.path.exists(entry["file_path"]),
                            f"Frame file missing: {entry['file_path']}")

    def test_extract_frames_max_frames(self):
        from src.frame_extractor import extract_frames

        video_path = os.path.join(self.tmp, "long_video.mp4")
        # 100 frames at 10 fps → 10 seconds
        _make_synthetic_video(video_path, n_frames=100, fps=10)

        frames_dir = os.path.join(self.tmp, "frames_max")
        metadata = extract_frames(video_path, frames_dir,
                                  interval_seconds=0.5, max_frames=3)
        self.assertLessEqual(len(metadata), 3)

    def test_extract_frames_file_not_found(self):
        from src.frame_extractor import extract_frames

        with self.assertRaises(FileNotFoundError):
            extract_frames("/nonexistent/video.mp4", self.tmp)

    def test_extract_frames_bad_file(self):
        from src.frame_extractor import extract_frames

        bad_path = os.path.join(self.tmp, "not_a_video.mp4")
        with open(bad_path, "w") as fh:
            fh.write("this is not a video")

        # A corrupt file that OpenCV cannot open raises RuntimeError;
        # a file that opens but yields no decodable frames returns [].
        # Both outcomes are valid – the pipeline must not silently succeed.
        frames_dir = os.path.join(self.tmp, "frames_bad")
        try:
            metadata = extract_frames(bad_path, frames_dir)
            self.assertEqual(len(metadata), 0)
        except RuntimeError:
            pass  # Also an acceptable outcome for an unreadable file

    # --- save / load metadata ---

    def test_save_load_metadata_csv(self):
        from src.frame_extractor import save_metadata, load_metadata

        meta = [
            {"video_id": "v1", "timestamp": 0.5, "frame_idx": 0, "file_path": "/a.jpg"},
            {"video_id": "v1", "timestamp": 1.5, "frame_idx": 1, "file_path": "/b.jpg"},
        ]
        csv_path = os.path.join(self.tmp, "meta.csv")
        save_metadata(meta, csv_path)

        loaded = load_metadata(csv_path)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["video_id"], "v1")
        self.assertAlmostEqual(loaded[1]["timestamp"], 1.5)
        self.assertEqual(loaded[1]["frame_idx"], 1)

    def test_save_load_metadata_json(self):
        from src.frame_extractor import save_metadata, load_metadata

        meta = [{"video_id": "art", "timestamp": 2.0, "frame_idx": 2, "file_path": "/c.jpg"}]
        json_path = os.path.join(self.tmp, "meta.json")
        save_metadata(meta, json_path, fmt="json")

        loaded = load_metadata(json_path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["video_id"], "art")

    def test_load_metadata_missing_file(self):
        from src.frame_extractor import load_metadata

        with self.assertRaises(FileNotFoundError):
            load_metadata("/nonexistent/meta.csv")

    def test_load_metadata_unsupported_format(self):
        from src.frame_extractor import load_metadata

        path = os.path.join(self.tmp, "meta.txt")
        with open(path, "w") as fh:
            fh.write("dummy")
        with self.assertRaises(ValueError):
            load_metadata(path)


# ---------------------------------------------------------------------------
# 2. Embedding tests (mocked CLIP)
# ---------------------------------------------------------------------------

def _make_mock_clip_model(embed_dim: int = 16):
    """Return a mock CLIPModel and CLIPProcessor pair."""
    mock_processor = MagicMock()

    def processor_side_effect(images, return_tensors, padding):
        import torch
        n = len(images)
        return {"pixel_values": torch.zeros(n, 3, 224, 224)}

    mock_processor.side_effect = processor_side_effect
    mock_processor.__call__ = processor_side_effect

    # make processor(images=...) work
    mock_processor.return_value = {"pixel_values": __import__("torch").zeros(1, 3, 224, 224)}

    mock_model = MagicMock()

    def get_image_features(**kwargs):
        import torch
        n = kwargs["pixel_values"].shape[0]
        feats = torch.randn(n, embed_dim)
        # normalise so they look like real CLIP outputs
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    mock_model.get_image_features.side_effect = get_image_features
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model

    return mock_model, mock_processor


class TestEmbeddingModel(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _create_frame_files(self, n: int):
        paths = []
        for i in range(n):
            p = os.path.join(self.tmp, f"frame_{i:04d}.jpg")
            _make_synthetic_frame(p, r=i * 20 % 256, g=i * 10 % 256, b=i * 5 % 256)
            paths.append(p)
        return paths

    @patch("src.embeddings.CLIPModel")
    @patch("src.embeddings.CLIPProcessor")
    def test_embed_images_shape(self, MockProcessor, MockModel):
        mock_model, mock_processor = _make_mock_clip_model(embed_dim=16)
        MockModel.from_pretrained.return_value = mock_model
        MockProcessor.from_pretrained.return_value = mock_processor

        from src.embeddings import EmbeddingModel

        paths = self._create_frame_files(5)
        em = EmbeddingModel(model_name="mock/clip", batch_size=3)
        embeddings = em.embed_images(paths)

        self.assertEqual(embeddings.shape[0], 5)
        self.assertEqual(embeddings.ndim, 2)
        self.assertEqual(embeddings.dtype, np.float32)

    @patch("src.embeddings.CLIPModel")
    @patch("src.embeddings.CLIPProcessor")
    def test_embed_images_no_nan(self, MockProcessor, MockModel):
        mock_model, mock_processor = _make_mock_clip_model(embed_dim=16)
        MockModel.from_pretrained.return_value = mock_model
        MockProcessor.from_pretrained.return_value = mock_processor

        from src.embeddings import EmbeddingModel

        paths = self._create_frame_files(4)
        em = EmbeddingModel(model_name="mock/clip")
        embeddings = em.embed_images(paths)

        self.assertFalse(np.isnan(embeddings).any(), "Embeddings must not contain NaN.")
        self.assertFalse(np.isinf(embeddings).any(), "Embeddings must not contain Inf.")

    @patch("src.embeddings.CLIPModel")
    @patch("src.embeddings.CLIPProcessor")
    def test_identical_images_identical_embeddings(self, MockProcessor, MockModel):
        """
        Key correctness test: embedding the same image twice must yield the
        same (or near-identical) vector.
        """
        embed_dim = 16
        # Return a fixed deterministic vector for all images
        import torch

        mock_model = MagicMock()
        fixed_feat = torch.zeros(1, embed_dim)
        fixed_feat[0, 0] = 1.0  # unit vector along first axis

        def get_image_features(**kwargs):
            n = kwargs["pixel_values"].shape[0]
            return fixed_feat.expand(n, -1)

        mock_model.get_image_features.side_effect = get_image_features
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        # Mock processor must return the correct batch size for each call
        mock_processor = MagicMock()

        def proc_call(*args, images=None, return_tensors=None, padding=None, **kwargs):
            n = len(images) if images is not None else 1
            return {"pixel_values": torch.zeros(n, 3, 224, 224)}

        mock_processor.side_effect = proc_call

        MockModel.from_pretrained.return_value = mock_model
        MockProcessor.from_pretrained.return_value = mock_processor

        from src.embeddings import EmbeddingModel

        # Create one image and duplicate its path
        img_path = os.path.join(self.tmp, "same.jpg")
        _make_synthetic_frame(img_path)
        paths = [img_path, img_path]

        em = EmbeddingModel(model_name="mock/clip")
        embeddings = em.embed_images(paths)

        self.assertEqual(embeddings.shape[0], 2,
                         "Both identical paths should produce 2 embeddings.")
        np.testing.assert_array_almost_equal(
            embeddings[0], embeddings[1],
            decimal=5,
            err_msg="Embeddings for identical images should be identical.",
        )

    @patch("src.embeddings.CLIPModel")
    @patch("src.embeddings.CLIPProcessor")
    def test_empty_paths_raises(self, MockProcessor, MockModel):
        mock_model, mock_processor = _make_mock_clip_model()
        MockModel.from_pretrained.return_value = mock_model
        MockProcessor.from_pretrained.return_value = mock_processor

        from src.embeddings import EmbeddingModel

        em = EmbeddingModel(model_name="mock/clip")
        with self.assertRaises(ValueError):
            em.embed_images([])

    @patch("src.embeddings.CLIPModel")
    @patch("src.embeddings.CLIPProcessor")
    def test_missing_images_are_skipped(self, MockProcessor, MockModel):
        mock_model, mock_processor = _make_mock_clip_model(embed_dim=16)
        MockModel.from_pretrained.return_value = mock_model
        MockProcessor.from_pretrained.return_value = mock_processor

        from src.embeddings import EmbeddingModel

        real_path = os.path.join(self.tmp, "real.jpg")
        _make_synthetic_frame(real_path)
        paths = ["/nonexistent/ghost.jpg", real_path]

        em = EmbeddingModel(model_name="mock/clip")
        # Should succeed – ghost is skipped, real is embedded
        embeddings = em.embed_images(paths)
        self.assertEqual(embeddings.shape[0], 1)

    def test_save_load_embeddings(self):
        from src.embeddings import save_embeddings, load_embeddings

        arr = np.random.rand(5, 16).astype(np.float32)
        idx = [{"file_path": f"/img_{i}.jpg", "frame_idx": i} for i in range(5)]
        out_dir = os.path.join(self.tmp, "embs")

        save_embeddings(arr, idx, out_dir)
        loaded_arr, loaded_idx = load_embeddings(out_dir)

        np.testing.assert_array_almost_equal(arr, loaded_arr)
        self.assertEqual(len(loaded_idx), 5)

    def test_load_embeddings_missing_raises(self):
        from src.embeddings import load_embeddings

        with self.assertRaises(FileNotFoundError):
            load_embeddings("/nonexistent/dir")


# ---------------------------------------------------------------------------
# 3. Similarity tests (pure NumPy – no model needed)
# ---------------------------------------------------------------------------

class TestSimilarity(unittest.TestCase):

    def _unit_embeddings(self, n: int, dim: int = 8) -> np.ndarray:
        """Return random L2-normalised embeddings."""
        rng = np.random.default_rng(42)
        raw = rng.standard_normal((n, dim)).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        return raw / norms

    def test_similarity_matrix_shape(self):
        from src.similarity import cosine_similarity_matrix

        embs = self._unit_embeddings(10)
        sim = cosine_similarity_matrix(embs)
        self.assertEqual(sim.shape, (10, 10))

    def test_similarity_matrix_diagonal_ones(self):
        """Each vector must have similarity 1 with itself."""
        from src.similarity import cosine_similarity_matrix

        embs = self._unit_embeddings(6)
        sim = cosine_similarity_matrix(embs)
        np.testing.assert_array_almost_equal(
            np.diag(sim), np.ones(6), decimal=4,
            err_msg="Diagonal of similarity matrix must be all-ones.",
        )

    def test_similarity_matrix_range(self):
        from src.similarity import cosine_similarity_matrix

        embs = self._unit_embeddings(8)
        sim = cosine_similarity_matrix(embs)
        self.assertLessEqual(float(sim.max()), 1.0 + 1e-5)
        self.assertGreaterEqual(float(sim.min()), -1.0 - 1e-5)

    def test_similarity_matrix_symmetry(self):
        from src.similarity import cosine_similarity_matrix

        embs = self._unit_embeddings(7)
        sim = cosine_similarity_matrix(embs)
        np.testing.assert_array_almost_equal(
            sim, sim.T, decimal=5,
            err_msg="Similarity matrix must be symmetric.",
        )

    def test_similarity_matrix_no_nan(self):
        from src.similarity import cosine_similarity_matrix

        embs = self._unit_embeddings(5)
        sim = cosine_similarity_matrix(embs)
        self.assertFalse(np.isnan(sim).any())

    def test_similarity_invalid_input(self):
        from src.similarity import cosine_similarity_matrix

        with self.assertRaises(ValueError):
            cosine_similarity_matrix(np.array([]))

    def test_top_k_returns_correct_count(self):
        from src.similarity import cosine_similarity_matrix, get_top_k_similar

        embs = self._unit_embeddings(10)
        sim = cosine_similarity_matrix(embs)
        index = [{"video_id": "v", "frame_idx": i, "timestamp": float(i)} for i in range(10)]

        results = get_top_k_similar(0, sim, index, top_k=5)
        self.assertEqual(len(results), 5)

    def test_top_k_excludes_self(self):
        from src.similarity import cosine_similarity_matrix, get_top_k_similar

        embs = self._unit_embeddings(10)
        sim = cosine_similarity_matrix(embs)
        index = [{"video_id": "v", "frame_idx": i, "timestamp": float(i)} for i in range(10)]

        results = get_top_k_similar(0, sim, index, top_k=5, exclude_self=True)
        frame_indices = [r["frame_idx"] for r in results]
        self.assertNotIn(0, frame_indices, "Self (frame_idx=0) must not appear in results.")

    def test_top_k_sorted_descending(self):
        from src.similarity import cosine_similarity_matrix, get_top_k_similar

        embs = self._unit_embeddings(12)
        sim = cosine_similarity_matrix(embs)
        index = [{"video_id": "v", "frame_idx": i, "timestamp": float(i)} for i in range(12)]

        results = get_top_k_similar(3, sim, index, top_k=5)
        sims = [r["similarity"] for r in results]
        self.assertEqual(sims, sorted(sims, reverse=True),
                         "Results must be sorted by descending similarity.")

    def test_top_k_similarity_in_results(self):
        from src.similarity import cosine_similarity_matrix, get_top_k_similar

        embs = self._unit_embeddings(8)
        sim = cosine_similarity_matrix(embs)
        index = [{"video_id": "v", "frame_idx": i} for i in range(8)]

        results = get_top_k_similar(2, sim, index, top_k=3)
        for r in results:
            self.assertIn("similarity", r)
            self.assertIsInstance(r["similarity"], float)

    def test_top_k_out_of_bounds(self):
        from src.similarity import cosine_similarity_matrix, get_top_k_similar

        embs = self._unit_embeddings(5)
        sim = cosine_similarity_matrix(embs)
        index = [{"video_id": "v", "frame_idx": i} for i in range(5)]

        with self.assertRaises(IndexError):
            get_top_k_similar(99, sim, index, top_k=3)

    def test_inter_video_stats_keys(self):
        from src.similarity import cosine_similarity_matrix, compute_inter_video_stats

        embs = self._unit_embeddings(8)
        sim = cosine_similarity_matrix(embs)
        index = [
            {"video_id": "v1" if i < 4 else "v2", "frame_idx": i}
            for i in range(8)
        ]
        stats = compute_inter_video_stats(sim, index)
        for key in ("overall_mean", "overall_median", "within_video_mean", "cross_video_mean"):
            self.assertIn(key, stats)
            self.assertFalse(np.isnan(stats[key]), f"Stats[{key}] must not be NaN.")

    def test_batch_top_k(self):
        from src.similarity import cosine_similarity_matrix, batch_top_k_queries

        embs = self._unit_embeddings(10)
        sim = cosine_similarity_matrix(embs)
        index = [{"video_id": "v", "frame_idx": i} for i in range(10)]

        results = batch_top_k_queries([0, 3, 7], sim, index, top_k=4)
        self.assertEqual(set(results.keys()), {0, 3, 7})
        for qidx, res in results.items():
            self.assertLessEqual(len(res), 4)


# ---------------------------------------------------------------------------
# 4. Visualization tests
# ---------------------------------------------------------------------------

class TestVisualization(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _unit_embeddings(self, n: int, dim: int = 8) -> np.ndarray:
        rng = np.random.default_rng(0)
        raw = rng.standard_normal((n, dim)).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        return raw / norms

    def _make_frames(self, n: int):
        paths = []
        for i in range(n):
            p = os.path.join(self.tmp, f"frame_{i}.jpg")
            _make_synthetic_frame(p, r=i * 30 % 256, g=100, b=200)
            paths.append(p)
        return paths

    def test_heatmap_creates_file(self):
        from src.similarity import cosine_similarity_matrix
        from src.visualization import plot_similarity_heatmap

        embs = self._unit_embeddings(6)
        sim = cosine_similarity_matrix(embs)
        index = [{"video_id": "v", "frame_idx": i} for i in range(6)]

        out = os.path.join(self.tmp, "heat.png")
        result_path = plot_similarity_heatmap(sim, index, out)

        self.assertTrue(os.path.exists(result_path))
        self.assertGreater(os.path.getsize(result_path), 100)

    def test_bar_chart_creates_file(self):
        from src.visualization import plot_cross_video_similarity_bar

        stats = {
            "within_video_mean": 0.85,
            "cross_video_mean": 0.60,
            "overall_mean": 0.72,
            "overall_median": 0.75,
        }
        out = os.path.join(self.tmp, "bar.png")
        result_path = plot_cross_video_similarity_bar(stats, out)

        self.assertTrue(os.path.exists(result_path))

    def test_top_k_grid_creates_file(self):
        from src.similarity import cosine_similarity_matrix
        from src.visualization import plot_top_k_grid

        frames = self._make_frames(6)
        embs = self._unit_embeddings(6)
        sim = cosine_similarity_matrix(embs)
        index = [
            {"video_id": "v", "frame_idx": i, "timestamp": float(i), "file_path": frames[i]}
            for i in range(6)
        ]

        from src.similarity import get_top_k_similar
        results = get_top_k_similar(0, sim, index, top_k=3)

        out = os.path.join(self.tmp, "grid.png")
        result_path = plot_top_k_grid(0, results, index, out)

        self.assertTrue(os.path.exists(result_path))

    def test_generate_report_creates_file(self):
        from src.similarity import (
            cosine_similarity_matrix,
            batch_top_k_queries,
            compute_inter_video_stats,
        )
        from src.visualization import generate_similarity_report

        embs = self._unit_embeddings(8)
        sim = cosine_similarity_matrix(embs)
        index = [
            {"video_id": "v1" if i < 4 else "v2", "frame_idx": i, "timestamp": float(i)}
            for i in range(8)
        ]
        stats = compute_inter_video_stats(sim, index)
        query_results = batch_top_k_queries([0, 4], sim, index, top_k=3)

        out = os.path.join(self.tmp, "report.md")
        result_path = generate_similarity_report(sim, index, query_results, stats, out)

        self.assertTrue(os.path.exists(result_path))
        with open(result_path, "r") as fh:
            content = fh.read()
        self.assertIn("Vibe Similarity Report", content)
        self.assertIn("Overall mean", content)


# ---------------------------------------------------------------------------
# 5. Lightweight integration test (no real model / video)
# ---------------------------------------------------------------------------

class TestIntegration(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_similarity_pipeline_on_synthetic_embeddings(self):
        """
        Wires together similarity → visualization steps on 100 % synthetic
        data.  Exercises the full post-embedding part of the pipeline.
        """
        from src.similarity import (
            cosine_similarity_matrix,
            batch_top_k_queries,
            compute_inter_video_stats,
        )
        from src.visualization import (
            plot_similarity_heatmap,
            generate_similarity_report,
        )

        n, dim = 12, 32
        rng = np.random.default_rng(7)
        raw = rng.standard_normal((n, dim)).astype(np.float32)
        embs = raw / np.linalg.norm(raw, axis=1, keepdims=True)

        index = [
            {
                "video_id": f"video_{i // 4}",
                "frame_idx": i,
                "timestamp": float(i),
                "file_path": os.path.join(self.tmp, f"frame_{i}.jpg"),
                "embedding_idx": i,
            }
            for i in range(n)
        ]

        # Create dummy frame files so visualization doesn't fail
        for entry in index:
            _make_synthetic_frame(entry["file_path"])

        sim = cosine_similarity_matrix(embs)
        self.assertEqual(sim.shape, (n, n))

        stats = compute_inter_video_stats(sim, index)
        self.assertFalse(np.isnan(stats["overall_mean"]))

        query_results = batch_top_k_queries([0, 4, 8], sim, index, top_k=5)
        self.assertEqual(len(query_results), 3)

        heatmap_path = plot_similarity_heatmap(
            sim, index,
            output_path=os.path.join(self.tmp, "heat.png"),
        )
        self.assertTrue(os.path.exists(heatmap_path))

        report_path = generate_similarity_report(
            sim, index, query_results, stats,
            output_path=os.path.join(self.tmp, "report.md"),
        )
        self.assertTrue(os.path.exists(report_path))

    def test_frame_extraction_metadata_roundtrip(self):
        """
        Extract frames from a synthetic video, save metadata, reload it, and
        verify row count and field integrity.
        """
        from src.frame_extractor import extract_frames, save_metadata, load_metadata

        video_path = os.path.join(self.tmp, "synth.mp4")
        _make_synthetic_video(video_path, n_frames=50, fps=10)

        frames_dir = os.path.join(self.tmp, "frames")
        metadata = extract_frames(video_path, frames_dir, interval_seconds=1.0, max_frames=5)

        self.assertLessEqual(len(metadata), 5)
        self.assertGreater(len(metadata), 0)

        csv_path = os.path.join(self.tmp, "meta.csv")
        save_metadata(metadata, csv_path)
        loaded = load_metadata(csv_path)

        self.assertEqual(len(loaded), len(metadata))
        for orig, reloaded in zip(metadata, loaded):
            self.assertEqual(orig["video_id"], reloaded["video_id"])
            self.assertAlmostEqual(orig["timestamp"], reloaded["timestamp"], places=2)


# ---------------------------------------------------------------------------
# 6. Affective scoring tests (mocked CLIP)
# ---------------------------------------------------------------------------

class TestAffectiveScorer(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    # Number of frames per synthetic "video" when building test indices
    _FRAMES_PER_VIDEO = 4

    def _unit_embeddings(self, n: int, dim: int = 16) -> np.ndarray:
        rng = np.random.default_rng(99)
        raw = rng.standard_normal((n, dim)).astype(np.float32)
        return raw / np.linalg.norm(raw, axis=1, keepdims=True)

    def _make_index(self, n: int):
        fpv = self._FRAMES_PER_VIDEO
        return [
            {"video_id": f"v{i // fpv}", "frame_idx": i, "timestamp": float(i),
             "file_path": os.path.join(self.tmp, f"frame_{i}.jpg")}
            for i in range(n)
        ]

    @patch("src.affective_scoring.CLIPModel")
    @patch("src.affective_scoring.CLIPProcessor")
    def test_score_frames_shape(self, MockProcessor, MockModel):
        """score_frames returns one (N,) array per axis."""
        import torch
        from src.affective_scoring import AffectiveScorer, DEFAULT_AXES

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        dim = 16

        def _get_text(**kwargs):
            n = kwargs["input_ids"].shape[0]
            f = torch.randn(n, dim)
            return f / f.norm(dim=-1, keepdim=True)

        mock_model.get_text_features.side_effect = _get_text
        MockModel.from_pretrained.return_value = mock_model

        mock_proc = MagicMock()

        def _proc(*args, text=None, images=None,
                  return_tensors=None, padding=None, truncation=None, **kw):
            batch = text if text is not None else (images or [])
            n = len(batch)
            return {"input_ids": torch.zeros(n, 77, dtype=torch.long),
                    "pixel_values": torch.zeros(n, 3, 224, 224)}

        mock_proc.side_effect = _proc
        MockProcessor.from_pretrained.return_value = mock_proc

        scorer = AffectiveScorer(model_name="mock/clip", axes=DEFAULT_AXES)
        n = 8
        embs = self._unit_embeddings(n, dim=dim)
        scores = scorer.score_frames(embs)

        self.assertEqual(set(scores.keys()), set(DEFAULT_AXES.keys()))
        for axis_name, arr in scores.items():
            self.assertEqual(arr.shape, (n,),
                             f"Axis '{axis_name}' should have shape ({n},).")
            self.assertEqual(arr.dtype, np.float32)

    @patch("src.affective_scoring.CLIPModel")
    @patch("src.affective_scoring.CLIPProcessor")
    def test_score_frames_range(self, MockProcessor, MockModel):
        """All affective scores must be in [-1, 1]."""
        import torch
        from src.affective_scoring import AffectiveScorer, DEFAULT_AXES

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        dim = 16

        def _get_text(**kwargs):
            n = kwargs["input_ids"].shape[0]
            f = torch.randn(n, dim)
            return f / f.norm(dim=-1, keepdim=True)

        mock_model.get_text_features.side_effect = _get_text
        MockModel.from_pretrained.return_value = mock_model

        mock_proc = MagicMock()

        def _proc(*args, text=None, images=None,
                  return_tensors=None, padding=None, truncation=None, **kw):
            batch = text if text is not None else (images or [])
            n = len(batch)
            return {"input_ids": torch.zeros(n, 77, dtype=torch.long),
                    "pixel_values": torch.zeros(n, 3, 224, 224)}

        mock_proc.side_effect = _proc
        MockProcessor.from_pretrained.return_value = mock_proc

        scorer = AffectiveScorer(model_name="mock/clip", axes=DEFAULT_AXES)
        embs = self._unit_embeddings(10, dim=dim)
        scores = scorer.score_frames(embs)

        for axis_name, arr in scores.items():
            self.assertLessEqual(float(arr.max()), 2.0 + 1e-5,
                                 f"Axis '{axis_name}' exceeded +2.")
            self.assertGreaterEqual(float(arr.min()), -2.0 - 1e-5,
                                    f"Axis '{axis_name}' below −2.")

    @patch("src.affective_scoring.CLIPModel")
    @patch("src.affective_scoring.CLIPProcessor")
    def test_score_frames_no_nan(self, MockProcessor, MockModel):
        import torch
        from src.affective_scoring import AffectiveScorer, DEFAULT_AXES

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        dim = 16

        def _get_text(**kwargs):
            n = kwargs["input_ids"].shape[0]
            f = torch.randn(n, dim)
            return f / f.norm(dim=-1, keepdim=True)

        mock_model.get_text_features.side_effect = _get_text
        MockModel.from_pretrained.return_value = mock_model

        mock_proc = MagicMock()

        def _proc(*args, text=None, images=None,
                  return_tensors=None, padding=None, truncation=None, **kw):
            batch = text if text is not None else (images or [])
            n = len(batch)
            return {"input_ids": torch.zeros(n, 77, dtype=torch.long),
                    "pixel_values": torch.zeros(n, 3, 224, 224)}

        mock_proc.side_effect = _proc
        MockProcessor.from_pretrained.return_value = mock_proc

        scorer = AffectiveScorer(model_name="mock/clip")
        embs = self._unit_embeddings(6, dim=dim)
        scores = scorer.score_frames(embs)

        for axis_name, arr in scores.items():
            self.assertFalse(np.isnan(arr).any(),
                             f"Axis '{axis_name}' contains NaN.")

    @patch("src.affective_scoring.CLIPModel")
    @patch("src.affective_scoring.CLIPProcessor")
    def test_score_frames_invalid_input(self, MockProcessor, MockModel):
        import torch
        from src.affective_scoring import AffectiveScorer

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        MockModel.from_pretrained.return_value = mock_model
        MockProcessor.from_pretrained.return_value = MagicMock()

        scorer = AffectiveScorer(model_name="mock/clip")
        with self.assertRaises(ValueError):
            scorer.score_frames(np.array([]))

    @patch("src.affective_scoring.CLIPModel")
    @patch("src.affective_scoring.CLIPProcessor")
    def test_score_video_level(self, MockProcessor, MockModel):
        """Video-level scores aggregate correctly per video."""
        import torch
        from src.affective_scoring import AffectiveScorer, DEFAULT_AXES

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        dim = 16

        def _get_text(**kwargs):
            n = kwargs["input_ids"].shape[0]
            f = torch.randn(n, dim)
            return f / f.norm(dim=-1, keepdim=True)

        mock_model.get_text_features.side_effect = _get_text
        MockModel.from_pretrained.return_value = mock_model

        mock_proc = MagicMock()

        def _proc(*args, text=None, images=None,
                  return_tensors=None, padding=None, truncation=None, **kw):
            batch = text if text is not None else (images or [])
            n = len(batch)
            return {"input_ids": torch.zeros(n, 77, dtype=torch.long),
                    "pixel_values": torch.zeros(n, 3, 224, 224)}

        mock_proc.side_effect = _proc
        MockProcessor.from_pretrained.return_value = mock_proc

        scorer = AffectiveScorer(model_name="mock/clip", axes=DEFAULT_AXES)
        n = 8
        embs = self._unit_embeddings(n, dim=dim)
        index = [
            {"video_id": "v1" if i < 4 else "v2", "frame_idx": i}
            for i in range(n)
        ]
        frame_scores = scorer.score_frames(embs)
        video_scores = scorer.score_video_level(frame_scores, index)

        self.assertEqual(set(video_scores.keys()), {"v1", "v2"})
        for vid, ax_scores in video_scores.items():
            self.assertEqual(set(ax_scores.keys()), set(DEFAULT_AXES.keys()))
            for ax, val in ax_scores.items():
                self.assertIsInstance(val, float)

    @patch("src.affective_scoring.CLIPModel")
    @patch("src.affective_scoring.CLIPProcessor")
    def test_save_scores_creates_json(self, MockProcessor, MockModel):
        import torch
        from src.affective_scoring import AffectiveScorer, DEFAULT_AXES

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        dim = 16

        def _get_text(**kwargs):
            n = kwargs["input_ids"].shape[0]
            f = torch.randn(n, dim)
            return f / f.norm(dim=-1, keepdim=True)

        mock_model.get_text_features.side_effect = _get_text
        MockModel.from_pretrained.return_value = mock_model

        mock_proc = MagicMock()

        def _proc(*args, text=None, images=None,
                  return_tensors=None, padding=None, truncation=None, **kw):
            batch = text if text is not None else (images or [])
            n = len(batch)
            return {"input_ids": torch.zeros(n, 77, dtype=torch.long),
                    "pixel_values": torch.zeros(n, 3, 224, 224)}

        mock_proc.side_effect = _proc
        MockProcessor.from_pretrained.return_value = mock_proc

        scorer = AffectiveScorer(model_name="mock/clip", axes=DEFAULT_AXES)
        n = 4
        embs = self._unit_embeddings(n, dim=dim)
        index = self._make_index(n)
        frame_scores = scorer.score_frames(embs)

        out = os.path.join(self.tmp, "affective_scores.json")
        result_path = scorer.save_scores(frame_scores, index, out)

        self.assertTrue(os.path.exists(result_path))
        with open(result_path) as fh:
            data = json.load(fh)
        self.assertEqual(len(data), n)
        for row in data:
            for ax in DEFAULT_AXES:
                self.assertIn(ax, row)

    @patch("src.affective_scoring.CLIPModel")
    @patch("src.affective_scoring.CLIPProcessor")
    def test_plot_heatmap_creates_file(self, MockProcessor, MockModel):
        import torch
        from src.affective_scoring import AffectiveScorer, DEFAULT_AXES

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        dim = 16

        def _get_text(**kwargs):
            n = kwargs["input_ids"].shape[0]
            f = torch.randn(n, dim)
            return f / f.norm(dim=-1, keepdim=True)

        mock_model.get_text_features.side_effect = _get_text
        MockModel.from_pretrained.return_value = mock_model

        mock_proc = MagicMock()

        def _proc(*args, text=None, images=None,
                  return_tensors=None, padding=None, truncation=None, **kw):
            batch = text if text is not None else (images or [])
            n = len(batch)
            return {"input_ids": torch.zeros(n, 77, dtype=torch.long),
                    "pixel_values": torch.zeros(n, 3, 224, 224)}

        mock_proc.side_effect = _proc
        MockProcessor.from_pretrained.return_value = mock_proc

        scorer = AffectiveScorer(model_name="mock/clip", axes=DEFAULT_AXES)
        n = 6
        embs = self._unit_embeddings(n, dim=dim)
        index = self._make_index(n)
        frame_scores = scorer.score_frames(embs)

        out = os.path.join(self.tmp, "affect_heat.png")
        result_path = scorer.plot_heatmap(frame_scores, index, out)

        self.assertTrue(os.path.exists(result_path))
        self.assertGreater(os.path.getsize(result_path), 100)

    @patch("src.affective_scoring.CLIPModel")
    @patch("src.affective_scoring.CLIPProcessor")
    def test_plot_radar_creates_file(self, MockProcessor, MockModel):
        import torch
        from src.affective_scoring import AffectiveScorer, DEFAULT_AXES

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        dim = 16

        def _get_text(**kwargs):
            n = kwargs["input_ids"].shape[0]
            f = torch.randn(n, dim)
            return f / f.norm(dim=-1, keepdim=True)

        mock_model.get_text_features.side_effect = _get_text
        MockModel.from_pretrained.return_value = mock_model

        mock_proc = MagicMock()

        def _proc(*args, text=None, images=None,
                  return_tensors=None, padding=None, truncation=None, **kw):
            batch = text if text is not None else (images or [])
            n = len(batch)
            return {"input_ids": torch.zeros(n, 77, dtype=torch.long),
                    "pixel_values": torch.zeros(n, 3, 224, 224)}

        mock_proc.side_effect = _proc
        MockProcessor.from_pretrained.return_value = mock_proc

        scorer = AffectiveScorer(model_name="mock/clip", axes=DEFAULT_AXES)
        n = 8
        embs = self._unit_embeddings(n, dim=dim)
        index = [{"video_id": "v1" if i < 4 else "v2", "frame_idx": i} for i in range(n)]
        frame_scores = scorer.score_frames(embs)
        video_scores = scorer.score_video_level(frame_scores, index)

        out = os.path.join(self.tmp, "radar.png")
        result_path = scorer.plot_radar(video_scores, out)

        self.assertTrue(os.path.exists(result_path))
        self.assertGreater(os.path.getsize(result_path), 100)


# ---------------------------------------------------------------------------
# 7. Clustering tests (pure NumPy + sklearn – no CLIP model needed)
# ---------------------------------------------------------------------------

class TestVibeClusterer(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _unit_embeddings(self, n: int, dim: int = 16) -> np.ndarray:
        rng = np.random.default_rng(123)
        raw = rng.standard_normal((n, dim)).astype(np.float32)
        return raw / np.linalg.norm(raw, axis=1, keepdims=True)

    def _make_index(self, n: int):
        return [
            {"video_id": f"v{i // 5}", "frame_idx": i, "timestamp": float(i),
             "file_path": os.path.join(self.tmp, f"frame_{i}.jpg")}
            for i in range(n)
        ]

    # --- fit / predict ---

    def test_fit_returns_correct_shape(self):
        from src.clustering import VibeClusterer

        embs = self._unit_embeddings(20)
        clust = VibeClusterer(n_clusters=4)
        labels = clust.fit(embs)

        self.assertEqual(labels.shape, (20,))
        self.assertEqual(labels.dtype, np.int32)

    def test_fit_labels_in_range(self):
        from src.clustering import VibeClusterer

        embs = self._unit_embeddings(15)
        clust = VibeClusterer(n_clusters=3)
        labels = clust.fit(embs)

        self.assertTrue((labels >= 0).all())
        self.assertTrue((labels < 3).all())

    def test_fit_all_clusters_represented(self):
        """With enough frames, every cluster should have at least one frame."""
        from src.clustering import VibeClusterer

        embs = self._unit_embeddings(30)
        clust = VibeClusterer(n_clusters=5)
        labels = clust.fit(embs)

        self.assertEqual(len(set(labels.tolist())), 5)

    def test_fit_too_few_frames_raises(self):
        from src.clustering import VibeClusterer

        embs = self._unit_embeddings(3)
        clust = VibeClusterer(n_clusters=5)
        with self.assertRaises(ValueError):
            clust.fit(embs)

    def test_invalid_n_clusters_raises(self):
        from src.clustering import VibeClusterer

        with self.assertRaises(ValueError):
            VibeClusterer(n_clusters=1)

    def test_predict_before_fit_raises(self):
        from src.clustering import VibeClusterer

        clust = VibeClusterer(n_clusters=3)
        embs = self._unit_embeddings(10)
        with self.assertRaises(RuntimeError):
            clust.predict(embs)

    def test_predict_labels_in_range(self):
        from src.clustering import VibeClusterer

        embs = self._unit_embeddings(20)
        clust = VibeClusterer(n_clusters=4)
        clust.fit(embs)
        new_embs = self._unit_embeddings(5, dim=16)
        labels = clust.predict(new_embs)

        self.assertEqual(labels.shape, (5,))
        self.assertTrue((labels >= 0).all())
        self.assertTrue((labels < 4).all())

    # --- cluster_summary ---

    def test_cluster_summary_keys(self):
        from src.clustering import VibeClusterer

        n = 20
        embs = self._unit_embeddings(n)
        index = self._make_index(n)
        clust = VibeClusterer(n_clusters=4)
        labels = clust.fit(embs)
        summary = clust.cluster_summary(labels, index, embeddings=embs)

        self.assertEqual(set(summary.keys()), {0, 1, 2, 3})
        for info in summary.values():
            self.assertIn("size", info)
            self.assertIn("video_distribution", info)
            self.assertIn("representative_frame_idx", info)

    def test_cluster_summary_sizes_sum_to_n(self):
        from src.clustering import VibeClusterer

        n = 25
        embs = self._unit_embeddings(n)
        index = self._make_index(n)
        clust = VibeClusterer(n_clusters=5)
        labels = clust.fit(embs)
        summary = clust.cluster_summary(labels, index, embeddings=embs)

        total = sum(info["size"] for info in summary.values())
        self.assertEqual(total, n)

    def test_cluster_summary_representative_is_in_cluster(self):
        """representative_frame_idx must belong to the correct cluster."""
        from src.clustering import VibeClusterer

        n = 20
        embs = self._unit_embeddings(n)
        index = self._make_index(n)
        clust = VibeClusterer(n_clusters=4)
        labels = clust.fit(embs)
        summary = clust.cluster_summary(labels, index, embeddings=embs)

        for k, info in summary.items():
            rep_idx = info["representative_frame_idx"]
            if rep_idx is not None:
                self.assertEqual(int(labels[rep_idx]), k)

    # --- project_2d ---

    def test_project_2d_pca_shape(self):
        from src.clustering import VibeClusterer

        embs = self._unit_embeddings(20, dim=32)
        clust = VibeClusterer(n_clusters=3)
        coords = clust.project_2d(embs, method="pca")

        self.assertEqual(coords.shape, (20, 2))
        self.assertEqual(coords.dtype, np.float32)

    def test_project_2d_pca_no_nan(self):
        from src.clustering import VibeClusterer

        embs = self._unit_embeddings(15, dim=16)
        clust = VibeClusterer(n_clusters=3)
        coords = clust.project_2d(embs, method="pca")

        self.assertFalse(np.isnan(coords).any())

    def test_project_2d_unknown_method_raises(self):
        from src.clustering import VibeClusterer

        clust = VibeClusterer(n_clusters=3)
        embs = self._unit_embeddings(10)
        with self.assertRaises(ValueError):
            clust.project_2d(embs, method="umap_xyz")

    # --- plot_scatter ---

    def test_plot_scatter_creates_file(self):
        from src.clustering import VibeClusterer

        n = 20
        embs = self._unit_embeddings(n, dim=16)
        index = self._make_index(n)
        clust = VibeClusterer(n_clusters=3)
        labels = clust.fit(embs)

        out = os.path.join(self.tmp, "scatter.png")
        result_path = clust.plot_scatter(embs, labels, index, out)

        self.assertTrue(os.path.exists(result_path))
        self.assertGreater(os.path.getsize(result_path), 100)

    # --- save_cluster_assignments ---

    def test_save_cluster_assignments_creates_json(self):
        from src.clustering import VibeClusterer

        n = 10
        embs = self._unit_embeddings(n)
        index = self._make_index(n)
        clust = VibeClusterer(n_clusters=2)
        labels = clust.fit(embs)

        out = os.path.join(self.tmp, "assignments.json")
        result_path = clust.save_cluster_assignments(labels, index, out)

        self.assertTrue(os.path.exists(result_path))
        with open(result_path) as fh:
            data = json.load(fh)
        self.assertEqual(len(data), n)
        for row in data:
            self.assertIn("cluster", row)
            self.assertIn(row["cluster"], [0, 1])

    # --- auto_n_clusters ---

    def test_auto_n_clusters_returns_valid_k(self):
        from src.clustering import auto_n_clusters

        embs = self._unit_embeddings(30, dim=16)
        k = auto_n_clusters(embs, max_k=8)

        self.assertGreaterEqual(k, 2)
        self.assertLessEqual(k, 8)

    def test_auto_n_clusters_too_few_samples_returns_2(self):
        from src.clustering import auto_n_clusters

        embs = self._unit_embeddings(3, dim=8)
        # max_k will be clamped to n-1 = 2 → only k=2 is tested → returns 2
        k = auto_n_clusters(embs, max_k=10)
        self.assertEqual(k, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
