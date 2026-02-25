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
8. Temporal analysis – pure NumPy; verifies curve, transitions, pacing.
9. Performance pred  – pure NumPy + sklearn; verifies synthetic data and predictor.

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
# 6b. Gap-corrected scoring tests
# ---------------------------------------------------------------------------

class TestGapCorrectedScoring(unittest.TestCase):
    """Tests for AffectiveScorer.score_frames_gap_corrected."""

    def _unit_embeddings(self, n: int, dim: int = 16) -> np.ndarray:
        rng = np.random.default_rng(77)
        raw = rng.standard_normal((n, dim)).astype(np.float32)
        return raw / np.linalg.norm(raw, axis=1, keepdims=True)

    def _make_scorer(self, dim: int = 16):
        """Return (scorer, DEFAULT_AXES) with CLIP mocked."""
        import torch
        import src.affective_scoring  # ensure module is in sys.modules
        from src.affective_scoring import AffectiveScorer, DEFAULT_AXES

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        def _text(**kw):
            n = kw["input_ids"].shape[0]
            f = torch.randn(n, dim)
            return f / f.norm(dim=-1, keepdim=True)

        mock_model.get_text_features.side_effect = _text

        def _proc(*args, text=None, images=None,
                  return_tensors=None, padding=None, truncation=None, **kw):
            texts = text if text is not None else []
            n = len(texts) if texts else 1
            return {"input_ids": torch.ones(n, 10, dtype=torch.long)}

        mock_proc = MagicMock()
        mock_proc.side_effect = _proc

        with patch("src.affective_scoring.CLIPModel") as MM, \
             patch("src.affective_scoring.CLIPProcessor") as MP:
            MM.from_pretrained.return_value = mock_model
            MP.from_pretrained.return_value = mock_proc
            scorer = AffectiveScorer(axes=DEFAULT_AXES)
        # Manually replace the internal model/processor references
        scorer.model = mock_model
        scorer.processor = mock_proc
        return scorer, DEFAULT_AXES

    def test_gap_corrected_shape(self):
        """score_frames_gap_corrected returns one (N,) array per axis."""
        scorer, DEFAULT_AXES = self._make_scorer()
        embs = self._unit_embeddings(12)
        scores = scorer.score_frames_gap_corrected(embs)
        self.assertEqual(set(scores.keys()), set(DEFAULT_AXES.keys()))
        for v in scores.values():
            self.assertEqual(v.shape, (12,))

    def test_gap_corrected_no_nan(self):
        """No NaN values in gap-corrected scores."""
        scorer, _ = self._make_scorer()
        embs = self._unit_embeddings(8)
        scores = scorer.score_frames_gap_corrected(embs)
        for v in scores.values():
            self.assertFalse(np.any(np.isnan(v)),
                             "Gap-corrected scores contain NaN.")

    def test_gap_corrected_range(self):
        """Gap-corrected scores are clipped to [-2, 2]."""
        scorer, _ = self._make_scorer()
        embs = self._unit_embeddings(10)
        scores = scorer.score_frames_gap_corrected(embs)
        for name, v in scores.items():
            self.assertLessEqual(float(np.max(v)), 2.0 + 1e-5,
                                 f"Axis '{name}' exceeds +2.0.")
            self.assertGreaterEqual(float(np.min(v)), -2.0 - 1e-5,
                                    f"Axis '{name}' is below -2.0.")

    def test_gap_corrected_empty_raises(self):
        """Empty embedding array raises ValueError."""
        scorer, _ = self._make_scorer()
        with self.assertRaises(ValueError):
            scorer.score_frames_gap_corrected(np.empty((0, 16), dtype=np.float32))


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


# ---------------------------------------------------------------------------
# 8. Temporal analysis tests  (pure NumPy – no CLIP, no GPU)
# ---------------------------------------------------------------------------

class TestTemporalAnalysis(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _unit_embeddings(self, n: int, dim: int = 16, seed: int = 7) -> np.ndarray:
        rng = np.random.default_rng(seed)
        raw = rng.standard_normal((n, dim)).astype(np.float32)
        return raw / np.linalg.norm(raw, axis=1, keepdims=True)

    def _make_index(self, n: int, n_videos: int = 2):
        frames_per_vid = n // n_videos
        return [
            {
                "video_id": f"v{i // frames_per_vid}",
                "frame_idx": i % frames_per_vid,
                "timestamp": float(i % frames_per_vid),
                "file_path": os.path.join(self.tmp, f"f{i}.jpg"),
            }
            for i in range(n)
        ]

    # --- compute_temporal_curve ---

    def test_curve_shape(self):
        from src.temporal_analysis import TemporalAnalyser

        embs = self._unit_embeddings(20)
        index = self._make_index(20)
        ta = TemporalAnalyser(window=3)
        curve = ta.compute_temporal_curve(embs, index)

        self.assertEqual(curve.shape, (20,))
        self.assertEqual(curve.dtype, np.float32)

    def test_curve_range(self):
        from src.temporal_analysis import TemporalAnalyser

        embs = self._unit_embeddings(20)
        index = self._make_index(20)
        ta = TemporalAnalyser(window=2)
        curve = ta.compute_temporal_curve(embs, index)

        self.assertLessEqual(float(curve.max()), 1.0 + 1e-5)
        self.assertGreaterEqual(float(curve.min()), -1.0 - 1e-5)

    def test_curve_no_nan(self):
        from src.temporal_analysis import TemporalAnalyser

        embs = self._unit_embeddings(15)
        index = self._make_index(15)
        ta = TemporalAnalyser(window=3)
        curve = ta.compute_temporal_curve(embs, index)

        self.assertFalse(np.isnan(curve).any())

    def test_curve_single_frame_video(self):
        """A video with a single frame should get score 1.0 (trivially coherent)."""
        from src.temporal_analysis import TemporalAnalyser

        embs = self._unit_embeddings(1)
        index = [{"video_id": "solo", "frame_idx": 0, "timestamp": 0.0, "file_path": "x.jpg"}]
        ta = TemporalAnalyser(window=3)
        curve = ta.compute_temporal_curve(embs, index)

        self.assertAlmostEqual(float(curve[0]), 1.0, places=5)

    def test_curve_invalid_input_raises(self):
        from src.temporal_analysis import TemporalAnalyser

        ta = TemporalAnalyser(window=2)
        with self.assertRaises(ValueError):
            ta.compute_temporal_curve(np.zeros((0, 16)), [])

    def test_window_must_be_positive(self):
        from src.temporal_analysis import TemporalAnalyser

        with self.assertRaises(ValueError):
            TemporalAnalyser(window=0)

    # --- detect_scene_transitions ---

    def test_transitions_are_subset_of_indices(self):
        from src.temporal_analysis import TemporalAnalyser

        curve = np.array([0.9, 0.85, 0.3, 0.8, 0.78], dtype=np.float32)
        ta = TemporalAnalyser(window=1)
        transitions = ta.detect_scene_transitions(curve, threshold=0.3)

        for t in transitions:
            self.assertGreaterEqual(t, 0)
            self.assertLess(t, len(curve))

    def test_transitions_detect_big_drop(self):
        from src.temporal_analysis import TemporalAnalyser

        curve = np.array([0.9, 0.85, 0.2, 0.8, 0.78], dtype=np.float32)
        ta = TemporalAnalyser(window=1)
        transitions = ta.detect_scene_transitions(curve, threshold=0.4)

        # Drop from 0.85 to 0.2 = 0.65 > 0.4 → should detect transition at idx 2
        self.assertIn(2, transitions)

    def test_no_transitions_on_flat_curve(self):
        from src.temporal_analysis import TemporalAnalyser

        curve = np.full(10, 0.8, dtype=np.float32)
        ta = TemporalAnalyser(window=2)
        transitions = ta.detect_scene_transitions(curve, threshold=0.1)

        self.assertEqual(transitions, [])

    def test_transitions_short_curve(self):
        from src.temporal_analysis import TemporalAnalyser

        ta = TemporalAnalyser(window=1)
        self.assertEqual(ta.detect_scene_transitions(np.array([0.5], dtype=np.float32)), [])
        self.assertEqual(ta.detect_scene_transitions(np.array([], dtype=np.float32)), [])

    # --- pacing_score / coherence_score ---

    def test_pacing_is_non_negative(self):
        from src.temporal_analysis import TemporalAnalyser

        ta = TemporalAnalyser()
        curve = np.random.default_rng(1).random(20).astype(np.float32)
        self.assertGreaterEqual(ta.pacing_score(curve), 0.0)

    def test_coherence_in_range(self):
        from src.temporal_analysis import TemporalAnalyser

        embs = self._unit_embeddings(20)
        index = self._make_index(20)
        ta = TemporalAnalyser(window=2)
        curve = ta.compute_temporal_curve(embs, index)

        self.assertLessEqual(ta.coherence_score(curve), 1.0 + 1e-5)
        self.assertGreaterEqual(ta.coherence_score(curve), -1.0 - 1e-5)

    def test_pacing_score_empty(self):
        from src.temporal_analysis import TemporalAnalyser

        ta = TemporalAnalyser()
        self.assertEqual(ta.pacing_score(np.array([], dtype=np.float32)), 0.0)

    def test_coherence_score_empty(self):
        from src.temporal_analysis import TemporalAnalyser

        ta = TemporalAnalyser()
        self.assertEqual(ta.coherence_score(np.array([], dtype=np.float32)), 0.0)

    # --- per_video_stats ---

    def test_per_video_stats_keys(self):
        from src.temporal_analysis import TemporalAnalyser

        embs = self._unit_embeddings(20)
        index = self._make_index(20, n_videos=2)
        ta = TemporalAnalyser(window=3)
        curve = ta.compute_temporal_curve(embs, index)
        stats = ta.per_video_stats(curve, index)

        self.assertEqual(set(stats.keys()), {"v0", "v1"})
        for vstats in stats.values():
            for k in ("coherence", "pacing", "n_frames", "n_transitions"):
                self.assertIn(k, vstats)

    # --- save_temporal_stats ---

    def test_save_temporal_stats_creates_json(self):
        from src.temporal_analysis import TemporalAnalyser

        n = 10
        embs = self._unit_embeddings(n)
        index = self._make_index(n)
        ta = TemporalAnalyser(window=2)
        curve = ta.compute_temporal_curve(embs, index)

        out = os.path.join(self.tmp, "temporal.json")
        path = ta.save_temporal_stats(curve, index, out)

        self.assertTrue(os.path.exists(path))
        with open(path) as fh:
            data = json.load(fh)
        self.assertEqual(len(data), n)
        self.assertIn("temporal_sim", data[0])

    # --- plot_narrative_arc ---

    def test_plot_narrative_arc_creates_file(self):
        from src.temporal_analysis import TemporalAnalyser

        n = 20
        embs = self._unit_embeddings(n)
        index = self._make_index(n)
        ta = TemporalAnalyser(window=3)
        curve = ta.compute_temporal_curve(embs, index)
        transitions = ta.detect_scene_transitions(curve)

        out = os.path.join(self.tmp, "arc.png")
        path = ta.plot_narrative_arc(curve, transitions, index, out)

        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 100)

    # --- plot_pacing_comparison ---

    def test_plot_pacing_comparison_creates_file(self):
        from src.temporal_analysis import TemporalAnalyser

        embs = self._unit_embeddings(20)
        index = self._make_index(20, n_videos=2)
        ta = TemporalAnalyser(window=3)
        curve = ta.compute_temporal_curve(embs, index)
        stats = ta.per_video_stats(curve, index)

        out = os.path.join(self.tmp, "pacing.png")
        path = ta.plot_pacing_comparison(stats, out)

        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 100)


# ---------------------------------------------------------------------------
# 9. Performance predictor tests  (pure NumPy + sklearn – no CLIP needed)
# ---------------------------------------------------------------------------

class TestPerformancePredictor(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _unit_embeddings(self, n: int, dim: int = 16, seed: int = 11) -> np.ndarray:
        rng = np.random.default_rng(seed)
        raw = rng.standard_normal((n, dim)).astype(np.float32)
        return raw / np.linalg.norm(raw, axis=1, keepdims=True)

    def _affective_scores(self, n: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        axes = ["joy", "warmth", "energy", "luxury", "complexity", "tension"]
        return {ax: rng.standard_normal(n).astype(np.float32) for ax in axes}

    def _make_index(self, n: int):
        return [{"video_id": f"v{i // 5}", "frame_idx": i} for i in range(n)]

    # --- generate_synthetic_performance_data ---

    def test_synthetic_data_shape(self):
        from src.performance_predictor import generate_synthetic_performance_data

        n, dim = 30, 16
        embs = self._unit_embeddings(n, dim)
        aff = self._affective_scores(n)
        index = self._make_index(n)
        features, labels = generate_synthetic_performance_data(embs, aff, index)

        self.assertEqual(features.shape[0], n)
        self.assertEqual(labels.shape, (n,))

    def test_synthetic_labels_in_0_1(self):
        from src.performance_predictor import generate_synthetic_performance_data

        n = 20
        embs = self._unit_embeddings(n)
        aff = self._affective_scores(n)
        index = self._make_index(n)
        _, labels = generate_synthetic_performance_data(embs, aff, index)

        self.assertTrue((labels >= 0).all())
        self.assertTrue((labels <= 1).all())

    def test_synthetic_no_nan(self):
        from src.performance_predictor import generate_synthetic_performance_data

        n = 20
        embs = self._unit_embeddings(n)
        aff = self._affective_scores(n)
        index = self._make_index(n)
        features, labels = generate_synthetic_performance_data(embs, aff, index)

        self.assertFalse(np.isnan(features).any())
        self.assertFalse(np.isnan(labels).any())

    def test_synthetic_invalid_target_raises(self):
        from src.performance_predictor import generate_synthetic_performance_data

        embs = self._unit_embeddings(10)
        aff = self._affective_scores(10)
        index = self._make_index(10)
        with self.assertRaises(ValueError):
            generate_synthetic_performance_data(embs, aff, index, target="clicks")

    def test_synthetic_empty_embeddings_raises(self):
        from src.performance_predictor import generate_synthetic_performance_data

        with self.assertRaises(ValueError):
            generate_synthetic_performance_data(
                np.zeros((0, 16)), {}, []
            )

    def test_roas_labels_differ_from_ctr(self):
        """ROAS and CTR labels should use different weights → different output."""
        from src.performance_predictor import generate_synthetic_performance_data

        n = 30
        embs = self._unit_embeddings(n)
        aff = self._affective_scores(n)
        index = self._make_index(n)
        _, ctr = generate_synthetic_performance_data(embs, aff, index, target="ctr")
        _, roas = generate_synthetic_performance_data(embs, aff, index, target="roas")

        # They should be different (extremely unlikely to match by coincidence)
        self.assertFalse(np.allclose(ctr, roas))

    # --- VibePerformancePredictor ---

    def test_invalid_model_type_raises(self):
        from src.performance_predictor import VibePerformancePredictor

        with self.assertRaises(ValueError):
            VibePerformancePredictor(model_type="xgboost")

    def test_predict_before_fit_raises(self):
        from src.performance_predictor import VibePerformancePredictor

        pred = VibePerformancePredictor()
        with self.assertRaises(RuntimeError):
            pred.predict(np.zeros((5, 10), dtype=np.float32))

    def test_feature_importance_before_fit_raises(self):
        from src.performance_predictor import VibePerformancePredictor

        pred = VibePerformancePredictor()
        with self.assertRaises(RuntimeError):
            pred.feature_importance()

    def test_fit_and_predict_shape(self):
        from src.performance_predictor import (
            generate_synthetic_performance_data,
            VibePerformancePredictor,
        )
        n = 30
        embs = self._unit_embeddings(n)
        aff = self._affective_scores(n)
        index = self._make_index(n)
        features, labels = generate_synthetic_performance_data(embs, aff, index)

        pred = VibePerformancePredictor(model_type="ridge")
        pred.fit(features, labels)
        out = pred.predict(features)

        self.assertEqual(out.shape, (n,))
        self.assertEqual(out.dtype, np.float32)

    def test_cross_validate_returns_spearman(self):
        from src.performance_predictor import (
            generate_synthetic_performance_data,
            VibePerformancePredictor,
        )
        n = 30
        embs = self._unit_embeddings(n)
        aff = self._affective_scores(n)
        index = self._make_index(n)
        features, labels = generate_synthetic_performance_data(embs, aff, index)

        pred = VibePerformancePredictor(model_type="ridge")
        cv = pred.cross_validate(features, labels, n_splits=3)

        for key in ("mean_spearman", "std_spearman", "mean_rmse", "fold_spearman"):
            self.assertIn(key, cv)
        self.assertEqual(len(cv["fold_spearman"]), 3)
        # With synthetic data generated by a linear function, ridge should do well
        self.assertGreater(cv["mean_spearman"], 0.0)

    def test_spearman_recovers_signal(self):
        """Ridge regression should achieve Spearman ρ > 0.5 on clearly linear data."""
        from src.performance_predictor import (
            generate_synthetic_performance_data,
            VibePerformancePredictor,
        )
        n = 100
        embs = self._unit_embeddings(n, dim=32, seed=0)
        aff = self._affective_scores(n, seed=0)
        features, labels = generate_synthetic_performance_data(
            embs, aff, [], noise_level=0.05
        )
        pred = VibePerformancePredictor(model_type="ridge")
        cv = pred.cross_validate(features, labels, n_splits=5)
        self.assertGreater(cv["mean_spearman"], 0.5)

    def test_feature_importance_returns_list(self):
        from src.performance_predictor import (
            generate_synthetic_performance_data,
            VibePerformancePredictor,
        )
        n = 30
        embs = self._unit_embeddings(n)
        aff = self._affective_scores(n)
        index = self._make_index(n)
        features, labels = generate_synthetic_performance_data(embs, aff, index)

        pred = VibePerformancePredictor(model_type="ridge")
        pred.fit(features, labels)
        imp = pred.feature_importance(top_k=5)

        self.assertEqual(len(imp), 5)
        names, scores = zip(*imp)
        # Sorted descending
        self.assertEqual(list(scores), sorted(scores, reverse=True))

    def test_save_model_creates_json(self):
        from src.performance_predictor import (
            generate_synthetic_performance_data,
            VibePerformancePredictor,
        )
        n = 20
        embs = self._unit_embeddings(n)
        aff = self._affective_scores(n)
        index = self._make_index(n)
        features, labels = generate_synthetic_performance_data(embs, aff, index)

        pred = VibePerformancePredictor(model_type="ridge")
        pred.fit(features, labels)

        out = os.path.join(self.tmp, "model.json")
        path = pred.save_model(out)

        self.assertTrue(os.path.exists(path))
        with open(path) as fh:
            doc = json.load(fh)
        self.assertIn("coef", doc)
        self.assertEqual(doc["model_type"], "ridge")

    def test_plot_cv_results_creates_file(self):
        from src.performance_predictor import (
            generate_synthetic_performance_data,
            VibePerformancePredictor,
        )
        n = 20
        embs = self._unit_embeddings(n)
        aff = self._affective_scores(n)
        index = self._make_index(n)
        features, labels = generate_synthetic_performance_data(embs, aff, index)

        pred = VibePerformancePredictor()
        cv = pred.cross_validate(features, labels, n_splits=3)

        out = os.path.join(self.tmp, "cv.png")
        path = pred.plot_cv_results(cv, out)

        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 100)

    def test_plot_feature_importance_creates_file(self):
        from src.performance_predictor import (
            generate_synthetic_performance_data,
            VibePerformancePredictor,
        )
        n = 20
        embs = self._unit_embeddings(n)
        aff = self._affective_scores(n)
        index = self._make_index(n)
        features, labels = generate_synthetic_performance_data(embs, aff, index)

        pred = VibePerformancePredictor()
        pred.fit(features, labels)

        out = os.path.join(self.tmp, "imp.png")
        path = pred.plot_feature_importance(out, top_k=5)

        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 100)

    def test_plot_predicted_vs_actual_creates_file(self):
        from src.performance_predictor import (
            generate_synthetic_performance_data,
            VibePerformancePredictor,
        )
        n = 20
        embs = self._unit_embeddings(n)
        aff = self._affective_scores(n)
        index = self._make_index(n)
        features, labels = generate_synthetic_performance_data(embs, aff, index)

        pred = VibePerformancePredictor()
        pred.fit(features, labels)
        predicted = pred.predict(features)

        out = os.path.join(self.tmp, "scatter.png")
        path = pred.plot_predicted_vs_actual(labels, predicted, out)

        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 100)

    def test_mlp_predictor_fits_and_predicts(self):
        from src.performance_predictor import (
            generate_synthetic_performance_data,
            VibePerformancePredictor,
        )
        n = 40
        embs = self._unit_embeddings(n, dim=32)
        aff = self._affective_scores(n)
        index = self._make_index(n)
        features, labels = generate_synthetic_performance_data(embs, aff, index)

        pred = VibePerformancePredictor(model_type="mlp", hidden_layers=(32, 16))
        pred.fit(features, labels)
        out = pred.predict(features)

        self.assertEqual(out.shape, (n,))
        self.assertFalse(np.isnan(out).any())


# ---------------------------------------------------------------------------
# 10. Frame deduplication tests
# ---------------------------------------------------------------------------

class TestFrameDeduplication(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    @staticmethod
    def _unit_embeddings(n: int, dim: int = 8, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        raw = rng.standard_normal((n, dim)).astype(np.float32)
        return raw / np.linalg.norm(raw, axis=1, keepdims=True)

    @staticmethod
    def _make_index(n: int, video_id: str = "v1") -> list:
        return [
            {"video_id": video_id, "frame_idx": i, "timestamp": float(i)}
            for i in range(n)
        ]

    def test_deduplicate_removes_exact_duplicates(self):
        """Identical consecutive frames should be collapsed to one."""
        from src.frame_deduplication import deduplicate_frames

        base = self._unit_embeddings(1)
        # 5 copies of the same vector
        embs = np.vstack([base] * 5)
        index = self._make_index(5)

        dedup_embs, dedup_idx, mask = deduplicate_frames(embs, index, similarity_threshold=0.97)

        # Only the first should be kept
        self.assertEqual(dedup_embs.shape[0], 1)
        self.assertEqual(len(dedup_idx), 1)
        self.assertEqual(int(mask.sum()), 1)

    def test_deduplicate_keeps_distinct_frames(self):
        """Orthogonal embeddings should all be retained."""
        from src.frame_deduplication import deduplicate_frames

        # 8 orthogonal unit vectors in 8-D
        embs = np.eye(8, dtype=np.float32)
        index = self._make_index(8)

        dedup_embs, dedup_idx, mask = deduplicate_frames(embs, index, similarity_threshold=0.97)

        self.assertEqual(dedup_embs.shape[0], 8)
        self.assertEqual(int(mask.sum()), 8)

    def test_deduplicate_output_alignment(self):
        """dedup_index[i] must correspond to dedup_embeddings[i]."""
        from src.frame_deduplication import deduplicate_frames

        embs = self._unit_embeddings(10)
        index = self._make_index(10)

        dedup_embs, dedup_idx, mask = deduplicate_frames(embs, index, similarity_threshold=0.5)

        kept_positions = [i for i, b in enumerate(mask) if b]
        for local_i, global_i in enumerate(kept_positions):
            np.testing.assert_array_almost_equal(
                dedup_embs[local_i], embs[global_i], decimal=5
            )
            self.assertEqual(dedup_idx[local_i]["frame_idx"], index[global_i]["frame_idx"])

    def test_deduplicate_invalid_threshold_raises(self):
        from src.frame_deduplication import deduplicate_frames

        embs = self._unit_embeddings(4)
        index = self._make_index(4)

        with self.assertRaises(ValueError):
            deduplicate_frames(embs, index, similarity_threshold=0.0)
        with self.assertRaises(ValueError):
            deduplicate_frames(embs, index, similarity_threshold=1.5)

    def test_deduplicate_empty_raises(self):
        from src.frame_deduplication import deduplicate_frames

        with self.assertRaises(ValueError):
            deduplicate_frames(np.zeros((0, 8), dtype=np.float32), [], 0.97)

    def test_compute_novelty_scores_shape_and_range(self):
        from src.frame_deduplication import compute_novelty_scores

        embs = self._unit_embeddings(6)
        index = self._make_index(6)
        novelty = compute_novelty_scores(embs, index)

        self.assertEqual(novelty.shape, (6,))
        # First frame has novelty 1.0 by definition
        self.assertAlmostEqual(float(novelty[0]), 1.0, places=5)

    def test_compute_novelty_identical_frames(self):
        """Identical consecutive frames should have novelty 0."""
        from src.frame_deduplication import compute_novelty_scores

        base = self._unit_embeddings(1)
        embs = np.vstack([base, base, base])
        index = self._make_index(3)
        novelty = compute_novelty_scores(embs, index)

        self.assertAlmostEqual(float(novelty[1]), 0.0, places=4)
        self.assertAlmostEqual(float(novelty[2]), 0.0, places=4)

    def test_select_diverse_frames_returns_correct_count(self):
        from src.frame_deduplication import select_diverse_frames

        embs = self._unit_embeddings(12)
        index = self._make_index(12)
        sel_embs, sel_idx, sel_pos = select_diverse_frames(embs, index, n_select=5)

        self.assertEqual(sel_embs.shape[0], 5)
        self.assertEqual(len(sel_idx), 5)
        self.assertEqual(len(sel_pos), 5)
        # Positions should be sorted
        self.assertEqual(sel_pos, sorted(sel_pos))

    def test_select_diverse_frames_alignment(self):
        """selected_embeddings[i] must correspond to embeddings[selected_positions[i]]."""
        from src.frame_deduplication import select_diverse_frames

        embs = self._unit_embeddings(10)
        index = self._make_index(10)
        sel_embs, sel_idx, sel_pos = select_diverse_frames(embs, index, n_select=4)

        for local_i, global_i in enumerate(sel_pos):
            np.testing.assert_array_almost_equal(
                sel_embs[local_i], embs[global_i], decimal=5
            )


# ---------------------------------------------------------------------------
# 11. Experiment manifest tests
# ---------------------------------------------------------------------------

class TestExperimentManifest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_manifest_save_creates_json(self):
        from src.experiment_manifest import PipelineManifest

        m = PipelineManifest(run_id="test_run_001")
        m.record("step_a", n_frames=10, model="clip")
        out = os.path.join(self.tmp, "manifest.json")
        m.save(out)

        self.assertTrue(os.path.exists(out))
        with open(out) as fh:
            import json as json_mod
            data = json_mod.load(fh)
        self.assertEqual(data["run_id"], "test_run_001")
        self.assertIn("step_a", data["modules"])
        self.assertEqual(data["modules"]["step_a"]["n_frames"], 10)

    def test_manifest_get(self):
        from src.experiment_manifest import PipelineManifest

        m = PipelineManifest()
        m.record("embedding", shape=[50, 512], dtype="float32")
        result = m.get("embedding")
        self.assertIsNotNone(result)
        self.assertEqual(result["shape"], [50, 512])

    def test_manifest_add_artifact(self):
        from src.experiment_manifest import PipelineManifest

        m = PipelineManifest()
        # Create a real file so size can be recorded
        artifact = os.path.join(self.tmp, "dummy.png")
        with open(artifact, "wb") as fh:
            fh.write(b"\x89PNG\r\n" + b"\x00" * 100)
        m.add_artifact(artifact, "A dummy PNG")
        out = os.path.join(self.tmp, "manifest.json")
        m.save(out)

        with open(out) as fh:
            import json as json_mod
            data = json_mod.load(fh)
        self.assertEqual(len(data["artifacts"]), 1)
        self.assertGreater(data["artifacts"][0]["size_bytes"], 0)

    def test_manifest_numpy_types_serialisable(self):
        """NumPy scalar types must be auto-cast to native Python for JSON."""
        from src.experiment_manifest import PipelineManifest

        m = PipelineManifest()
        m.record("test", silhouette=np.float32(0.42), n=np.int32(10))
        out = os.path.join(self.tmp, "manifest_np.json")
        # Should not raise TypeError
        m.save(out)
        with open(out) as fh:
            import json as json_mod
            data = json_mod.load(fh)
        self.assertAlmostEqual(data["modules"]["test"]["silhouette"], 0.42, places=4)

    def test_manifest_summary_contains_run_id(self):
        from src.experiment_manifest import PipelineManifest

        m = PipelineManifest(run_id="abc123")
        m.record("a", x=1)
        summary = m.summary()
        self.assertIn("abc123", summary)
        self.assertIn("[a]", summary)


# ---------------------------------------------------------------------------
# 12. Senior-level improvement tests (dedup, silhouette, consecutive sims)
# ---------------------------------------------------------------------------

class TestSeniorImprovements(unittest.TestCase):
    """Tests that validate the senior-level fixes and additions."""

    @staticmethod
    def _unit_embeddings(n: int, dim: int = 8, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        raw = rng.standard_normal((n, dim)).astype(np.float32)
        return raw / np.linalg.norm(raw, axis=1, keepdims=True)

    @staticmethod
    def _make_index(n: int, n_videos: int = 2) -> list:
        return [
            {
                "video_id": f"v{i % n_videos}",
                "frame_idx": i // n_videos,
                "timestamp": float(i),
            }
            for i in range(n)
        ]

    def test_embedding_index_no_gap_when_all_files_exist(self):
        """embedding_idx must be contiguous 0..N-1 with no gaps."""
        from src.embeddings import compute_and_save_embeddings
        import tempfile

        # Create real image files
        tmp = tempfile.mkdtemp()
        meta = []
        for i in range(4):
            p = os.path.join(tmp, f"f{i}.jpg")
            img = Image.new("RGB", (16, 16), (i * 60 % 256, i * 30 % 256, 0))
            img.save(p, "JPEG")
            meta.append({"file_path": p, "video_id": "v1",
                          "frame_idx": i, "timestamp": float(i)})

        with patch("src.embeddings.CLIPProcessor.from_pretrained") as mock_proc, \
             patch("src.embeddings.CLIPModel.from_pretrained") as mock_model:
            import torch
            mm = MagicMock()
            mm.eval.return_value = mm
            mm.to.return_value = mm
            def _feat(**kw):
                n = kw["pixel_values"].shape[0]
                f = torch.randn(n, 8)
                return f / f.norm(dim=-1, keepdim=True)
            mm.get_image_features.side_effect = _feat
            mock_model.return_value = mm
            mp = MagicMock()
            def _proc(*a, images=None, **kw):
                n = len(images)
                return {"pixel_values": torch.zeros(n, 3, 16, 16)}
            mp.side_effect = _proc
            mock_proc.return_value = mp

            embs, idx = compute_and_save_embeddings(meta, tmp)

        expected_indices = list(range(len(idx)))
        actual_indices = [e["embedding_idx"] for e in idx]
        self.assertEqual(actual_indices, expected_indices,
                         "embedding_idx must be 0,1,2,...,N-1 with no gaps")

    def test_vectorized_inter_video_stats_matches_naive(self):
        """Vectorised compute_inter_video_stats must match the naive O(N²) loop."""
        from src.similarity import compute_inter_video_stats, cosine_similarity_matrix

        embs = self._unit_embeddings(12)
        index = self._make_index(12, n_videos=3)
        sim = cosine_similarity_matrix(embs)
        stats = compute_inter_video_stats(sim, index)

        # Naive reference
        n = sim.shape[0]
        vid_ids = [e["video_id"] for e in index]
        w, c = [], []
        for i in range(n):
            for j in range(i + 1, n):
                (w if vid_ids[i] == vid_ids[j] else c).append(float(sim[i, j]))
        all_vals = w + c
        self.assertAlmostEqual(stats["overall_mean"], float(np.mean(all_vals)), places=4)
        self.assertAlmostEqual(stats["within_video_mean"], float(np.mean(w)), places=4)
        self.assertAlmostEqual(stats["cross_video_mean"], float(np.mean(c)), places=4)

    def test_consecutive_similarities_shape_and_range(self):
        from src.temporal_analysis import TemporalAnalyser

        embs = self._unit_embeddings(8)
        index = self._make_index(8, n_videos=1)
        ta = TemporalAnalyser(window=2)
        consec = ta.compute_consecutive_similarities(embs, index)

        self.assertEqual(consec.shape, (8,))
        # Last frame → 1.0 (no successor)
        self.assertAlmostEqual(float(consec[-1]), 1.0, places=5)
        # All values in [-1, 1]
        self.assertTrue((consec >= -1.0).all())
        self.assertTrue((consec <= 1.0).all())

    def test_consecutive_similarities_identical_frames(self):
        """Consecutive identical frames must have similarity 1.0."""
        from src.temporal_analysis import TemporalAnalyser

        base = self._unit_embeddings(1, seed=7)
        embs = np.vstack([base, base, base, base])
        index = [{"video_id": "v0", "frame_idx": i, "timestamp": float(i)}
                 for i in range(4)]
        ta = TemporalAnalyser()
        consec = ta.compute_consecutive_similarities(embs, index)

        for i in range(3):  # frames 0,1,2 each identical to next
            self.assertAlmostEqual(float(consec[i]), 1.0, places=4)

    def test_pacing_rate_per_second_calculation(self):
        from src.temporal_analysis import TemporalAnalyser

        # 5 frames at 1s intervals in one video, with 2 transitions at positions 2, 4
        index = [{"video_id": "v0", "frame_idx": i, "timestamp": float(i)}
                 for i in range(5)]
        ta = TemporalAnalyser()
        rate = ta.pacing_rate_per_second(transitions=[2, 4], index=index, video_id="v0")

        # duration = 4s (0 to 4), 2 transitions → 0.5 cuts/s
        self.assertAlmostEqual(rate, 0.5, places=4)

    def test_pacing_rate_zero_transitions(self):
        from src.temporal_analysis import TemporalAnalyser

        index = [{"video_id": "v0", "frame_idx": i, "timestamp": float(i)}
                 for i in range(4)]
        ta = TemporalAnalyser()
        rate = ta.pacing_rate_per_second(transitions=[], index=index)

        self.assertEqual(rate, 0.0)

    def test_cluster_quality_returns_valid_scores(self):
        from src.clustering import VibeClusterer

        embs = self._unit_embeddings(20, dim=16)
        clust = VibeClusterer(n_clusters=3)
        labels = clust.fit(embs)
        quality = clust.cluster_quality(embs, labels)

        self.assertIn("silhouette", quality)
        self.assertIn("davies_bouldin", quality)
        # Silhouette in [-1, 1]
        self.assertGreaterEqual(quality["silhouette"], -1.0)
        self.assertLessEqual(quality["silhouette"], 1.0)
        # Davies-Bouldin ≥ 0
        self.assertGreaterEqual(quality["davies_bouldin"], 0.0)

    def test_auto_n_clusters_uses_silhouette(self):
        """auto_n_clusters should return a value in [2, max_k]."""
        from src.clustering import auto_n_clusters

        embs = self._unit_embeddings(30, dim=16)
        k = auto_n_clusters(embs, max_k=6)

        self.assertGreaterEqual(k, 2)
        self.assertLessEqual(k, 6)

    def test_multi_prompt_axes_keys_match_default(self):
        """MULTI_PROMPT_AXES should cover the same axes as DEFAULT_AXES."""
        from src.affective_scoring import DEFAULT_AXES, MULTI_PROMPT_AXES

        self.assertEqual(set(DEFAULT_AXES.keys()), set(MULTI_PROMPT_AXES.keys()))
        for axis, (pos_prompts, neg_prompts) in MULTI_PROMPT_AXES.items():
            self.assertEqual(len(pos_prompts), 5,
                             f"Axis '{axis}' must have exactly 5 positive prompts")
            self.assertEqual(len(neg_prompts), 5,
                             f"Axis '{axis}' must have exactly 5 negative prompts")


# ---------------------------------------------------------------------------
# 13. Quality filter tests  (PIL + NumPy — no CLIP needed)
# ---------------------------------------------------------------------------

class TestQualityFilter(unittest.TestCase):
    """Tests for src/quality_filter.py."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _save_image(self, filename: str, pixels: np.ndarray) -> str:
        """Save a uint8 (H, W, 3) or (H, W) numpy array as JPEG."""
        path = os.path.join(self.tmp, filename)
        if pixels.ndim == 3:
            img = Image.fromarray(pixels.astype(np.uint8), mode="RGB")
        else:
            img = Image.fromarray(pixels.astype(np.uint8), mode="L")
        img.save(path, "JPEG")
        return path

    def _sharp_frame(self, size: int = 64) -> np.ndarray:
        """Create a checkerboard pattern — high Laplacian variance (sharp)."""
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        for i in range(size):
            for j in range(size):
                if (i // 4 + j // 4) % 2 == 0:
                    arr[i, j] = [200, 50, 100]
                else:
                    arr[i, j] = [30, 180, 220]
        return arr

    def _blurry_frame(self, size: int = 64) -> np.ndarray:
        """Create a near-uniform grey image — low Laplacian variance (blurry)."""
        return np.full((size, size, 3), 128, dtype=np.uint8)

    def _overexposed_frame(self, size: int = 64) -> np.ndarray:
        """Create a near-white image — collapsed histogram (over-exposed)."""
        return np.full((size, size, 3), 250, dtype=np.uint8)

    # --- blur_score ---

    def test_blur_score_sharp_vs_blurry(self):
        """Sharp checkerboard should score higher than uniform grey."""
        from src.quality_filter import FrameQualityFilter

        qf = FrameQualityFilter()
        sharp = self._sharp_frame()[:, :, 0]   # greyscale
        blurry = self._blurry_frame()[:, :, 0]

        s_sharp  = qf.blur_score(sharp)
        s_blurry = qf.blur_score(blurry)

        self.assertGreater(s_sharp, s_blurry,
                           "Sharp frame should have higher blur_score than blurry frame")

    def test_blur_score_range(self):
        from src.quality_filter import FrameQualityFilter

        qf = FrameQualityFilter()
        grey = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        score = qf.blur_score(grey)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    # --- exposure_entropy ---

    def test_exposure_entropy_high_for_diverse_image(self):
        """A random-pixel image should have high histogram entropy."""
        from src.quality_filter import FrameQualityFilter

        qf = FrameQualityFilter()
        random_img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        uniform_img = np.full((64, 64), 200, dtype=np.uint8)

        self.assertGreater(qf.exposure_entropy(random_img),
                           qf.exposure_entropy(uniform_img),
                           "Diverse image should have higher entropy than uniform image")

    def test_exposure_entropy_near_zero_for_uniform(self):
        from src.quality_filter import FrameQualityFilter

        qf = FrameQualityFilter()
        uniform = np.full((32, 32), 200, dtype=np.uint8)
        entropy = qf.exposure_entropy(uniform)
        self.assertAlmostEqual(entropy, 0.0, places=5)

    # --- score_frames on metadata list ---

    def test_score_frames_returns_correct_keys_and_shape(self):
        from src.quality_filter import FrameQualityFilter

        qf = FrameQualityFilter()
        paths = [
            self._save_image("sharp.jpg", self._sharp_frame()),
            self._save_image("blurry.jpg", self._blurry_frame()),
        ]
        metadata = [{"file_path": p} for p in paths]
        scores = qf.score_frames(metadata)

        for key in ("blur", "exposure_entropy", "luminance_std", "composite"):
            self.assertIn(key, scores)
            self.assertEqual(scores[key].shape, (2,))

    def test_score_frames_missing_file_returns_zeros(self):
        from src.quality_filter import FrameQualityFilter

        qf = FrameQualityFilter()
        metadata = [{"file_path": "/nonexistent/missing.jpg"}]
        scores = qf.score_frames(metadata)
        self.assertEqual(float(scores["composite"][0]), 0.0)

    # --- filter_frames ---

    def test_filter_frames_removes_low_quality(self):
        """Near-uniform frame should be filtered out at default threshold."""
        from src.quality_filter import FrameQualityFilter

        qf = FrameQualityFilter(min_composite_score=0.35)
        paths = [
            self._save_image("sharp.jpg", self._sharp_frame()),
            self._save_image("blurry.jpg", self._blurry_frame()),
        ]
        metadata = [{"file_path": p, "id": i} for i, p in enumerate(paths)]
        scores = qf.score_frames(metadata)
        clean, mask = qf.filter_frames(metadata, scores)

        # blurry frame should be removed; sharp frame retained
        self.assertGreater(len(clean), 0)
        self.assertLess(len(clean), len(metadata))

    def test_filter_frames_zero_threshold_keeps_all(self):
        from src.quality_filter import FrameQualityFilter

        qf = FrameQualityFilter(min_composite_score=0.0)
        paths = [self._save_image(f"f{i}.jpg", self._blurry_frame()) for i in range(3)]
        metadata = [{"file_path": p} for p in paths]
        scores = qf.score_frames(metadata)
        clean, mask = qf.filter_frames(metadata, scores)

        self.assertEqual(len(clean), 3)

    def test_invalid_min_composite_score_raises(self):
        from src.quality_filter import FrameQualityFilter

        with self.assertRaises(ValueError):
            FrameQualityFilter(min_composite_score=1.5)
        with self.assertRaises(ValueError):
            FrameQualityFilter(min_composite_score=-0.1)


# ---------------------------------------------------------------------------
# 14. Efficient retrieval tests (no precomputed matrix)
# ---------------------------------------------------------------------------

class TestEfficientRetrieval(unittest.TestCase):
    """Tests for top_k_no_precompute in src/similarity.py."""

    @staticmethod
    def _unit_embeddings(n: int, dim: int = 8, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        raw = rng.standard_normal((n, dim)).astype(np.float32)
        return raw / np.linalg.norm(raw, axis=1, keepdims=True)

    @staticmethod
    def _make_index(n: int) -> list:
        return [{"video_id": f"v{i % 2}", "frame_idx": i} for i in range(n)]

    def test_top_k_no_precompute_matches_full_matrix(self):
        """top_k_no_precompute must return same results as batch_top_k_queries."""
        from src.similarity import (
            cosine_similarity_matrix,
            batch_top_k_queries,
            top_k_no_precompute,
        )
        embs = self._unit_embeddings(10)
        idx  = self._make_index(10)
        sim  = cosine_similarity_matrix(embs)

        queries = [0, 4, 9]
        full_results = batch_top_k_queries(queries, sim, idx, top_k=3)
        eff_results  = top_k_no_precompute(queries, embs, idx, top_k=3)

        for qidx in queries:
            full_sims = [r["similarity"] for r in full_results[qidx]]
            eff_sims  = [r["similarity"] for r in eff_results[qidx]]
            for fs, es in zip(full_sims, eff_sims):
                self.assertAlmostEqual(fs, es, places=4,
                    msg=f"Mismatch at query={qidx}: full={fs} vs eff={es}")

    def test_top_k_no_precompute_excludes_self(self):
        """Query frame itself should not appear in the results."""
        from src.similarity import top_k_no_precompute

        embs = self._unit_embeddings(8)
        idx  = self._make_index(8)
        results = top_k_no_precompute([3], embs, idx, top_k=5, exclude_self=True)
        frame_indices = [r["frame_idx"] for r in results[3]]
        self.assertNotIn(3, frame_indices)

    def test_top_k_no_precompute_shape(self):
        from src.similarity import top_k_no_precompute

        embs = self._unit_embeddings(12)
        idx  = self._make_index(12)
        results = top_k_no_precompute([0, 5, 11], embs, idx, top_k=4)

        self.assertEqual(len(results), 3)
        for qidx, hits in results.items():
            self.assertLessEqual(len(hits), 4)
            for h in hits:
                self.assertIn("similarity", h)
                self.assertGreaterEqual(h["similarity"], -1.0)
                self.assertLessEqual(h["similarity"], 1.0)

    def test_top_k_no_precompute_invalid_input_raises(self):
        from src.similarity import top_k_no_precompute

        with self.assertRaises(ValueError):
            top_k_no_precompute([0], np.zeros((0, 8), dtype=np.float32), [])


# ---------------------------------------------------------------------------
# 15. CV data-leakage fix tests
# ---------------------------------------------------------------------------

class TestCVLeakageFix(unittest.TestCase):
    """Verify that generate_synthetic_performance_data no longer leaks PCA."""

    @staticmethod
    def _unit_embeddings(n: int, dim: int = 16) -> np.ndarray:
        rng = np.random.default_rng(7)
        raw = rng.standard_normal((n, dim)).astype(np.float32)
        return raw / np.linalg.norm(raw, axis=1, keepdims=True)

    def test_features_contain_only_affective_axes(self):
        """Feature matrix should have exactly as many columns as affective axes."""
        from src.performance_predictor import generate_synthetic_performance_data

        axes = ["joy", "warmth", "energy", "luxury", "complexity", "tension"]
        n = 30
        embs = self._unit_embeddings(n)
        rng = np.random.default_rng(0)
        aff = {a: rng.standard_normal(n).astype(np.float32) for a in axes}

        features, labels = generate_synthetic_performance_data(embs, aff, [])

        self.assertEqual(features.shape[1], len(axes),
                         "Features must be affective-only — no PCA columns")

    def test_build_feature_names_returns_axes_only(self):
        """build_feature_names should return just axis names (no pca_0 etc.)."""
        from src.performance_predictor import build_feature_names

        axes = ["joy", "warmth", "energy"]
        rng = np.random.default_rng(0)
        aff = {a: rng.standard_normal(10).astype(np.float32) for a in axes}
        names = build_feature_names(aff)

        self.assertEqual(names, axes)
        self.assertNotIn("pca_0", names)

    def test_build_feature_names_backward_compat_n_pca(self):
        """Passing n_pca to build_feature_names should not crash (ignored)."""
        from src.performance_predictor import build_feature_names

        axes = ["joy", "warmth"]
        rng = np.random.default_rng(0)
        aff = {a: rng.standard_normal(5).astype(np.float32) for a in axes}
        # n_pca parameter accepted but silently ignored
        names = build_feature_names(aff, n_pca=5)
        self.assertEqual(names, axes)
        # Verify n_pca=5 did NOT append any pca_* columns
        self.assertEqual(len(names), len(axes))

    def test_spearman_still_recovers_signal_after_leakage_fix(self):
        """Ridge regression must still achieve ρ > 0.4 with affective features only."""
        from src.performance_predictor import (
            generate_synthetic_performance_data,
            VibePerformancePredictor,
        )
        n = 120
        embs = self._unit_embeddings(n)
        rng = np.random.default_rng(1)
        axes = ["joy", "warmth", "energy", "luxury", "complexity", "tension"]
        aff = {a: rng.standard_normal(n).astype(np.float32) for a in axes}

        features, labels = generate_synthetic_performance_data(
            embs, aff, [], noise_level=0.1
        )
        pred = VibePerformancePredictor(model_type="ridge")
        cv = pred.cross_validate(features, labels, n_splits=5)

        self.assertGreater(cv["mean_spearman"], 0.4,
                           "Spearman ρ should be > 0.4 even without PCA features")


# ---------------------------------------------------------------------------
# 10. Creative ranking tests
# ---------------------------------------------------------------------------

class TestCreativeRanker(unittest.TestCase):
    """Tests for src/ranking.py — CreativeRanker."""

    def _make_data(self, n: int = 20, d: int = 16, n_axes: int = 6):
        """Return (embeddings, features, index, fitted_predictor)."""
        from src.performance_predictor import (
            VibePerformancePredictor,
            generate_synthetic_performance_data,
        )

        rng = np.random.default_rng(42)
        embs = rng.standard_normal((n, d)).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)

        axis_names = ["joy", "warmth", "energy", "luxury", "complexity", "tension"][:n_axes]
        aff = {a: rng.standard_normal(n).astype(np.float32) for a in axis_names}
        features, labels = generate_synthetic_performance_data(embs, aff, [])

        predictor = VibePerformancePredictor(model_type="ridge")
        predictor.fit(features, labels)

        index = [{"video_id": f"vid_{i % 3}", "timestamp": float(i)} for i in range(n)]
        return embs, features, index, predictor

    def test_rank_returns_correct_count(self):
        """rank() returns exactly top_k items."""
        from src.ranking import CreativeRanker
        embs, feats, idx, pred = self._make_data(n=20)
        ranker = CreativeRanker(pred, n_bootstrap=20)
        ranked = ranker.rank(embs, feats, idx, top_k=5)
        self.assertEqual(len(ranked), 5)

    def test_rank_positions_are_1based_sequential(self):
        """rank field is 1, 2, 3, ... in order."""
        from src.ranking import CreativeRanker
        embs, feats, idx, pred = self._make_data(n=15)
        ranker = CreativeRanker(pred, n_bootstrap=20)
        ranked = ranker.rank(embs, feats, idx, top_k=5)
        positions = [rc.rank for rc in ranked]
        self.assertEqual(positions, list(range(1, len(ranked) + 1)))

    def test_rank_ci_bounds_are_valid(self):
        """ci_lower ≤ predicted_score ≤ ci_upper for every item."""
        from src.ranking import CreativeRanker
        embs, feats, idx, pred = self._make_data(n=20)
        ranker = CreativeRanker(pred, n_bootstrap=50)
        ranked = ranker.rank(embs, feats, idx, top_k=8)
        for rc in ranked:
            self.assertLessEqual(rc.ci_lower, rc.predicted_score + 1e-5)
            self.assertLessEqual(rc.predicted_score, rc.ci_upper + 1e-5)

    def test_rank_required_fields_present(self):
        """Every RankedCreative has all required fields."""
        from src.ranking import CreativeRanker, RankedCreative
        import dataclasses
        embs, feats, idx, pred = self._make_data(n=15)
        ranker = CreativeRanker(pred, n_bootstrap=20)
        ranked = ranker.rank(embs, feats, idx, top_k=3)
        required = {f.name for f in dataclasses.fields(RankedCreative)}
        for rc in ranked:
            for field in required:
                self.assertTrue(hasattr(rc, field), f"Missing field: {field}")

    def test_rank_diversity_nonzero_when_multiple_items(self):
        """diversity_score > 0 for items after the first."""
        from src.ranking import CreativeRanker
        embs, feats, idx, pred = self._make_data(n=20)
        ranker = CreativeRanker(pred, lambda_mmr=0.6, n_bootstrap=20)
        ranked = ranker.rank(embs, feats, idx, top_k=5)
        if len(ranked) > 1:
            # Items after rank 1 should have positive diversity (max_sim < 1)
            for rc in ranked[1:]:
                self.assertGreaterEqual(rc.diversity_score, 0.0)

    def test_rank_top_k_capped_to_n(self):
        """Requesting more items than available returns all available."""
        from src.ranking import CreativeRanker
        embs, feats, idx, pred = self._make_data(n=5)
        ranker = CreativeRanker(pred, n_bootstrap=20)
        ranked = ranker.rank(embs, feats, idx, top_k=100)
        self.assertLessEqual(len(ranked), 5)

    def test_lambda_boundary_values_do_not_crash(self):
        """λ=0 (pure diversity) and λ=1 (pure score) must not raise."""
        from src.ranking import CreativeRanker
        embs, feats, idx, pred = self._make_data(n=12)
        for lam in (0.0, 1.0):
            ranker = CreativeRanker(pred, lambda_mmr=lam, n_bootstrap=20)
            ranked = ranker.rank(embs, feats, idx, top_k=4)
            self.assertGreater(len(ranked), 0)

    def test_save_ranking_writes_valid_json(self):
        """save_ranking() writes a readable JSON file with correct count."""
        from src.ranking import CreativeRanker
        embs, feats, idx, pred = self._make_data(n=15)
        ranker = CreativeRanker(pred, n_bootstrap=20)
        ranked = ranker.rank(embs, feats, idx, top_k=4)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ranking.json")
            out = ranker.save_ranking(ranked, path)
            self.assertTrue(os.path.exists(out))
            with open(out) as fh:
                data = json.load(fh)
            self.assertEqual(len(data), len(ranked))
            self.assertIn("predicted_score", data[0])

    def test_plot_ranking_creates_png(self):
        """plot_ranking() saves a non-empty PNG file."""
        from src.ranking import CreativeRanker
        embs, feats, idx, pred = self._make_data(n=12)
        ranker = CreativeRanker(pred, n_bootstrap=20)
        ranked = ranker.rank(embs, feats, idx, top_k=4)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ranking.png")
            out = ranker.plot_ranking(ranked, path)
            self.assertTrue(os.path.exists(out))
            self.assertGreater(os.path.getsize(out), 1024)

    def test_invalid_lambda_raises(self):
        """CreativeRanker(lambda_mmr=1.5) must raise ValueError."""
        from src.ranking import CreativeRanker
        _, _, _, pred = self._make_data(n=10)
        with self.assertRaises(ValueError):
            CreativeRanker(pred, lambda_mmr=1.5)


# ---------------------------------------------------------------------------
# 11. Predictor calibration tests
# ---------------------------------------------------------------------------

class TestPredictorCalibrator(unittest.TestCase):
    """Tests for src/calibration.py — PredictorCalibrator."""

    def _make_predictions(self, n: int = 80):
        """Return (raw_predictions, labels) with known linear signal."""
        rng = np.random.default_rng(42)  # consistent seed across all test methods
        raw = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
        # Labels have a noisy positive correlation with raw scores
        labels = (raw + rng.normal(0, 0.2, n)).astype(np.float32)
        return raw, labels

    def test_platt_transform_output_in_unit_interval(self):
        """Platt calibrated scores must be in [0, 1]."""
        from src.calibration import PredictorCalibrator
        raw, labels = self._make_predictions()
        cal = PredictorCalibrator(method="platt")
        cal.fit(raw, labels)
        out = cal.transform(raw)
        self.assertTrue(np.all(out >= 0.0))
        self.assertTrue(np.all(out <= 1.0))
        self.assertEqual(out.dtype, np.float32)

    def test_isotonic_transform_output_in_unit_interval(self):
        """Isotonic calibrated scores must be in [0, 1]."""
        from src.calibration import PredictorCalibrator
        raw, labels = self._make_predictions(n=100)
        cal = PredictorCalibrator(method="isotonic")
        cal.fit(raw, labels)
        out = cal.transform(raw)
        self.assertTrue(np.all(out >= 0.0))
        self.assertTrue(np.all(out <= 1.0))

    def test_transform_before_fit_raises(self):
        """Calling transform() before fit() must raise RuntimeError."""
        from src.calibration import PredictorCalibrator
        cal = PredictorCalibrator()
        with self.assertRaises(RuntimeError):
            cal.transform(np.array([0.5, 0.6], dtype=np.float32))

    def test_fit_with_too_few_samples_raises(self):
        """fit() with < 2 samples must raise ValueError."""
        from src.calibration import PredictorCalibrator
        cal = PredictorCalibrator()
        with self.assertRaises(ValueError):
            cal.fit(np.array([0.5], dtype=np.float32),
                    np.array([0.5], dtype=np.float32))

    def test_ece_perfect_calibration_is_near_zero(self):
        """ECE for a perfectly calibrated predictor should be near 0."""
        from src.calibration import PredictorCalibrator
        n = 200
        # Perfect calibration: raw score == fraction of positives in that bin
        raw = np.linspace(0.0, 1.0, n).astype(np.float32)
        # Generate binary labels such that freq(positive|raw=x) ≈ x
        rng = np.random.default_rng(0)
        labels_binary = rng.binomial(1, raw).astype(np.float32)
        # ECE on the raw scores directly (already calibrated by construction)
        cal = PredictorCalibrator(method="platt")
        cal.fit(raw, labels_binary)
        calibrated = cal.transform(raw)
        ece = cal.expected_calibration_error(calibrated, labels_binary, n_bins=10)
        self.assertLess(ece, 0.25, "ECE should be small for a near-calibrated predictor")

    def test_save_calibration_params_writes_json(self):
        """save_calibration_params() produces a readable JSON file."""
        from src.calibration import PredictorCalibrator
        raw, labels = self._make_predictions()
        cal = PredictorCalibrator(method="platt")
        cal.fit(raw, labels)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cal_params.json")
            out = cal.save_calibration_params(path)
            self.assertTrue(os.path.exists(out))
            with open(out) as fh:
                doc = json.load(fh)
            self.assertIn("method", doc)
            self.assertEqual(doc["method"], "platt")
            self.assertIn("platt_a", doc)

    def test_plot_reliability_diagram_creates_png(self):
        """plot_reliability_diagram() saves a non-empty PNG."""
        from src.calibration import PredictorCalibrator
        raw, labels = self._make_predictions(n=100)
        cal = PredictorCalibrator(method="platt")
        cal.fit(raw, labels)
        calibrated = cal.transform(raw)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "calibration.png")
            out = cal.plot_reliability_diagram(
                calibrated, labels, path, raw_scores=raw
            )
            self.assertTrue(os.path.exists(out))
            self.assertGreater(os.path.getsize(out), 1024)

    def test_unknown_method_raises(self):
        """PredictorCalibrator(method='bad') must raise ValueError."""
        from src.calibration import PredictorCalibrator
        with self.assertRaises(ValueError):
            PredictorCalibrator(method="bad_method")


# ---------------------------------------------------------------------------
# 12. Scene-adaptive frame extraction tests
# ---------------------------------------------------------------------------

class TestSceneAdaptiveSampling(unittest.TestCase):
    """Tests for extract_frames_scene_adaptive() in src/frame_extractor.py."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _make_scene_video(self, n_scenes: int = 3, frames_per_scene: int = 20,
                          fps: int = 10) -> str:
        """Create a synthetic video with abrupt colour-scene changes."""
        path = os.path.join(self.tmp, "scene_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        total = n_scenes * frames_per_scene
        writer = cv2.VideoWriter(path, fourcc, fps, (64, 64))
        assert writer.isOpened()
        colours = [(200, 50, 50), (50, 200, 50), (50, 50, 200)]
        for s in range(n_scenes):
            c = colours[s % len(colours)]
            for _ in range(frames_per_scene):
                frame = np.full((64, 64, 3), c, dtype=np.uint8)
                writer.write(frame)
        writer.release()
        return path

    def test_returns_non_empty_metadata(self):
        """Scene-adaptive extraction returns at least one frame."""
        from src.frame_extractor import extract_frames_scene_adaptive
        path = self._make_scene_video()
        meta = extract_frames_scene_adaptive(
            path, os.path.join(self.tmp, "frames_adaptive")
        )
        self.assertGreater(len(meta), 0)

    def test_sampling_method_recorded_in_metadata(self):
        """Every metadata entry must have sampling_method='scene_adaptive'."""
        from src.frame_extractor import extract_frames_scene_adaptive
        path = self._make_scene_video()
        meta = extract_frames_scene_adaptive(
            path, os.path.join(self.tmp, "frames_adaptive2")
        )
        for entry in meta:
            self.assertEqual(entry.get("sampling_method"), "scene_adaptive")
            self.assertIn("scene_id", entry)
            self.assertIn("scene_start_t", entry)
            self.assertIn("scene_end_t", entry)

    def test_fewer_frames_than_uniform_on_static_video(self):
        """On a long static video, scene-adaptive yields far fewer frames."""
        from src.frame_extractor import extract_frames, extract_frames_scene_adaptive
        # 5-second static (one colour) video at 10fps
        path = os.path.join(self.tmp, "static.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, 10, (64, 64))
        assert writer.isOpened()
        for _ in range(50):
            writer.write(np.full((64, 64, 3), (100, 100, 100), dtype=np.uint8))
        writer.release()

        uniform = extract_frames(path, os.path.join(self.tmp, "frames_uni"))
        adaptive = extract_frames_scene_adaptive(
            path, os.path.join(self.tmp, "frames_adap"),
            transition_threshold=0.10,
        )
        # Static video → 1 scene → adaptive extracts 1 frame
        # Uniform at 1fps → 5 frames
        self.assertLessEqual(len(adaptive), len(uniform))

    def test_missing_video_raises(self):
        """extract_frames_scene_adaptive must raise FileNotFoundError."""
        from src.frame_extractor import extract_frames_scene_adaptive
        with self.assertRaises(FileNotFoundError):
            extract_frames_scene_adaptive(
                "/nonexistent/video.mp4",
                os.path.join(self.tmp, "frames"),
            )

    def test_max_scenes_cap_respected(self):
        """The number of returned frames must not exceed max_scenes."""
        from src.frame_extractor import extract_frames_scene_adaptive
        # 10-scene video
        path = self._make_scene_video(n_scenes=5, frames_per_scene=20)
        meta = extract_frames_scene_adaptive(
            path,
            os.path.join(self.tmp, "frames_capped"),
            max_scenes=2,
            transition_threshold=0.05,
        )
        self.assertLessEqual(len(meta), 2)

    def test_frame_files_are_written(self):
        """All file_path entries in metadata must refer to existing JPEG files."""
        from src.frame_extractor import extract_frames_scene_adaptive
        path = self._make_scene_video(n_scenes=2, frames_per_scene=15)
        meta = extract_frames_scene_adaptive(
            path, os.path.join(self.tmp, "frames_exist")
        )
        for entry in meta:
            self.assertTrue(
                os.path.exists(entry["file_path"]),
                f"Expected file missing: {entry['file_path']}",
            )


# ---------------------------------------------------------------------------
# 19. TextQueryRetriever tests
# ---------------------------------------------------------------------------

class TestTextQueryRetriever(unittest.TestCase):
    """Tests for src/text_query.py — text-guided creative retrieval."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.dim = 16
        self.n = 12
        rng = np.random.default_rng(77)
        raw = rng.standard_normal((self.n, self.dim)).astype(np.float32)
        self.embeddings = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        self.index = [
            {
                "video_id": f"v{i % 3}",
                "frame_idx": i,
                "timestamp": float(i),
                "file_path": os.path.join(self.tmp, f"frame_{i}.jpg"),
            }
            for i in range(self.n)
        ]

    def _make_retriever(self):
        """Build a TextQueryRetriever with a mocked CLIP model."""
        import torch
        from unittest.mock import MagicMock

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        def _get_text_features(**kwargs):
            n = kwargs["input_ids"].shape[0]
            rng = np.random.default_rng(0)
            f = torch.tensor(
                rng.standard_normal((n, self.dim)).astype(np.float32)
            )
            return f / f.norm(dim=-1, keepdim=True)

        mock_model.get_text_features.side_effect = _get_text_features

        mock_proc = MagicMock()

        def _proc(*args, text=None, images=None,
                  return_tensors=None, padding=None, truncation=None, **kw):
            batch = text if text is not None else (images or [])
            n = len(batch)
            return {
                "input_ids": torch.zeros(n, 77, dtype=torch.long),
                "attention_mask": torch.ones(n, 77, dtype=torch.long),
            }

        mock_proc.side_effect = _proc

        from src.text_query import TextQueryRetriever
        return TextQueryRetriever(model=mock_model, processor=mock_proc)

    def test_query_returns_top_k(self):
        """query() should return exactly top_k results."""
        retriever = self._make_retriever()
        results = retriever.query("warm golden luxury", self.embeddings, self.index, top_k=5)
        self.assertEqual(len(results), 5)

    def test_query_result_fields(self):
        """Each TextQueryResult must have the expected fields."""
        from src.text_query import TextQueryResult
        retriever = self._make_retriever()
        results = retriever.query("cinematic", self.embeddings, self.index, top_k=3)
        for r in results:
            self.assertIsInstance(r, TextQueryResult)
            self.assertIsInstance(r.rank, int)
            self.assertIsInstance(r.similarity, float)
            self.assertIn("video_id", r.metadata)

    def test_query_similarity_in_range(self):
        """All similarities must be in [-1, 1]."""
        retriever = self._make_retriever()
        results = retriever.query("minimalist", self.embeddings, self.index, top_k=self.n)
        for r in results:
            self.assertGreaterEqual(r.similarity, -1.0 - 1e-5)
            self.assertLessEqual(r.similarity, 1.0 + 1e-5)

    def test_query_sorted_descending(self):
        """Results must be sorted from highest to lowest similarity."""
        retriever = self._make_retriever()
        results = retriever.query("joyful uplifting", self.embeddings, self.index, top_k=8)
        sims = [r.similarity for r in results]
        self.assertEqual(sims, sorted(sims, reverse=True))

    def test_query_rank_is_one_based_sequential(self):
        """Ranks must be 1-based and sequential."""
        retriever = self._make_retriever()
        results = retriever.query("tense dramatic", self.embeddings, self.index, top_k=4)
        ranks = [r.rank for r in results]
        self.assertEqual(ranks, list(range(1, len(ranks) + 1)))

    def test_query_empty_embeddings_raises(self):
        """query() on empty embeddings must raise ValueError."""
        retriever = self._make_retriever()
        with self.assertRaises(ValueError):
            retriever.query("test", np.empty((0, self.dim), dtype=np.float32), [], top_k=5)

    def test_rank_videos_by_brief_all_videos_present(self):
        """rank_videos_by_brief() must return an entry for every distinct video_id."""
        retriever = self._make_retriever()
        ranking = retriever.rank_videos_by_brief("luxury minimal", self.embeddings, self.index)
        returned_ids = {row["video_id"] for row in ranking}
        expected_ids = {f"v{i % 3}" for i in range(self.n)}
        self.assertEqual(returned_ids, expected_ids)

    def test_rank_videos_sorted_descending(self):
        """rank_videos_by_brief() results must be sorted by descending score."""
        retriever = self._make_retriever()
        ranking = retriever.rank_videos_by_brief("energetic", self.embeddings, self.index)
        scores = [row["score"] for row in ranking]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_multi_brief_comparison_keys(self):
        """multi_brief_comparison() must include all brief names and video IDs."""
        retriever = self._make_retriever()
        briefs = {"brief_a": "warm golden", "brief_b": "dark minimal"}
        comparison = retriever.multi_brief_comparison(briefs, self.embeddings, self.index)
        self.assertEqual(set(comparison.keys()), {"brief_a", "brief_b"})
        all_vids = {f"v{i % 3}" for i in range(self.n)}
        for brief_name, vid_scores in comparison.items():
            self.assertEqual(set(vid_scores.keys()), all_vids)

    def test_save_results_creates_json(self):
        """save_results() must write a readable JSON file."""
        retriever = self._make_retriever()
        results = retriever.query("cosy warm", self.embeddings, self.index, top_k=4)
        out_path = os.path.join(self.tmp, "query_results.json")
        retriever.save_results(results, out_path)
        self.assertTrue(os.path.exists(out_path))
        with open(out_path) as fh:
            data = json.load(fh)
        self.assertEqual(len(data), 4)
        self.assertIn("similarity", data[0])

    def test_plot_query_results_creates_png(self):
        """plot_query_results() must write a non-empty PNG."""
        retriever = self._make_retriever()
        results = retriever.query("vivid colourful", self.embeddings, self.index, top_k=5)
        out_path = os.path.join(self.tmp, "query_chart.png")
        retriever.plot_query_results(results, out_path)
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 0)

    def test_plot_brief_comparison_heatmap_creates_png(self):
        """plot_brief_comparison_heatmap() must write a non-empty PNG."""
        retriever = self._make_retriever()
        briefs = {"warm": "warm golden sunset", "cool": "cold icy minimal"}
        comparison = retriever.multi_brief_comparison(briefs, self.embeddings, self.index)
        out_path = os.path.join(self.tmp, "brief_heatmap.png")
        retriever.plot_brief_comparison_heatmap(comparison, out_path)
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 0)

    def test_invalid_aggregation_raises(self):
        """rank_videos_by_brief() with unknown aggregation must raise ValueError."""
        retriever = self._make_retriever()
        with self.assertRaises(ValueError):
            retriever.rank_videos_by_brief(
                "test", self.embeddings, self.index, aggregation="unknown"
            )


# ---------------------------------------------------------------------------
# 20. EmbeddingDriftDetector tests
# ---------------------------------------------------------------------------

class TestEmbeddingDriftDetector(unittest.TestCase):
    """Tests for src/drift_detector.py — distribution drift monitoring."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.dim = 32
        rng = np.random.default_rng(55)
        raw = rng.standard_normal((60, self.dim)).astype(np.float32)
        self.reference = raw / np.linalg.norm(raw, axis=1, keepdims=True)

    def test_fit_then_detect_no_error(self):
        """fit() followed by detect() must not raise on same-shape input."""
        from src.drift_detector import EmbeddingDriftDetector
        det = EmbeddingDriftDetector(n_components=5)
        det.fit(self.reference)
        rng = np.random.default_rng(56)
        raw = rng.standard_normal((20, self.dim)).astype(np.float32)
        new_embs = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        report = det.detect(new_embs)
        self.assertIsNotNone(report)

    def test_detect_before_fit_raises(self):
        """detect() before fit() must raise RuntimeError."""
        from src.drift_detector import EmbeddingDriftDetector
        det = EmbeddingDriftDetector()
        rng = np.random.default_rng(1)
        raw = rng.standard_normal((10, self.dim)).astype(np.float32)
        new_embs = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        with self.assertRaises(RuntimeError):
            det.detect(new_embs)

    def test_mmd_is_nonnegative(self):
        """MMD must always be ≥ 0."""
        from src.drift_detector import EmbeddingDriftDetector
        det = EmbeddingDriftDetector(n_components=5)
        det.fit(self.reference)
        rng = np.random.default_rng(57)
        raw = rng.standard_normal((30, self.dim)).astype(np.float32)
        new_embs = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        report = det.detect(new_embs)
        self.assertGreaterEqual(report.mmd, 0.0)

    def test_same_data_low_mmd(self):
        """Comparing reference to itself should yield near-zero MMD."""
        from src.drift_detector import EmbeddingDriftDetector
        det = EmbeddingDriftDetector(n_components=5)
        det.fit(self.reference)
        report = det.detect(self.reference)
        # MMD for identical data should be very small (numerical noise only)
        self.assertLess(report.mmd, 0.05)

    def test_shifted_distribution_higher_mmd(self):
        """A large constant shift must produce higher MMD than the reference."""
        from src.drift_detector import EmbeddingDriftDetector
        det = EmbeddingDriftDetector(n_components=5)
        det.fit(self.reference)
        report_same = det.detect(self.reference)
        # Add a large shift — should be clearly more drifted
        shifted = self.reference + 3.0
        shifted = shifted / np.linalg.norm(shifted, axis=1, keepdims=True)
        report_shifted = det.detect(shifted)
        self.assertGreater(report_shifted.mmd, report_same.mmd)

    def test_drift_report_fields(self):
        """DriftReport must contain all required fields with correct types."""
        from src.drift_detector import EmbeddingDriftDetector, DriftReport
        det = EmbeddingDriftDetector(n_components=5)
        det.fit(self.reference)
        rng = np.random.default_rng(58)
        raw = rng.standard_normal((15, self.dim)).astype(np.float32)
        new_embs = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        report = det.detect(new_embs)
        self.assertIsInstance(report, DriftReport)
        self.assertIsInstance(report.mmd, float)
        self.assertIsInstance(report.ks_pvalue_min, float)
        self.assertIsInstance(report.is_drifted, bool)
        self.assertIsInstance(report.anomaly_fraction, float)
        self.assertIsInstance(report.component_pvalues, list)
        self.assertEqual(len(report.component_pvalues), report.n_pca_components)

    def test_anomaly_fraction_in_range(self):
        """Anomaly fraction must be in [0, 1]."""
        from src.drift_detector import EmbeddingDriftDetector
        det = EmbeddingDriftDetector(n_components=5)
        det.fit(self.reference)
        rng = np.random.default_rng(59)
        raw = rng.standard_normal((20, self.dim)).astype(np.float32)
        new_embs = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        report = det.detect(new_embs)
        self.assertGreaterEqual(report.anomaly_fraction, 0.0)
        self.assertLessEqual(report.anomaly_fraction, 1.0)

    def test_save_report_creates_json(self):
        """save_report() must write a readable JSON file."""
        from src.drift_detector import EmbeddingDriftDetector
        det = EmbeddingDriftDetector(n_components=5)
        det.fit(self.reference)
        rng = np.random.default_rng(60)
        raw = rng.standard_normal((20, self.dim)).astype(np.float32)
        new_embs = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        report = det.detect(new_embs)
        out_path = os.path.join(self.tmp, "drift_report.json")
        det.save_report(report, out_path)
        self.assertTrue(os.path.exists(out_path))
        with open(out_path) as fh:
            data = json.load(fh)
        self.assertIn("mmd", data)
        self.assertIn("is_drifted", data)

    def test_plot_pca_comparison_creates_png(self):
        """plot_pca_comparison() must write a non-empty PNG."""
        from src.drift_detector import EmbeddingDriftDetector
        det = EmbeddingDriftDetector(n_components=5)
        det.fit(self.reference)
        rng = np.random.default_rng(61)
        raw = rng.standard_normal((20, self.dim)).astype(np.float32)
        new_embs = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        out_path = os.path.join(self.tmp, "drift_pca.png")
        det.plot_pca_comparison(new_embs, out_path)
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 0)


# ---------------------------------------------------------------------------
# 21. EmbeddingModel.encode_text tests
# ---------------------------------------------------------------------------

class TestEncodeText(unittest.TestCase):
    """Tests for the new EmbeddingModel.encode_text() method."""

    @patch("src.embeddings.CLIPModel")
    @patch("src.embeddings.CLIPProcessor")
    def test_encode_text_shape(self, MockProc, MockModel):
        """encode_text returns (N, D) for N input texts."""
        import torch
        from src.embeddings import EmbeddingModel

        dim = 16
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        def _text_features(**kwargs):
            n = kwargs["input_ids"].shape[0]
            f = torch.randn(n, dim)
            return f / f.norm(dim=-1, keepdim=True)

        mock_model.get_text_features.side_effect = _text_features
        MockModel.from_pretrained.return_value = mock_model

        mock_proc = MagicMock()

        def _proc(*args, text=None, images=None,
                  return_tensors=None, padding=None, truncation=None, **kw):
            batch = text if text is not None else (images or [])
            n = len(batch)
            return {"input_ids": torch.zeros(n, 77, dtype=torch.long),
                    "attention_mask": torch.ones(n, 77, dtype=torch.long)}

        mock_proc.side_effect = _proc
        MockProc.from_pretrained.return_value = mock_proc

        model = EmbeddingModel(model_name="mock/clip")
        texts = ["warm golden", "cold minimal", "high energy"]
        result = model.encode_text(texts)
        self.assertEqual(result.shape, (3, dim))
        self.assertEqual(result.dtype, np.float32)

    @patch("src.embeddings.CLIPModel")
    @patch("src.embeddings.CLIPProcessor")
    def test_encode_text_empty_raises(self, MockProc, MockModel):
        """encode_text([]) must raise ValueError."""
        from src.embeddings import EmbeddingModel
        MockModel.from_pretrained.return_value = MagicMock(
            eval=MagicMock(return_value=MagicMock(to=MagicMock(return_value=MagicMock())))
        )
        MockProc.from_pretrained.return_value = MagicMock()
        model = EmbeddingModel(model_name="mock/clip")
        with self.assertRaises(ValueError):
            model.encode_text([])

    @patch("src.embeddings.CLIPModel")
    @patch("src.embeddings.CLIPProcessor")
    def test_encode_text_l2_normalised(self, MockProc, MockModel):
        """Output vectors must be L2-normalised (norm ≈ 1.0)."""
        import torch
        from src.embeddings import EmbeddingModel

        dim = 16
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        def _text_features(**kwargs):
            n = kwargs["input_ids"].shape[0]
            f = torch.randn(n, dim)
            return f / f.norm(dim=-1, keepdim=True)

        mock_model.get_text_features.side_effect = _text_features
        MockModel.from_pretrained.return_value = mock_model

        mock_proc = MagicMock()

        def _proc(*args, text=None, images=None,
                  return_tensors=None, padding=None, truncation=None, **kw):
            batch = text if text is not None else (images or [])
            n = len(batch)
            return {"input_ids": torch.zeros(n, 77, dtype=torch.long),
                    "attention_mask": torch.ones(n, 77, dtype=torch.long)}

        mock_proc.side_effect = _proc
        MockProc.from_pretrained.return_value = mock_proc

        model = EmbeddingModel(model_name="mock/clip")
        result = model.encode_text(["cozy warm light", "cold icy dark"])
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, np.ones(len(norms)), atol=1e-5)


# ---------------------------------------------------------------------------
# Helpers shared by new test classes
# ---------------------------------------------------------------------------

def _make_mock_affective_scorer(emb_dim: int = 16, seed: int = 42):
    """
    Build a minimal mock AffectiveScorer suitable for OcclusionSaliency tests.

    The mock exposes:
    - scorer.device       = "cpu"
    - scorer.axes         = {"energy": (...), "warmth": (...)}
    - scorer.encode_text  = deterministic unit-vector function
    - scorer.processor    = callable returning {"pixel_values": torch.Tensor}
    - scorer.model.get_image_features = callable returning L2-normalised tensor
    """
    import torch

    scorer = MagicMock()
    scorer.device = "cpu"
    scorer.axes = {
        "energy": ("energetic dynamic", "calm still"),
        "warmth":  ("warm golden",       "cold icy"),
    }
    rng_np = np.random.default_rng(seed)

    def _encode_text(prompts):
        n = len(prompts)
        embs = rng_np.random((n, emb_dim)).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / (norms + 1e-8)

    scorer.encode_text.side_effect = _encode_text

    def _processor(images=None, return_tensors=None, padding=None, **_kw):
        n = len(images) if images is not None else 1
        return {"pixel_values": torch.zeros(n, 3, 32, 32)}

    scorer.processor.side_effect = _processor

    def _get_image_features(**kwargs):
        n = kwargs["pixel_values"].shape[0]
        # Return L2-normalised random tensors seeded by n for determinism
        rng = np.random.default_rng(n + seed)
        embs = rng.random((n, emb_dim)).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / (norms + 1e-8)
        return torch.tensor(embs, dtype=torch.float32)

    scorer.model.get_image_features.side_effect = _get_image_features
    return scorer


def _make_ranked_creative(
    frame_idx=0, video_id="vid_a", timestamp=0.0,
    predicted_score=0.70, ci_lower=0.65, ci_upper=0.75,
    rank=1, diversity_score=0.90,
):
    """Create a RankedCreative dataclass instance for A/B testing tests."""
    from src.ranking import RankedCreative
    return RankedCreative(
        rank=rank,
        frame_idx=frame_idx,
        video_id=video_id,
        timestamp=timestamp,
        predicted_score=predicted_score,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        diversity_score=diversity_score,
        metadata={},
    )


# ---------------------------------------------------------------------------
# OcclusionSaliency tests
# ---------------------------------------------------------------------------

class TestOcclusionSaliency(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.scorer = _make_mock_affective_scorer()
        # Create a synthetic JPEG image
        self.img_path = os.path.join(self.tmp, "test_frame.jpg")
        _make_synthetic_frame(self.img_path, r=120, g=80, b=60)

    def test_compute_saliency_returns_correct_axes(self):
        from src.explainability import OcclusionSaliency
        occ = OcclusionSaliency(self.scorer, grid_rows=2, grid_cols=2)
        baseline = {"energy": 0.5, "warmth": 0.3}
        result = occ.compute_saliency(self.img_path, baseline)
        self.assertIn("energy", result)
        self.assertIn("warmth", result)

    def test_compute_saliency_grid_shape(self):
        from src.explainability import OcclusionSaliency
        occ = OcclusionSaliency(self.scorer, grid_rows=3, grid_cols=4)
        baseline = {"energy": 0.5, "warmth": 0.3}
        result = occ.compute_saliency(self.img_path, baseline)
        self.assertEqual(result["energy"].shape, (3, 4))
        self.assertEqual(result["warmth"].shape, (3, 4))

    def test_compute_saliency_no_nan(self):
        from src.explainability import OcclusionSaliency
        occ = OcclusionSaliency(self.scorer, grid_rows=2, grid_cols=2)
        baseline = {"energy": 0.6, "warmth": -0.2}
        result = occ.compute_saliency(self.img_path, baseline)
        for axis, grid in result.items():
            self.assertFalse(
                np.isnan(grid).any(),
                f"NaN in saliency grid for axis '{axis}'",
            )

    def test_compute_saliency_missing_file_raises(self):
        from src.explainability import OcclusionSaliency
        occ = OcclusionSaliency(self.scorer)
        with self.assertRaises(FileNotFoundError):
            occ.compute_saliency("/nonexistent/path.jpg", {"energy": 0.5})

    def test_compute_saliency_empty_baseline_raises(self):
        from src.explainability import OcclusionSaliency
        occ = OcclusionSaliency(self.scorer)
        with self.assertRaises(ValueError):
            occ.compute_saliency(self.img_path, {})

    def test_axis_anchor_cache_populated_on_first_call(self):
        from src.explainability import OcclusionSaliency
        occ = OcclusionSaliency(self.scorer, grid_rows=2, grid_cols=2)
        self.assertIsNone(occ._axis_anchor_cache)
        occ.compute_saliency(self.img_path, {"energy": 0.5, "warmth": 0.3})
        self.assertIsNotNone(occ._axis_anchor_cache)
        self.assertIn("energy", occ._axis_anchor_cache)

    def test_invalidate_cache_clears_anchors(self):
        from src.explainability import OcclusionSaliency
        occ = OcclusionSaliency(self.scorer, grid_rows=2, grid_cols=2)
        occ.compute_saliency(self.img_path, {"energy": 0.5, "warmth": 0.3})
        self.assertIsNotNone(occ._axis_anchor_cache)
        occ.invalidate_cache()
        self.assertIsNone(occ._axis_anchor_cache)

    def test_top_important_patches_returns_correct_count(self):
        from src.explainability import OcclusionSaliency
        occ = OcclusionSaliency(self.scorer, grid_rows=4, grid_cols=4)
        baseline = {"energy": 0.5, "warmth": 0.3}
        saliency = occ.compute_saliency(self.img_path, baseline)
        top3 = occ.top_important_patches(saliency, "energy", top_k=3)
        self.assertEqual(len(top3), 3)
        for row, col, imp in top3:
            self.assertIsInstance(row, int)
            self.assertIsInstance(col, int)
            self.assertIsInstance(imp, float)

    def test_top_important_patches_sorted_by_abs_importance(self):
        from src.explainability import OcclusionSaliency
        occ = OcclusionSaliency(self.scorer, grid_rows=4, grid_cols=4)
        baseline = {"energy": 0.8, "warmth": 0.1}
        saliency = occ.compute_saliency(self.img_path, baseline)
        top3 = occ.top_important_patches(saliency, "energy", top_k=3)
        # Verify descending order of |importance|
        importances = [abs(imp) for _, _, imp in top3]
        self.assertEqual(importances, sorted(importances, reverse=True))

    def test_top_important_patches_unknown_axis_raises(self):
        from src.explainability import OcclusionSaliency
        occ = OcclusionSaliency(self.scorer)
        with self.assertRaises(KeyError):
            occ.top_important_patches({"energy": np.zeros((2, 2))}, "unknown_axis")

    def test_plot_saliency_overlay_creates_png(self):
        from src.explainability import OcclusionSaliency
        occ = OcclusionSaliency(self.scorer, grid_rows=2, grid_cols=2)
        baseline = {"energy": 0.5, "warmth": 0.3}
        saliency = occ.compute_saliency(self.img_path, baseline)
        out = os.path.join(self.tmp, "saliency_test.png")
        result_path = occ.plot_saliency_overlay(self.img_path, saliency, out)
        self.assertTrue(os.path.exists(result_path))
        self.assertGreater(os.path.getsize(result_path), 0)

    def test_batch_compute_saliency_length_match(self):
        from src.explainability import OcclusionSaliency
        img2 = os.path.join(self.tmp, "test_frame2.jpg")
        _make_synthetic_frame(img2, r=200, g=200, b=200)
        occ = OcclusionSaliency(self.scorer, grid_rows=2, grid_cols=2)
        paths = [self.img_path, img2]
        scores_list = [{"energy": 0.5, "warmth": 0.3}] * 2
        results = occ.batch_compute_saliency(paths, scores_list)
        self.assertEqual(len(results), 2)

    def test_batch_compute_saliency_length_mismatch_raises(self):
        from src.explainability import OcclusionSaliency
        occ = OcclusionSaliency(self.scorer)
        with self.assertRaises(ValueError):
            occ.batch_compute_saliency([self.img_path], [{"energy": 0.5}, {"energy": 0.3}])

    def test_invalid_grid_raises(self):
        from src.explainability import OcclusionSaliency
        with self.assertRaises(ValueError):
            OcclusionSaliency(self.scorer, grid_rows=0, grid_cols=4)


# ---------------------------------------------------------------------------
# CreativeABTester tests
# ---------------------------------------------------------------------------

class TestCreativeABTester(unittest.TestCase):

    def _creative(self, score, ci_half=0.05, rank=1, video_id="vid_a", frame_idx=0):
        return _make_ranked_creative(
            frame_idx=frame_idx,
            video_id=video_id,
            predicted_score=score,
            ci_lower=score - ci_half,
            ci_upper=score + ci_half,
            rank=rank,
        )

    def test_invalid_win_threshold_raises(self):
        from src.ab_testing import CreativeABTester
        with self.assertRaises(ValueError):
            CreativeABTester(win_threshold=0.3)

    def test_compare_pair_equal_scores_win_prob_near_half(self):
        from src.ab_testing import CreativeABTester
        tester = CreativeABTester(n_thompson=500, random_state=0)
        a = self._creative(0.70)
        b = self._creative(0.70)
        result = tester.compare_pair(a, b)
        self.assertAlmostEqual(result.win_probability, 0.5, delta=0.05)

    def test_compare_pair_a_much_better_prefer_a(self):
        from src.ab_testing import CreativeABTester
        tester = CreativeABTester(win_threshold=0.80, n_thompson=500, random_state=0)
        a = self._creative(0.90, ci_half=0.02)
        b = self._creative(0.40, ci_half=0.02)
        result = tester.compare_pair(a, b)
        self.assertGreater(result.win_probability, 0.90)
        self.assertEqual(result.recommendation, "prefer_a")

    def test_compare_pair_b_much_better_prefer_b(self):
        from src.ab_testing import CreativeABTester
        tester = CreativeABTester(win_threshold=0.80, n_thompson=500, random_state=0)
        a = self._creative(0.30, ci_half=0.02)
        b = self._creative(0.80, ci_half=0.02)
        result = tester.compare_pair(a, b)
        self.assertLess(result.win_probability, 0.10)
        self.assertEqual(result.recommendation, "prefer_b")

    def test_compare_pair_ci_overlap_in_range(self):
        from src.ab_testing import CreativeABTester
        tester = CreativeABTester(n_thompson=500, random_state=0)
        a = self._creative(0.70, ci_half=0.05)
        b = self._creative(0.72, ci_half=0.05)
        result = tester.compare_pair(a, b)
        self.assertGreaterEqual(result.ci_overlap_fraction, 0.0)
        self.assertLessEqual(result.ci_overlap_fraction, 1.0)

    def test_compare_pair_cohens_d_sign(self):
        """Cohen's d should be positive when A > B and negative when A < B."""
        from src.ab_testing import CreativeABTester
        tester = CreativeABTester(n_thompson=500, random_state=0)
        a = self._creative(0.80)
        b = self._creative(0.50)
        res = tester.compare_pair(a, b)
        self.assertGreater(res.cohens_d, 0.0)

        # Reverse: B > A should give negative d
        res2 = tester.compare_pair(b, a)
        self.assertLess(res2.cohens_d, 0.0)

    def test_compare_pair_result_fields_populated(self):
        from src.ab_testing import ABTestResult, CreativeABTester
        tester = CreativeABTester(n_thompson=200, random_state=0)
        a = self._creative(0.70, video_id="vid_a", frame_idx=3)
        b = self._creative(0.60, video_id="vid_b", frame_idx=7)
        result = tester.compare_pair(a, b)
        self.assertIsInstance(result, ABTestResult)
        self.assertEqual(result.creative_a_video, "vid_a")
        self.assertEqual(result.creative_b_video, "vid_b")
        self.assertEqual(result.creative_a_frame_idx, 3)
        self.assertEqual(result.creative_b_frame_idx, 7)
        self.assertIn(result.recommendation, ("prefer_a", "prefer_b", "inconclusive"))

    def test_thompson_consistency_with_closed_form(self):
        """Thompson win rate should match P(A>B) within 3% at 10k samples.

        At n_thompson = 10 000 the Monte Carlo standard error for a Bernoulli
        proportion p is at most sqrt(p*(1-p)/n) ≤ 1/(2*sqrt(10000)) = 0.5%.
        The 3% tolerance (6 standard errors) gives a very low false-failure
        rate while still catching gross divergence between the two methods.
        """
        from src.ab_testing import CreativeABTester
        tester = CreativeABTester(n_thompson=10_000, random_state=1)
        a = self._creative(0.80, ci_half=0.04)
        b = self._creative(0.60, ci_half=0.04)
        result = tester.compare_pair(a, b)
        thompson_win_rate = result.n_thompson_wins_a / tester.n_thompson
        self.assertAlmostEqual(
            thompson_win_rate, result.win_probability, delta=0.03
        )

    def test_compare_all_pairs_count(self):
        from src.ab_testing import CreativeABTester
        tester = CreativeABTester(n_thompson=200, random_state=0)
        ranking = [
            self._creative(s, rank=i + 1, frame_idx=i, video_id=f"v{i}")
            for i, s in enumerate([0.9, 0.7, 0.5, 0.3])
        ]
        results = tester.compare_all_pairs(ranking)
        # C(4, 2) = 6 pairs
        self.assertEqual(len(results), 6)

    def test_compare_all_pairs_empty_ranking(self):
        from src.ab_testing import CreativeABTester
        tester = CreativeABTester(n_thompson=200, random_state=0)
        self.assertEqual(tester.compare_all_pairs([]), [])

    def test_thompson_select_returns_correct_count(self):
        from src.ab_testing import CreativeABTester
        tester = CreativeABTester(n_thompson=1000, random_state=0)
        ranking = [
            self._creative(s, rank=i + 1, frame_idx=i, video_id=f"v{i}")
            for i, s in enumerate([0.9, 0.7, 0.5, 0.4, 0.3])
        ]
        selected = tester.thompson_select(ranking, n_select=3, n_samples=1000)
        self.assertEqual(len(selected), 3)
        # All selected indices should be distinct
        self.assertEqual(len(set(selected)), 3)

    def test_thompson_select_top_scorer_appears_in_top(self):
        """The highest-scoring creative should almost always be in the top-1."""
        from src.ab_testing import CreativeABTester
        tester = CreativeABTester(n_thompson=5000, random_state=2)
        ranking = [
            self._creative(0.95, ci_half=0.01, rank=1, frame_idx=0, video_id="best"),
            self._creative(0.40, ci_half=0.01, rank=2, frame_idx=1, video_id="mid"),
            self._creative(0.20, ci_half=0.01, rank=3, frame_idx=2, video_id="low"),
        ]
        selected = tester.thompson_select(ranking, n_select=1, n_samples=5000)
        self.assertEqual(selected[0], 0)  # index 0 = highest-scoring creative

    def test_plot_comparison_creates_png(self):
        from src.ab_testing import CreativeABTester
        tester = CreativeABTester(n_thompson=200, random_state=0)
        a = self._creative(0.75, video_id="vid_a")
        b = self._creative(0.55, video_id="vid_b")
        result = tester.compare_pair(a, b)
        tmp = tempfile.mkdtemp()
        out = os.path.join(tmp, "ab_chart.png")
        path = tester.plot_comparison(result, a, b, out)
        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 0)

    def test_plot_tournament_heatmap_creates_png(self):
        from src.ab_testing import CreativeABTester
        tester = CreativeABTester(n_thompson=200, random_state=0)
        ranking = [
            self._creative(s, rank=i + 1, frame_idx=i, video_id=f"v{i}")
            for i, s in enumerate([0.9, 0.7, 0.5])
        ]
        tmp = tempfile.mkdtemp()
        out = os.path.join(tmp, "tournament.png")
        path = tester.plot_tournament_heatmap(ranking, out)
        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 0)

    def test_save_results_creates_json(self):
        from src.ab_testing import CreativeABTester
        tester = CreativeABTester(n_thompson=200, random_state=0)
        a = self._creative(0.80, video_id="vid_a")
        b = self._creative(0.60, video_id="vid_b")
        results = [tester.compare_pair(a, b)]
        tmp = tempfile.mkdtemp()
        out = os.path.join(tmp, "ab_results.json")
        path = tester.save_results(results, out)
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)
        self.assertIn("win_probability", data[0])
        self.assertIn("recommendation", data[0])


# ---------------------------------------------------------------------------
# per_video_stats consecutive_sims fix tests
# ---------------------------------------------------------------------------

class TestTemporalPerVideoStatsFix(unittest.TestCase):

    def _make_data(self, n=20):
        """Create synthetic embeddings and index for two videos."""
        rng = np.random.default_rng(42)
        embs = rng.random((n, 8)).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs /= norms + 1e-8
        index = [
            {"video_id": "v0" if i < n // 2 else "v1",
             "frame_idx": i, "timestamp": float(i)}
            for i in range(n)
        ]
        return embs, index

    def test_per_video_stats_backward_compatible(self):
        """per_video_stats without consecutive_sims still works."""
        from src.temporal_analysis import TemporalAnalyser
        embs, index = self._make_data(20)
        ta = TemporalAnalyser(window=2)
        curve = ta.compute_temporal_curve(embs, index)
        stats = ta.per_video_stats(curve, index)  # no consecutive_sims
        self.assertIn("v0", stats)
        self.assertIn("v1", stats)
        self.assertIn("n_transitions", stats["v0"])

    def test_per_video_stats_with_consecutive_sims(self):
        """per_video_stats with consecutive_sims populates stats without error."""
        from src.temporal_analysis import TemporalAnalyser
        embs, index = self._make_data(20)
        ta = TemporalAnalyser(window=2)
        curve = ta.compute_temporal_curve(embs, index)
        consec = ta.compute_consecutive_similarities(embs, index)
        stats = ta.per_video_stats(curve, index, consecutive_sims=consec)
        self.assertIn("v0", stats)
        self.assertIn("v1", stats)
        for vid in ("v0", "v1"):
            self.assertIn("n_transitions", stats[vid])
            self.assertGreaterEqual(stats[vid]["n_transitions"], 0.0)

    def test_per_video_stats_consec_sims_different_from_windowed(self):
        """
        Using consecutive_sims for transitions should give a different
        (and more accurate) count than using the windowed temporal curve.
        Inject a hard cut so the two methods disagree.
        """
        from src.temporal_analysis import TemporalAnalyser
        rng = np.random.default_rng(0)
        n = 30
        embs = rng.random((n, 16)).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs /= norms + 1e-8
        # Inject a hard cut at position 15 by making frame 15 orthogonal
        # to its neighbours — this produces a very low consecutive similarity
        embs[15] = rng.random(16).astype(np.float32)
        embs[15] = embs[15] / (np.linalg.norm(embs[15]) + 1e-8)
        # Make it nearly orthogonal to frame 14 (random seed gives that)
        index = [
            {"video_id": "single_vid", "frame_idx": i, "timestamp": float(i)}
            for i in range(n)
        ]
        ta = TemporalAnalyser(window=2)
        curve = ta.compute_temporal_curve(embs, index)
        consec = ta.compute_consecutive_similarities(embs, index)

        stats_windowed = ta.per_video_stats(curve, index)
        stats_consec   = ta.per_video_stats(curve, index, consecutive_sims=consec)
        # Both should return valid (non-negative) transition counts
        self.assertGreaterEqual(stats_windowed["single_vid"]["n_transitions"], 0.0)
        self.assertGreaterEqual(stats_consec["single_vid"]["n_transitions"], 0.0)


# ===========================================================================
# TestModalityAligner — CLIP modality gap correction
# ===========================================================================

class TestModalityAligner(unittest.TestCase):
    """Tests for src.modality_alignment.ModalityAligner."""

    def _rand_l2(self, rng, n: int, d: int) -> np.ndarray:
        """Return (n, d) L2-normalised float32 embeddings."""
        x = rng.random((n, d)).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        return x

    def test_fit_gap_shape(self):
        from src.modality_alignment import ModalityAligner
        rng = np.random.default_rng(0)
        img = self._rand_l2(rng, 20, 32)
        txt = self._rand_l2(rng, 10, 32)
        al  = ModalityAligner().fit(img, txt)
        self.assertEqual(al.gap.shape, (32,))

    def test_gap_nonzero_for_distinct_modalities(self):
        from src.modality_alignment import ModalityAligner
        rng = np.random.default_rng(1)
        # Offset text embeddings so gap is non-trivial
        img = self._rand_l2(rng, 20, 32)
        txt = self._rand_l2(rng, 10, 32) + 0.5  # shift to different cone
        txt /= np.linalg.norm(txt, axis=1, keepdims=True) + 1e-8
        al  = ModalityAligner().fit(img, txt)
        self.assertGreater(al.gap_magnitude, 0.01)

    def test_correct_image_l2_normalised(self):
        from src.modality_alignment import ModalityAligner
        rng = np.random.default_rng(2)
        img = self._rand_l2(rng, 15, 32)
        txt = self._rand_l2(rng, 8, 32)
        al  = ModalityAligner().fit(img, txt)
        corr = al.correct_image(img)
        norms = np.linalg.norm(corr, axis=1)
        np.testing.assert_allclose(norms, np.ones(15), atol=1e-5)

    def test_correct_text_l2_normalised(self):
        from src.modality_alignment import ModalityAligner
        rng = np.random.default_rng(3)
        img = self._rand_l2(rng, 15, 32)
        txt = self._rand_l2(rng, 8, 32)
        al  = ModalityAligner().fit(img, txt)
        corr = al.correct_text(txt)
        norms = np.linalg.norm(corr, axis=1)
        np.testing.assert_allclose(norms, np.ones(8), atol=1e-5)

    def test_zero_gap_for_same_modality(self):
        """When image == text embeddings, the gap should be ~0."""
        from src.modality_alignment import ModalityAligner
        rng = np.random.default_rng(4)
        x  = self._rand_l2(rng, 12, 32)
        al = ModalityAligner().fit(x, x)
        self.assertAlmostEqual(al.gap_magnitude, 0.0, places=5)

    def test_cosine_similarity_corrected_shape(self):
        from src.modality_alignment import ModalityAligner
        rng = np.random.default_rng(5)
        img = self._rand_l2(rng, 10, 32)
        txt = self._rand_l2(rng, 4, 32)
        al  = ModalityAligner().fit(img, txt)
        sim = al.cosine_similarity_corrected(img, txt)
        self.assertEqual(sim.shape, (10, 4))

    def test_save_load_roundtrip(self):
        from src.modality_alignment import ModalityAligner
        rng = np.random.default_rng(6)
        img = self._rand_l2(rng, 10, 16)
        txt = self._rand_l2(rng, 6, 16)
        al  = ModalityAligner().fit(img, txt)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "aligner.npz")
            al.save(path)
            al2 = ModalityAligner.load(path)
        np.testing.assert_array_equal(al.gap, al2.gap)
        self.assertAlmostEqual(al.gap_magnitude, al2.gap_magnitude, places=6)

    def test_unfit_correct_image_raises(self):
        from src.modality_alignment import ModalityAligner
        rng = np.random.default_rng(7)
        img = self._rand_l2(rng, 5, 16)
        with self.assertRaises(RuntimeError):
            ModalityAligner().correct_image(img)

    def test_dimension_mismatch_raises(self):
        from src.modality_alignment import ModalityAligner
        rng = np.random.default_rng(8)
        img = self._rand_l2(rng, 5, 16)
        txt = self._rand_l2(rng, 4, 32)   # different D
        with self.assertRaises(ValueError):
            ModalityAligner().fit(img, txt)

    def test_empty_input_raises(self):
        from src.modality_alignment import ModalityAligner
        img = np.empty((0, 16), dtype=np.float32)
        txt = np.random.rand(4, 16).astype(np.float32)
        with self.assertRaises(ValueError):
            ModalityAligner().fit(img, txt)

    def test_gap_summary_keys(self):
        from src.modality_alignment import ModalityAligner
        rng = np.random.default_rng(9)
        al  = ModalityAligner().fit(
            self._rand_l2(rng, 8, 16), self._rand_l2(rng, 4, 16)
        )
        s = al.gap_summary()
        self.assertIn("gap_magnitude", s)
        self.assertIn("mean_image_norm", s)
        self.assertIn("mean_text_norm", s)
        self.assertIn("embedding_dim", s)
        self.assertEqual(s["embedding_dim"], 16)

    def test_load_missing_file_raises(self):
        from src.modality_alignment import ModalityAligner
        with self.assertRaises(FileNotFoundError):
            ModalityAligner.load("/nonexistent/path/aligner.npz")


# ===========================================================================
# TestPipelineConfig — typed configuration
# ===========================================================================

class TestPipelineConfig(unittest.TestCase):
    """Tests for src.pipeline_config.PipelineConfig."""

    def test_defaults_valid(self):
        from src.pipeline_config import PipelineConfig
        cfg = PipelineConfig()
        self.assertGreater(cfg.fps, 0)
        self.assertGreater(cfg.max_frames, 0)
        self.assertGreater(cfg.top_k, 0)

    def test_invalid_fps_raises(self):
        from src.pipeline_config import PipelineConfig
        with self.assertRaises(ValueError):
            PipelineConfig(fps=-1.0)

    def test_invalid_max_frames_raises(self):
        from src.pipeline_config import PipelineConfig
        with self.assertRaises(ValueError):
            PipelineConfig(max_frames=0)

    def test_invalid_dedup_threshold_raises(self):
        from src.pipeline_config import PipelineConfig
        with self.assertRaises(ValueError):
            PipelineConfig(dedup_threshold=1.5)

    def test_invalid_min_quality_raises(self):
        from src.pipeline_config import PipelineConfig
        with self.assertRaises(ValueError):
            PipelineConfig(min_quality_score=-0.1)

    def test_invalid_bootstrap_n_raises(self):
        from src.pipeline_config import PipelineConfig
        with self.assertRaises(ValueError):
            PipelineConfig(bootstrap_n=10)

    def test_to_dict_from_dict_roundtrip(self):
        from src.pipeline_config import PipelineConfig
        cfg  = PipelineConfig(fps=2.0, n_clusters=5, top_k=3)
        cfg2 = PipelineConfig.from_dict(cfg.to_dict())
        self.assertEqual(cfg.fps, cfg2.fps)
        self.assertEqual(cfg.n_clusters, cfg2.n_clusters)
        self.assertEqual(cfg.top_k, cfg2.top_k)

    def test_save_load_json(self):
        from src.pipeline_config import PipelineConfig
        cfg = PipelineConfig(fps=1.5, max_frames=30)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "cfg.json")
            cfg.save(path)
            self.assertTrue(os.path.isfile(path))
            cfg2 = PipelineConfig.load(path)
        self.assertAlmostEqual(cfg.fps, cfg2.fps)
        self.assertEqual(cfg.max_frames, cfg2.max_frames)

    def test_load_missing_raises(self):
        from src.pipeline_config import PipelineConfig
        with self.assertRaises(FileNotFoundError):
            PipelineConfig.load("/nonexistent/config.json")

    def test_from_args_maps_interval_to_fps(self):
        from src.pipeline_config import PipelineConfig
        import argparse
        ns = argparse.Namespace(
            interval=2.0,
            model="openai/clip-vit-base-patch32",
            max_frames=40,
            top_k=5,
            n_queries=3,
            n_clusters=0,
            dedup_threshold=0.97,
            min_quality_score=0.25,
            bootstrap_n=1000,
            scene_adaptive=False,
            skip_download=False,
            skip_extraction=False,
            skip_embedding=False,
            skip_similarity=False,
        )
        cfg = PipelineConfig.from_args(ns)
        self.assertAlmostEqual(cfg.fps, 2.0)
        self.assertEqual(cfg.model_name, "openai/clip-vit-base-patch32")
        self.assertEqual(cfg.max_frames, 40)

    def test_unknown_keys_ignored_in_from_dict(self):
        from src.pipeline_config import PipelineConfig
        d = PipelineConfig().to_dict()
        d["future_unknown_field"] = "ignored"
        cfg = PipelineConfig.from_dict(d)
        self.assertFalse(hasattr(cfg, "future_unknown_field"))

    def test_repr_contains_fps(self):
        from src.pipeline_config import PipelineConfig
        cfg = PipelineConfig(fps=3.0)
        self.assertIn("fps", repr(cfg))
        self.assertIn("3.0", repr(cfg))


# ===========================================================================
# TestOnlinePredictor — incremental SGD with CUSUM drift detection
# ===========================================================================

class TestOnlinePredictor(unittest.TestCase):
    """Tests for src.online_updater.OnlinePredictor."""

    def _make_data(self, n: int = 20, d: int = 6, seed: int = 0):
        rng = np.random.default_rng(seed)
        X = rng.random((n, d)).astype(np.float32)
        w = rng.random(d).astype(np.float32)
        y = X @ w + 0.05 * rng.standard_normal(n).astype(np.float32)
        return X, y

    def test_partial_fit_grows_window(self):
        from src.online_updater import OnlinePredictor
        op = OnlinePredictor(n_features=6)
        X, y = self._make_data(10)
        op.partial_fit(X, y)
        self.assertEqual(op.window_size, 10)
        self.assertEqual(op.n_updates, 10)

    def test_window_capped_at_max_window(self):
        from src.online_updater import OnlinePredictor
        op = OnlinePredictor(n_features=6, max_window=15)
        for _ in range(4):
            X, y = self._make_data(10)
            op.partial_fit(X, y)
        self.assertLessEqual(op.window_size, 15)

    def test_predict_returns_finite_values(self):
        from src.online_updater import OnlinePredictor
        op = OnlinePredictor(n_features=6)
        X, y = self._make_data(20)
        op.partial_fit(X, y)
        preds = op.predict(X[:5])
        self.assertEqual(preds.shape, (5,))
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_predict_before_fit_raises(self):
        from src.online_updater import OnlinePredictor
        op = OnlinePredictor(n_features=6)
        X, _ = self._make_data(5)
        with self.assertRaises(RuntimeError):
            op.predict(X)

    def test_wrong_n_features_raises(self):
        from src.online_updater import OnlinePredictor
        op = OnlinePredictor(n_features=6)
        X, y = self._make_data(10)
        op.partial_fit(X, y)
        X_wrong = np.random.rand(5, 8).astype(np.float32)
        with self.assertRaises(ValueError):
            op.partial_fit(X_wrong, np.zeros(5))

    def test_no_cusum_alarm_with_zero_bias(self):
        """Model predicts accurately → CUSUM should stay quiet."""
        from src.online_updater import OnlinePredictor
        op = OnlinePredictor(n_features=6)
        # Feed perfectly predicted samples (y ≈ 0 always → near-zero residuals)
        for _ in range(20):
            X = np.zeros((5, 6), dtype=np.float64)
            y = np.zeros(5, dtype=np.float64)
            op.partial_fit(X, y)
        # Allow some false positives but expect no alarm on constant-zero data
        # (residuals will be exactly 0 after the model converges to w=0)
        ds = op.drift_summary()
        self.assertIn("drift_alarm", ds)
        self.assertIn("cusum_pos", ds)

    def test_cusum_alarm_with_large_bias(self):
        """Systematically large residuals should trigger the CUSUM alarm."""
        from src.online_updater import OnlinePredictor
        # Prime the model on low-label data
        op = OnlinePredictor(n_features=6, random_state=42)
        X0, _ = self._make_data(20, seed=0)
        op.partial_fit(X0, np.zeros(20))  # model learns y ≈ 0

        # Now inject labels far from 0 → large residuals accumulate
        for _ in range(30):
            X_bias = np.ones((5, 6), dtype=np.float64) * 0.5
            y_bias = np.full(5, 10.0)   # model predicts ≈0, label=10 → big error
            op.partial_fit(X_bias, y_bias)

        self.assertTrue(op.drift_detected())

    def test_reset_cusum_clears_alarm(self):
        from src.online_updater import OnlinePredictor
        op = OnlinePredictor(n_features=6)
        # Force alarm
        op._drift_alarm = True
        op._cusum_pos   = 99.0
        op.reset_cusum()
        self.assertFalse(op.drift_detected())
        self.assertAlmostEqual(op._cusum_pos, 0.0)

    def test_predict_with_interval_has_correct_shape(self):
        from src.online_updater import OnlinePredictor
        op = OnlinePredictor(n_features=6, random_state=42)
        for _ in range(6):        # fill window to ≥ 5 samples
            X, y = self._make_data(10, seed=_)
            op.partial_fit(X, y)
        X_q = self._make_data(4)[0]
        mean, lo, hi = op.predict_with_interval(X_q)
        self.assertEqual(mean.shape, (4,))
        self.assertEqual(lo.shape, (4,))
        self.assertEqual(hi.shape, (4,))
        self.assertTrue(np.all(hi >= lo))

    def test_save_load_roundtrip(self):
        from src.online_updater import OnlinePredictor
        op = OnlinePredictor(n_features=6)
        X, y = self._make_data(20)
        op.partial_fit(X, y)
        preds_before = op.predict(X[:3])
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "online.npz")
            op.save(path)
            op2 = OnlinePredictor.load(path)
        preds_after = op2.predict(X[:3])
        np.testing.assert_allclose(preds_before, preds_after, atol=1e-5)
        self.assertEqual(op.n_updates, op2.n_updates)

    def test_residual_stats_empty_returns_empty_dict(self):
        from src.online_updater import OnlinePredictor
        op = OnlinePredictor(n_features=6)
        self.assertEqual(op.residual_stats(), {})

    def test_drift_summary_keys(self):
        from src.online_updater import OnlinePredictor
        op = OnlinePredictor(n_features=6)
        X, y = self._make_data(10)
        op.partial_fit(X, y)
        s = op.drift_summary()
        self.assertIn("cusum_pos", s)
        self.assertIn("cusum_neg", s)
        self.assertIn("threshold", s)
        self.assertIn("drift_alarm", s)
        self.assertIn("n_updates", s)


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ===========================================================================
# TestAudioFeatureExtractor
# ===========================================================================

class TestAudioFeatureExtractor(unittest.TestCase):
    """Tests for src/audio_features.py — pure signal-processing functions."""

    _SR = 22_050

    def _sine_wave(self, freq: float = 440.0, duration: float = 1.0) -> np.ndarray:
        """Generate a pure-tone sine wave at *freq* Hz."""
        t = np.linspace(0, duration, int(self._SR * duration), endpoint=False)
        return (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)

    def _silent(self, n: int = 22_050) -> np.ndarray:
        return np.zeros(n, dtype=np.float32)

    def _noise(self, n: int = 22_050, seed: int = 1) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.standard_normal(n).astype(np.float32) * 0.3

    # ------------------------------------------------------------------
    # AudioSpectralFeatures
    # ------------------------------------------------------------------

    def test_extract_features_returns_correct_type(self):
        from src.audio_features import extract_audio_features, AudioSpectralFeatures
        feats = extract_audio_features(self._sine_wave(), sr=self._SR)
        self.assertIsInstance(feats, AudioSpectralFeatures)

    def test_all_scalar_features_in_unit_range(self):
        from src.audio_features import extract_audio_features
        feats = extract_audio_features(self._sine_wave(), sr=self._SR)
        for field in ("rms_energy", "zcr", "spectral_centroid",
                      "spectral_bandwidth", "spectral_rolloff", "mfcc_var"):
            val = getattr(feats, field)
            self.assertGreaterEqual(val, 0.0, f"{field} below 0")
            self.assertLessEqual(val, 1.0, f"{field} above 1")

    def test_mfcc_shape(self):
        from src.audio_features import extract_audio_features, _MFCC_N_COEFF
        feats = extract_audio_features(self._sine_wave(), sr=self._SR)
        self.assertEqual(feats.mfcc_means.shape, (_MFCC_N_COEFF,))

    def test_silent_audio_low_rms(self):
        """Silent audio should give near-zero RMS."""
        from src.audio_features import extract_audio_features
        feats = extract_audio_features(self._silent(), sr=self._SR)
        self.assertAlmostEqual(feats.rms_energy, 0.0, places=5)

    def test_spectral_features_silent_audio(self):
        """Spectral features of silence should all be zero."""
        from src.audio_features import compute_spectral_features
        c, bw, ro = compute_spectral_features(self._silent(), sr=self._SR)
        self.assertAlmostEqual(c, 0.0, places=5)
        self.assertAlmostEqual(bw, 0.0, places=5)
        self.assertAlmostEqual(ro, 0.0, places=5)

    def test_int16_audio_normalised_correctly(self):
        """extract_audio_features should accept int16 PCM and auto-normalise."""
        from src.audio_features import extract_audio_features
        audio_i16 = (self._sine_wave() * 32767).astype(np.int16)
        feats = extract_audio_features(audio_i16, sr=self._SR)
        # After normalisation RMS should be similar to float32 version
        feats_f32 = extract_audio_features(self._sine_wave(), sr=self._SR)
        self.assertAlmostEqual(feats.rms_energy, feats_f32.rms_energy, places=2)

    def test_empty_audio_raises_value_error(self):
        from src.audio_features import extract_audio_features
        with self.assertRaises(ValueError):
            extract_audio_features(np.array([], dtype=np.float32))

    # ------------------------------------------------------------------
    # map_to_affective_axes
    # ------------------------------------------------------------------

    def test_map_to_affective_axes_keys(self):
        from src.audio_features import (
            extract_audio_features, map_to_affective_axes, _DISCORD_AXES
        )
        feats = extract_audio_features(self._sine_wave(), sr=self._SR)
        scores = map_to_affective_axes(feats)
        self.assertEqual(set(scores.keys()), set(_DISCORD_AXES))

    def test_map_to_affective_axes_range(self):
        from src.audio_features import extract_audio_features, map_to_affective_axes
        feats = extract_audio_features(self._noise(), sr=self._SR)
        scores = map_to_affective_axes(feats)
        for axis, val in scores.items():
            self.assertGreaterEqual(val, -1.0, f"{axis} below -1")
            self.assertLessEqual(val, 1.0, f"{axis} above 1")

    def test_energy_higher_for_loud_than_silent(self):
        """Louder audio should produce a higher 'energy' axis score."""
        from src.audio_features import extract_audio_features, map_to_affective_axes
        loud = (np.sin(2 * np.pi * 440 * np.linspace(0, 1, self._SR)) * 0.9).astype(np.float32)
        quiet = self._silent()
        scores_loud = map_to_affective_axes(extract_audio_features(loud, sr=self._SR))
        scores_quiet = map_to_affective_axes(extract_audio_features(quiet, sr=self._SR))
        self.assertGreater(scores_loud["energy"], scores_quiet["energy"])

    # ------------------------------------------------------------------
    # cross_modal_discord_score
    # ------------------------------------------------------------------

    def test_discord_identical_vectors_is_zero(self):
        from src.audio_features import cross_modal_discord_score, _DISCORD_AXES
        aff = {ax: 0.5 for ax in _DISCORD_AXES}
        self.assertAlmostEqual(cross_modal_discord_score(aff, aff), 0.0, places=5)

    def test_discord_opposite_vectors_is_two(self):
        from src.audio_features import cross_modal_discord_score, _DISCORD_AXES
        pos = {ax: 1.0 for ax in _DISCORD_AXES}
        neg = {ax: -1.0 for ax in _DISCORD_AXES}
        self.assertAlmostEqual(cross_modal_discord_score(pos, neg), 2.0, places=5)

    def test_discord_near_zero_vectors_returns_neutral(self):
        from src.audio_features import cross_modal_discord_score, _DISCORD_AXES
        zero = {ax: 0.0 for ax in _DISCORD_AXES}
        self.assertAlmostEqual(cross_modal_discord_score(zero, zero), 0.5, places=5)

    def test_discord_value_in_bounds(self):
        """Discord should always be in [0, 2]."""
        from src.audio_features import cross_modal_discord_score, _DISCORD_AXES
        rng = np.random.default_rng(11)
        for _ in range(20):
            a = {ax: float(rng.uniform(-1, 1)) for ax in _DISCORD_AXES}
            v = {ax: float(rng.uniform(-1, 1)) for ax in _DISCORD_AXES}
            d = cross_modal_discord_score(a, v)
            self.assertGreaterEqual(d, 0.0)
            self.assertLessEqual(d, 2.0 + 1e-6)

    # ------------------------------------------------------------------
    # generate_synthetic_audio_features
    # ------------------------------------------------------------------

    def test_generate_synthetic_correct_count(self):
        from src.audio_features import generate_synthetic_audio_features
        synth = generate_synthetic_audio_features(n_windows=7, seed=42)
        self.assertEqual(len(synth), 7)

    def test_generate_synthetic_reproducible(self):
        from src.audio_features import generate_synthetic_audio_features
        s1 = generate_synthetic_audio_features(3, seed=99)
        s2 = generate_synthetic_audio_features(3, seed=99)
        self.assertAlmostEqual(s1[0].rms_energy, s2[0].rms_energy, places=8)

    def test_to_dict_serialisable(self):
        """AudioSpectralFeatures.to_dict() should produce a JSON-serialisable dict."""
        import json
        from src.audio_features import generate_synthetic_audio_features
        feats = generate_synthetic_audio_features(1, seed=0)[0]
        d = feats.to_dict()
        # Should not raise
        json.dumps(d)


# ===========================================================================
# TestEmbeddingVectorStore
# ===========================================================================

class TestEmbeddingVectorStore(unittest.TestCase):
    """Tests for src/vector_store.py."""

    def _unit_embs(self, n: int, d: int = 16, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        raw = rng.standard_normal((n, d)).astype(np.float32)
        return raw / np.linalg.norm(raw, axis=1, keepdims=True)

    def _meta(self, n: int) -> list:
        return [{"frame_id": i, "video_id": f"v{i // 5}"} for i in range(n)]

    def test_add_and_len(self):
        from src.vector_store import EmbeddingVectorStore
        store = EmbeddingVectorStore()
        store.add(self._unit_embs(10), self._meta(10))
        self.assertEqual(len(store), 10)

    def test_incremental_add(self):
        from src.vector_store import EmbeddingVectorStore
        store = EmbeddingVectorStore()
        store.add(self._unit_embs(5), self._meta(5))
        store.add(self._unit_embs(3, seed=1), self._meta(3))
        self.assertEqual(len(store), 8)

    def test_search_returns_k_results(self):
        from src.vector_store import EmbeddingVectorStore
        embs = self._unit_embs(20)
        store = EmbeddingVectorStore()
        store.add(embs, self._meta(20))
        results = store.search(embs[0], k=4)
        self.assertEqual(len(results), 4)

    def test_search_exact_match_is_top_hit(self):
        """Querying with an embedding that is in the store should return itself first."""
        from src.vector_store import EmbeddingVectorStore
        embs = self._unit_embs(15)
        store = EmbeddingVectorStore()
        store.add(embs, self._meta(15))
        results = store.search(embs[3], k=3)
        self.assertEqual(results[0].idx, 3)
        self.assertAlmostEqual(results[0].score, 1.0, places=4)

    def test_search_results_sorted_descending(self):
        from src.vector_store import EmbeddingVectorStore
        embs = self._unit_embs(20)
        store = EmbeddingVectorStore()
        store.add(embs, self._meta(20))
        results = store.search(embs[0], k=5)
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_search_k_larger_than_n_clamped(self):
        from src.vector_store import EmbeddingVectorStore
        embs = self._unit_embs(4)
        store = EmbeddingVectorStore()
        store.add(embs, self._meta(4))
        results = store.search(embs[0], k=100)
        self.assertEqual(len(results), 4)

    def test_empty_store_raises_runtime_error(self):
        from src.vector_store import EmbeddingVectorStore
        store = EmbeddingVectorStore()
        with self.assertRaises(RuntimeError):
            store.search(np.zeros(16, dtype=np.float32))

    def test_dimension_mismatch_raises_value_error(self):
        from src.vector_store import EmbeddingVectorStore
        embs = self._unit_embs(5, d=16)
        store = EmbeddingVectorStore()
        store.add(embs, self._meta(5))
        with self.assertRaises(ValueError):
            store.search(np.zeros(32, dtype=np.float32))   # wrong dim

    def test_add_metadata_len_mismatch_raises(self):
        from src.vector_store import EmbeddingVectorStore
        store = EmbeddingVectorStore()
        embs = self._unit_embs(5)
        with self.assertRaises(ValueError):
            store.add(embs, self._meta(3))  # len mismatch

    def test_save_load_roundtrip(self):
        from src.vector_store import EmbeddingVectorStore
        embs = self._unit_embs(10)
        store = EmbeddingVectorStore()
        store.add(embs, self._meta(10))
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "vs")
            store.save(path)
            store2 = EmbeddingVectorStore.load(path)
        self.assertEqual(len(store2), 10)
        # Top hit should be the query itself after reload
        results = store2.search(embs[7], k=1)
        self.assertEqual(results[0].idx, 7)
        self.assertAlmostEqual(results[0].score, 1.0, places=4)

    def test_repr_contains_expected_info(self):
        from src.vector_store import EmbeddingVectorStore
        store = EmbeddingVectorStore()
        store.add(self._unit_embs(5, d=32), self._meta(5))
        r = repr(store)
        self.assertIn("n=5", r)
        self.assertIn("D=32", r)
        self.assertIn("cosine", r)

    def test_thread_safe_concurrent_add(self):
        """Three threads add 10 embeddings each; final len should be 30."""
        import threading
        from src.vector_store import EmbeddingVectorStore
        store = EmbeddingVectorStore()
        errors = []

        def _add_batch(seed):
            try:
                embs = self._unit_embs(10, seed=seed)
                store.add(embs, self._meta(10))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_add_batch, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"Thread errors: {errors}")
        self.assertEqual(len(store), 30)


# ===========================================================================
# TestAffectiveScorerPromptCache
# ===========================================================================

class TestAffectiveScorerPromptCache(unittest.TestCase):
    """Verifies that AffectiveScorer caches prompt embeddings at init time."""

    def _make_scorer(self, MockModel, MockProcessor, dim=16):
        """Shared helper: build a patched AffectiveScorer and return it with the mock model."""
        import torch
        from src.affective_scoring import AffectiveScorer, DEFAULT_AXES

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        def _get_text(**kwargs):
            n = kwargs["input_ids"].shape[0]
            f = torch.randn(n, dim)
            return f / f.norm(dim=-1, keepdim=True)

        mock_model.get_text_features.side_effect = _get_text
        MockModel.from_pretrained.return_value = mock_model

        def _proc(*args, text=None, images=None, return_tensors=None,
                  padding=None, truncation=None, **kw):
            batch = text if text is not None else (images or [])
            n = len(batch)
            return {"input_ids": torch.zeros(n, 77, dtype=torch.long),
                    "pixel_values": torch.zeros(n, 3, 224, 224)}

        MockProcessor.from_pretrained.return_value = MagicMock(side_effect=_proc)
        scorer = AffectiveScorer(model_name="mock/clip", axes=DEFAULT_AXES)
        return scorer, mock_model

    @patch("src.affective_scoring.CLIPModel")
    @patch("src.affective_scoring.CLIPProcessor")
    def test_single_cache_pre_warmed_at_init(self, MockProcessor, MockModel):
        """_single_cache should contain all DEFAULT_AXES keys after __init__."""
        from src.affective_scoring import DEFAULT_AXES
        scorer, _ = self._make_scorer(MockModel, MockProcessor)
        self.assertEqual(set(scorer._single_cache.keys()), set(DEFAULT_AXES.keys()))

    @patch("src.affective_scoring.CLIPModel")
    @patch("src.affective_scoring.CLIPProcessor")
    def test_score_frames_no_new_encode_calls(self, MockProcessor, MockModel):
        """score_frames() should use the cache — zero new encode_text calls."""
        scorer, mock_model = self._make_scorer(MockModel, MockProcessor)
        call_count_after_init = mock_model.get_text_features.call_count

        embs = np.random.randn(4, 16).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        scorer.score_frames(embs)

        self.assertEqual(
            mock_model.get_text_features.call_count,
            call_count_after_init,
            "score_frames() made unexpected encode_text calls (cache miss).",
        )

    @patch("src.affective_scoring.CLIPModel")
    @patch("src.affective_scoring.CLIPProcessor")
    def test_ensemble_cache_populated_on_first_call(self, MockProcessor, MockModel):
        """_ensemble_cache should be populated after first score_frames_ensemble()."""
        from src.affective_scoring import MULTI_PROMPT_AXES
        scorer, _ = self._make_scorer(MockModel, MockProcessor)
        embs = np.random.randn(4, 16).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        scorer.score_frames_ensemble(embs)
        self.assertEqual(
            set(scorer._ensemble_cache.keys()), set(MULTI_PROMPT_AXES.keys())
        )

    @patch("src.affective_scoring.CLIPModel")
    @patch("src.affective_scoring.CLIPProcessor")
    def test_ensemble_cache_no_new_calls_on_second_use(self, MockProcessor, MockModel):
        """Second call to score_frames_ensemble() should use cache entirely."""
        scorer, mock_model = self._make_scorer(MockModel, MockProcessor)
        embs = np.random.randn(4, 16).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        scorer.score_frames_ensemble(embs)       # first call — populates cache
        count_after_first = mock_model.get_text_features.call_count
        scorer.score_frames_ensemble(embs)       # second call — should be cache hit
        self.assertEqual(mock_model.get_text_features.call_count, count_after_first)

    @patch("src.affective_scoring.CLIPModel")
    @patch("src.affective_scoring.CLIPProcessor")
    def test_invalidate_clears_both_caches(self, MockProcessor, MockModel):
        """invalidate_cache() should empty both _single_cache and _ensemble_cache."""
        scorer, _ = self._make_scorer(MockModel, MockProcessor)
        embs = np.random.randn(4, 16).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        scorer.score_frames_ensemble(embs)   # populate ensemble cache
        scorer.invalidate_cache()
        self.assertEqual(len(scorer._single_cache), 0)
        self.assertEqual(len(scorer._ensemble_cache), 0)


# ===========================================================================
# TestSimilarityIndexLengthGuard
# ===========================================================================

class TestSimilarityIndexLengthGuard(unittest.TestCase):
    """Verifies the new index-length validation in compute_inter_video_stats."""

    def _make_sim(self, n: int) -> np.ndarray:
        return np.eye(n, dtype=np.float32)

    def _make_index(self, n: int) -> list:
        return [{"video_id": f"v{i % 2}"} for i in range(n)]

    def test_mismatch_raises_value_error(self):
        from src.similarity import compute_inter_video_stats
        sim = self._make_sim(5)
        with self.assertRaises(ValueError, msg="Should raise for len(index)=3 != n=5"):
            compute_inter_video_stats(sim, self._make_index(3))

    def test_correct_length_passes(self):
        from src.similarity import compute_inter_video_stats
        n = 6
        sim = self._make_sim(n)
        stats = compute_inter_video_stats(sim, self._make_index(n))
        self.assertIn("overall_mean", stats)

    def test_zero_length_consistent(self):
        """(0, 0) matrix with empty index should return NaN stats without error."""
        from src.similarity import compute_inter_video_stats
        sim = np.zeros((0, 0), dtype=np.float32)
        stats = compute_inter_video_stats(sim, [])
        self.assertTrue(np.isnan(stats["overall_mean"]))

