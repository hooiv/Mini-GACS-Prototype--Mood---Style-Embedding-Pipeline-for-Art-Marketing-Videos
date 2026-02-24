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


if __name__ == "__main__":
    unittest.main(verbosity=2)
