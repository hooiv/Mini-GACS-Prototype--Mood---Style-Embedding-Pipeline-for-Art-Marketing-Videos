"""
embeddings.py
-------------
Computes fixed-size visual embeddings for image frames using OpenAI CLIP
(loaded via HuggingFace *transformers*).  Results are stored as a NumPy
array alongside a JSON index so they can be reloaded without re-running
the model.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import torch
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False
    torch = None  # type: ignore[assignment]

try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError:  # pragma: no cover
    CLIPModel = None  # type: ignore[assignment]
    CLIPProcessor = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Default CLIP checkpoint – can be overridden at construction time
DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"


class EmbeddingModel:
    """
    Wraps a CLIP model to produce L2-normalised image embeddings.

    Args:
        model_name:  HuggingFace model identifier.  Defaults to
                     ``"openai/clip-vit-base-patch32"``.
        device:      Torch device string (e.g. ``"cpu"``, ``"cuda"``).
                     Auto-detected when *None*.
        batch_size:  Number of images processed per forward pass.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        batch_size: int = 16,
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError(
                "torch and transformers are required for EmbeddingModel. "
                "Install with: pip install torch transformers"
            )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        self.model_name = model_name

        logger.info("Loading CLIP model '%s' on device '%s'.", model_name, device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()
        logger.info("CLIP model loaded successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_images(self, image_paths: List[str]) -> np.ndarray:
        """
        Compute L2-normalised CLIP embeddings for a list of image paths.

        Skips files that cannot be opened and logs a warning for each.

        Args:
            image_paths:  Ordered list of absolute or relative image paths.

        Returns:
            Float32 NumPy array of shape ``(N, D)`` where *N* ≤ len(image_paths)
            and *D* is the CLIP embedding dimension (512 for ViT-B/32).

        Raises:
            ValueError: if *image_paths* is empty or all images fail to load.
        """
        if not image_paths:
            raise ValueError("image_paths must be a non-empty list.")

        embeddings: List[np.ndarray] = []

        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i : i + self.batch_size]
            images, valid_indices = self._load_images(batch_paths)
            if not images:
                continue

            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)

            # L2 normalise so cosine similarity reduces to dot product
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            embeddings.append(image_features.cpu().numpy().astype(np.float32))

            logger.debug(
                "Embedded batch %d–%d (%d valid images).",
                i, i + len(batch_paths), len(images),
            )

        if not embeddings:
            raise ValueError("No valid images could be embedded.")

        result = np.vstack(embeddings)
        self._verify_embeddings(result)
        return result

    def embed_single(self, image_path: str) -> np.ndarray:
        """
        Convenience wrapper to embed a single image.

        Returns:
            1-D float32 array of length *D*.
        """
        return self.embed_images([image_path])[0]

    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of text strings to L2-normalised CLIP text embeddings.

        CLIP places text and image embeddings in the same cosine-similarity
        space, so the output can be directly dot-producted with image embeddings
        (produced by :meth:`embed_images`) to measure text-to-image alignment.

        Args:
            texts:  List of strings (prompts, captions, brand briefs, …).

        Returns:
            Float32 array ``(N, D)`` of L2-normalised text embeddings, where
            *N* = ``len(texts)`` and *D* is the CLIP embedding dimension.

        Raises:
            ValueError: if *texts* is empty or any embedding is NaN.
        """
        if not texts:
            raise ValueError("texts must be a non-empty list.")

        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        # L2 normalise so dot product == cosine similarity with image embeddings
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        result = text_features.cpu().numpy().astype(np.float32)

        if np.isnan(result).any():
            raise ValueError("Text embeddings contain NaN values.")

        logger.debug(
            "Encoded %d text string(s): shape=%s, range=[%.4f, %.4f].",
            len(texts), result.shape,
            float(result.min()), float(result.max()),
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_images(
        paths: List[str],
    ) -> Tuple[List[Image.Image], List[int]]:
        """
        Load PIL images, skipping unreadable files.

        Returns:
            Tuple of (list-of-PIL-images, list-of-valid-original-indices).
        """
        images: List[Image.Image] = []
        valid_indices: List[int] = []
        for idx, path in enumerate(paths):
            if not os.path.exists(path):
                logger.warning("Image not found, skipping: %s", path)
                continue
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_indices.append(idx)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cannot load image '%s': %s", path, exc)
        return images, valid_indices

    @staticmethod
    def _verify_embeddings(embeddings: np.ndarray) -> None:
        """
        Assert basic sanity constraints on the embedding matrix.

        Raises:
            ValueError: if NaNs or Infs are present.
        """
        if np.isnan(embeddings).any():
            raise ValueError("Embeddings contain NaN values – check input images.")
        if np.isinf(embeddings).any():
            raise ValueError("Embeddings contain Inf values – check model outputs.")
        logger.debug(
            "Embedding verification passed: shape=%s, dtype=%s, range=[%.4f, %.4f].",
            embeddings.shape, embeddings.dtype,
            float(embeddings.min()), float(embeddings.max()),
        )


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_embeddings(
    embeddings: np.ndarray,
    index: List[Dict],
    output_dir: str,
    tag: str = "embeddings",
) -> Tuple[str, str]:
    """
    Save embeddings array and index mapping to *output_dir*.

    Args:
        embeddings:   Float32 array ``(N, D)``.
        index:        List of dicts (one per embedding) with at minimum
                      ``{"file_path": ..., "video_id": ..., ...}``.
        output_dir:   Directory where files are written.
        tag:          Filename prefix (default ``"embeddings"``).

    Returns:
        Tuple of ``(npy_path, json_path)``.
    """
    os.makedirs(output_dir, exist_ok=True)
    npy_path = os.path.join(output_dir, f"{tag}.npy")
    json_path = os.path.join(output_dir, f"{tag}_index.json")

    np.save(npy_path, embeddings)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)

    logger.info(
        "Saved embeddings: shape=%s → %s | index (%d entries) → %s.",
        embeddings.shape, npy_path, len(index), json_path,
    )
    return npy_path, json_path


def load_embeddings(
    output_dir: str,
    tag: str = "embeddings",
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Load previously saved embeddings and their index.

    Args:
        output_dir:  Directory that was passed to :func:`save_embeddings`.
        tag:         Filename prefix used when saving.

    Returns:
        Tuple of ``(embeddings_array, index_list)``.

    Raises:
        FileNotFoundError: if either expected file is missing.
    """
    npy_path = os.path.join(output_dir, f"{tag}.npy")
    json_path = os.path.join(output_dir, f"{tag}_index.json")

    for path in (npy_path, json_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected embedding file not found: {path}")

    embeddings = np.load(npy_path).astype(np.float32)
    with open(json_path, "r", encoding="utf-8") as fh:
        index = json.load(fh)

    logger.info("Loaded embeddings: shape=%s from %s.", embeddings.shape, npy_path)
    return embeddings, index


def compute_and_save_embeddings(
    metadata: List[Dict],
    output_dir: str,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 16,
    tag: str = "embeddings",
) -> Tuple[np.ndarray, List[Dict]]:
    """
    High-level convenience function: embed all frames described in *metadata*
    and persist the results.

    Args:
        metadata:    Frame metadata list from ``frame_extractor``.
        output_dir:  Where to store ``.npy`` and ``_index.json`` files.
        model_name:  CLIP HuggingFace identifier.
        batch_size:  Images per batch.
        tag:         File prefix.

    Returns:
        ``(embeddings, index)`` – also written to disk.
    """
    # Only pass files that actually exist; build the index from this filtered
    # list so that embedding row i always corresponds to index[i].  Using the
    # full `metadata` for enumerate() while filtering by existence would
    # produce an index whose embedding_idx values skip integers whenever an
    # image is missing, silently misaligning every downstream lookup.
    valid_metadata = [e for e in metadata if os.path.exists(e["file_path"])]
    skipped = len(metadata) - len(valid_metadata)
    if skipped:
        logger.warning(
            "%d / %d frame files not found and will be skipped.",
            skipped, len(metadata),
        )

    image_paths = [e["file_path"] for e in valid_metadata]
    model = EmbeddingModel(model_name=model_name, batch_size=batch_size)
    embeddings = model.embed_images(image_paths)

    # embedding row i ↔ valid_metadata[i] — indices are contiguous
    index = [{**entry, "embedding_idx": i} for i, entry in enumerate(valid_metadata)]

    save_embeddings(embeddings, index, output_dir, tag=tag)
    return embeddings, index
