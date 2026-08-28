"""Vision helpers — multimodal message construction for LangChain.

Builds ``HumanMessage`` objects that carry both text and image data
for use with vision-capable models served via vLLM's OpenAI-compatible API.

Usage::

    from sovereign.vision import build_multimodal_message

    msg = build_multimodal_message(
        text="Describe this image",
        image_path="/path/to/photo.jpg",
    )
    # msg is a HumanMessage with content=[
    #   {"type": "text", "text": "Describe this image"},
    #   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
    # ]
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp", ".gif"})

# MIME type mapping
_MIME_MAP: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def encode_image(image_path: str | Path) -> str:
    """Read a local image file and return its base64-encoded content.

    Parameters
    ----------
    image_path : str | Path
        Absolute or relative path to the image file.

    Returns
    -------
    str
        Base64-encoded string of the image bytes.

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.
    ValueError
        If the file extension is not a recognized image type.
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported image type '{suffix}'. "
            f"Supported: {', '.join(sorted(IMAGE_EXTENSIONS))}"
        )

    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


def get_mime_type(image_path: str | Path) -> str:
    """Return the MIME type for an image file based on its extension."""
    suffix = Path(image_path).suffix.lower()
    return _MIME_MAP.get(suffix, "image/jpeg")


def build_multimodal_message(
    text: str,
    image_path: str | Path | None = None,
) -> HumanMessage:
    """Build a ``HumanMessage`` with optional image attachment.

    If *image_path* is ``None`` or empty, returns a plain text message.

    Parameters
    ----------
    text : str
        The text prompt to send to the model.
    image_path : str | Path | None
        Path to a local image file.  If provided, the image is base64-encoded
        and embedded in the message as an ``image_url`` content block.

    Returns
    -------
    HumanMessage
        A LangChain HumanMessage ready for ``llm.invoke()``.
    """
    if not image_path:
        return HumanMessage(content=text)

    path = Path(image_path)
    if not path.exists():
        logger.warning("Image not found at %s — falling back to text-only message", path)
        return HumanMessage(content=text)

    try:
        image_b64 = encode_image(path)
        mime = get_mime_type(path)
        data_uri = f"data:{mime};base64,{image_b64}"
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("Failed to encode image: %s — falling back to text-only", exc)
        return HumanMessage(content=text)

    return HumanMessage(content=[
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": data_uri}},
    ])


def is_image_file(file_path: str | Path) -> bool:
    """Check if a file path points to a supported image type."""
    return Path(file_path).suffix.lower() in IMAGE_EXTENSIONS
