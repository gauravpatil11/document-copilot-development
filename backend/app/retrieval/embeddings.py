"""Google Gemini query embedding for live retrieval."""

from __future__ import annotations

from google import genai
from google.genai.types import EmbedContentConfig

from app.config import settings


def embed_query(text: str) -> list[float]:
    client = genai.Client(api_key=settings.gemini_api_key)
    response = client.models.embed_content(
        model=settings.gemini_embedding_model,
        contents=text,
        config=EmbedContentConfig(output_dimensionality=settings.embedding_dimensions),
    )
    embedding = response.embeddings[0].values
    expected_dims = settings.embedding_dimensions
    if len(embedding) != expected_dims:
        raise ValueError(
            f"Expected embedding dimension {expected_dims}, got {len(embedding)}"
        )
    return embedding
