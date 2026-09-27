"""Google Gemini embedding generation for document chunks."""

from __future__ import annotations

import re
import time

from google import genai
from google.genai import types
from google.genai.errors import ClientError

from app.config import settings

EMBED_BATCH_SIZE = 15


def _embed_batch_with_retry(
    client: genai.Client,
    model: str,
    contents: list[types.Content],
    config: types.EmbedContentConfig,
    max_retries: int = 10,
) -> list[list[float]]:
    expected_dims = config.output_dimensionality
    for attempt in range(max_retries):
        try:
            response = client.models.embed_content(
                model=model,
                contents=contents,
                config=config,
            )
            vectors: list[list[float]] = []
            for emb in response.embeddings:
                vec = emb.values
                if len(vec) != expected_dims:
                    raise ValueError(
                        f"Expected embedding dimension {expected_dims}, got {len(vec)}"
                    )
                vectors.append(vec)
            return vectors
        except ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                delay = 15.0
                match = re.search(r"retry\s*(?:in|Delay['\"]?:\s*['\"]?)(\d+)", str(e), re.I)
                if match:
                    delay = float(match.group(1)) + 2.0
                print(f"  [Rate limit reached: pausing {delay:.0f}s before resuming...]", flush=True)
                time.sleep(delay)
            else:
                raise
    raise RuntimeError(f"Embedding failed after {max_retries} retries due to rate limits.")


def embed_texts(texts: list[str], *, batch_size: int = EMBED_BATCH_SIZE) -> list[list[float]]:
    if not texts:
        return []

    expected_dims = settings.embedding_dimensions
    client = genai.Client(api_key=settings.gemini_api_key)
    config = types.EmbedContentConfig(output_dimensionality=expected_dims)
    vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in batch]
        batch_vectors = _embed_batch_with_retry(
            client=client,
            model=settings.gemini_embedding_model,
            contents=contents,
            config=config,
        )
        vectors.extend(batch_vectors)
        time.sleep(1.0)

    return vectors
