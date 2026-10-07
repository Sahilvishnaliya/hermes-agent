"""Image-payload helpers for :mod:`agent.context_compressor`, split out to keep the facade inside
its line cap (see root ``AGENTS.md`` on topical siblings).

Two families live here, both moved verbatim from ``agent/context_compressor.py``:

* the multimodal part predicates and strippers the compaction passes use
  (``_strip_historical_media``, ``_retire_stale_tool_result_images``);
* the STATELESS send-path eviction pass ``evict_stale_outbound_tool_images``, which retires stale
  screenshot/vision payloads from the per-call API copy so OpenAI-style ``image_url`` tool results
  do not ride every later request until a 413 forces a reactive strip (#89286, #89296).

``context_compressor`` re-imports every name below, so existing importers (and the tests that pull
these helpers off ``agent.context_compressor``) keep working unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from agent.image_eviction_policy import outbound_image_retire_count, images_all_within_many_image_limit
from agent.turn_context import drop_stale_api_content

# Newest image-bearing tool results kept verbatim; older image payloads retire
# even inside protect_last_n (matches the Anthropic adapter's keep-window).
# Native vision_analyze / computer_use screenshots that sit inside the protected tail cannot be demoted by
# pass 2, so they ride every later request until anti-thrash disables compression (#92699).
_MAX_KEEP_TOOL_IMAGES = 3
# Compaction window only. The send path's same-valued OUTBOUND_IMAGE_FLOOR (agent/image_eviction_policy.py)
# is a satisfiability floor with different semantics; do not merge the two.


def _replace_image_parts(parts: Any, placeholder: str) -> Optional[List[Any]]:
    """New parts list with every image part replaced by a text placeholder; None if no images."""
    if not isinstance(parts, list) or not any(_is_image_part(p) for p in parts):
        return None
    return [{"type": "text", "text": placeholder} if _is_image_part(p) else p for p in parts]


def _tool_result_parts(content: Any) -> Any:
    """Part list of a tool-result body, unwrapping the ``_multimodal`` envelope."""
    return content.get("content") if isinstance(content, dict) and content.get("_multimodal") else content


def _tool_content_has_images(content: Any) -> bool:
    """True when a tool-result body (part list or ``_multimodal`` envelope) carries images."""
    return _content_has_images(_tool_result_parts(content))


def _strip_images_from_tool_msg(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Copy of a tool message with image payloads replaced (stale ``api_content`` dropped); ``None`` if nothing to strip."""
    content = msg.get("content")
    if isinstance(content, dict) and content.get("_multimodal"):
        summary = content.get("text_summary") or "[screenshot removed to save context]"
        return _rewritten(msg, f"[screenshot removed] {str(summary)[:200]}")
    stripped = _replace_image_parts(content, "[screenshot removed to save context]")
    return None if stripped is None else _rewritten(msg, stripped)


def _rewritten(msg: Dict[str, Any], content: Any) -> Dict[str, Any]:
    """Copy of ``msg`` carrying ``content``; drops the stale ``api_content`` sidecar so replay can't resend it."""
    new_msg = {**msg, "content": content}
    drop_stale_api_content(new_msg)
    return new_msg


def _retire_stale_tool_result_images(
    result: List[Dict[str, Any]], keep_newest: int = _MAX_KEEP_TOOL_IMAGES, spared: range = range(0),
) -> int:
    """Replace image payloads on older tool results with text placeholders.
    Keeps the newest ``keep_newest`` image-bearing tool messages and any spared pending round;
    spared images still count toward the newest window. User uploads are untouched. Mutates
    ``result`` in place; returns the number of messages rewritten. Compaction only: it commits the
    rewrite into the canonical transcript once. The send path uses
    :func:`evict_stale_outbound_tool_images` (a per-request keep-newest window rewrites the cached
    prefix on every new image, #113517)."""
    seen = pruned = 0
    for i in range(len(result) - 1, -1, -1):
        msg = result[i]
        if not isinstance(msg, dict) or msg.get("role") != "tool" or not _tool_content_has_images(msg.get("content")):
            continue
        seen += 1
        if seen <= max(keep_newest, 0) or i in spared:
            continue
        new_msg = _strip_images_from_tool_msg(msg)
        if new_msg is not None:
            result[i] = new_msg
            pruned += 1
    return pruned


def _image_payload(msg: Dict[str, Any]) -> Tuple[int, int, List[str]]:
    """``(blocks, bytes, sources)`` of image payload in a message.

    The provider counts BLOCKS: one ``tool_result`` carrying three screenshots is three against
    the per-request limit. Bytes are the data-URL / base64 length — the payload is ASCII and the
    JSON framing around it is noise against a 24 MB budget, so no per-request re-serialization.
    ``sources`` holds each image's data URL/base64 string so the caller can read dimensions from
    headers only, without a second walk or a full decode.
    """
    parts = _tool_result_parts(msg.get("content"))
    if not isinstance(parts, list):
        return 0, 0, []
    blocks = payload = 0
    sources: List[str] = []
    for p in parts:
        if not _is_image_part(p):
            continue
        blocks += 1
        image_url = p.get("image_url")
        source = p.get("source")
        data = (
            (image_url.get("url") if isinstance(image_url, dict) else image_url)
            or (source.get("data") if isinstance(source, dict) else None)
        )
        if isinstance(data, str):
            payload += len(data)
            sources.append(data)
        else:
            sources.append("")
    return blocks, payload, sources


def evict_stale_outbound_tool_images(api_messages: List[Dict[str, Any]]) -> int:
    """Drop stale screenshot/vision payloads from the per-call API copy.

    Compression's keep-newest pass only runs when prune/compress fires, and the Anthropic
    adapter's screenshot eviction only sees nested ``tool_result`` blocks. OpenAI-style
    ``image_url`` tool results otherwise ride every subsequent request until a 413 forces
    the reactive strip (#89286). Call this on the cloned ``api_messages`` list after
    sanitization (#89296). Do not pass persisted history — the rewrite is send-path only.

    Eviction is driven by the provider limit, counted in image BLOCKS, with user uploads
    reserved against the ceiling but never rewritten — policy and rationale in
    :mod:`agent.image_eviction_policy`. The block ceiling is conditional on the request's actual
    image dimensions: the strict 20 only when some image exceeds the many-image per-side cap, the
    hard count otherwise, so an image-heavy run of pre-shrunk screenshots no longer pays a prefix
    rewrite per batch. Returns the number of messages rewritten.
    """
    carriers: List[Tuple[int, int, int, List[str]]] = []  # (index, blocks, bytes, sources)
    reserved_blocks = reserved_bytes = 0
    reserved_sources: List[str] = []
    for i in range(len(api_messages) - 1, -1, -1):
        msg = api_messages[i]
        if not isinstance(msg, dict):
            continue
        blocks, size, sources = _image_payload(msg)
        if not blocks:
            continue
        if msg.get("role") == "tool":
            carriers.append((i, blocks, size, sources))
        else:
            reserved_blocks += blocks
            reserved_bytes += size
            reserved_sources.extend(sources)
    many_image_safe = images_all_within_many_image_limit(
        [*reserved_sources, *(source for _, _, _, sources in carriers for source in sources)]
    )
    retire = outbound_image_retire_count(
        [blocks for _, blocks, _, _ in carriers],
        reserved_blocks,
        carrier_bytes_newest_first=[size for _, _, size, _ in carriers],
        reserved_bytes=reserved_bytes,
        many_image_safe=many_image_safe,
    )
    pruned = 0
    for i, _, _, _ in carriers[len(carriers) - retire:]:
        new_msg = _strip_images_from_tool_msg(api_messages[i])
        if new_msg is not None:
            api_messages[i] = new_msg
            pruned += 1
    return pruned


_IMAGE_PART_TYPES = frozenset({"image_url", "input_image", "image"})


def _is_image_part(part: Any) -> bool:
    """True if ``part`` is an image block (``image_url``, ``input_image``, or ``image``)."""
    return isinstance(part, dict) and part.get("type") in _IMAGE_PART_TYPES


def _content_has_images(content: Any) -> bool:
    """True if a message's ``content`` is a multimodal list with image parts."""
    return isinstance(content, list) and any(_is_image_part(p) for p in content)


def _strip_images_from_content(content: Any) -> Any:
    """``content`` with image parts replaced by placeholders; unchanged (same object) when none."""
    stripped = _replace_image_parts(content, "[Attached image — stripped after compression]")
    return content if stripped is None else stripped
