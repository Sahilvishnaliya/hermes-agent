"""Arithmetic contract of the shared send-path image-eviction policy (#113517).

Both outbound passes (``context_compressor.evict_stale_outbound_tool_images`` and
``anthropic_message_convert._evict_old_screenshots``) translate their message shapes into
``(carrier blocks newest-first, reserved)`` and call this one function; the shape tests in
``test_outbound_stale_vision.py`` / ``test_computer_use.py`` cover the translation, this file
covers the numbers once.
"""

from __future__ import annotations

import base64

import pytest

from agent.anthropic_message_convert import _evict_old_screenshots
from agent.context_compressor import evict_stale_outbound_tool_images
from agent.image_eviction_policy import (
    IMAGE_EVICTION_BATCH,
    OUTBOUND_IMAGE_FLOOR,
    OUTBOUND_IMAGE_LIMIT,
    outbound_image_retire_count,
)

MB = 1_000_000

# Anthropic's >20-images stricter per-side cap in px. Kept as a literal, not an import, so the
# two invariants below still run (and fail) against the pre-change policy module.
MANY_IMAGE_CAP_PX = 2000


def _png_data_url(width: int, height: int) -> str:
    """A data URL whose PNG IHDR declares ``width`` x ``height``.

    Only the 24 bytes a header-only reader inspects are needed, which is exactly the contract
    the policy relies on: classifying a request must not decode whole payloads.
    """
    ihdr = "IHDR".encode("ascii") + width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x06\x00\x00\x00"
    raw = b"\x89PNG\r\n\x1a\n" + len(ihdr).to_bytes(4, "big") + ihdr
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def _screenshot_tool_messages(n: int, url: str) -> list[dict]:
    """``n`` OpenAI-shaped tool results, one image block each."""
    messages: list[dict] = []
    for i in range(n):
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"call_{i}",
                "content": [
                    {"type": "text", "text": f"shot {i}"},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        )
    return messages


def _wire_tool_results(n: int, url: str) -> list[dict]:
    """``n`` Anthropic wire user-turns, each carrying one image ``tool_result``."""
    data = url.split(",", 1)[1]
    result: list[dict] = [{"role": "user", "content": [{"type": "text", "text": "start"}]}]
    for i in range(n):
        result.append({"role": "assistant", "content": [{"type": "text", "text": f"shot {i}"}]})
        result.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": f"t{i}",
                        "content": [
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": "image/png", "data": data},
                            }
                        ],
                    }
                ],
            }
        )
    return result


def _wire_image_blocks(result: list[dict]) -> int:
    return sum(
        1
        for msg in result
        for block in (msg.get("content") if isinstance(msg.get("content"), list) else [])
        for inner in (block.get("content") if block.get("type") == "tool_result" else [block])
        if isinstance(inner, dict) and inner.get("type") == "image"
    )


@pytest.mark.parametrize(
    ("blocks", "reserved", "sizes", "reserved_bytes", "expected"),
    [
        # at the limit: append-only, nothing rewritten
        ([1] * OUTBOUND_IMAGE_LIMIT, 0, None, 0, 0),
        # one over: exactly one batch
        ([1] * (OUTBOUND_IMAGE_LIMIT + 1), 0, None, 0, IMAGE_EVICTION_BATCH),
        # still within the first window after a batch: same retire count (frontier holds)
        ([1] * (OUTBOUND_IMAGE_LIMIT + IMAGE_EVICTION_BATCH), 0, None, 0, IMAGE_EVICTION_BATCH),
        # second window: two batches, never a fixed single batch
        ([1] * (OUTBOUND_IMAGE_LIMIT + IMAGE_EVICTION_BATCH + 1), 0, None, 0, 2 * IMAGE_EVICTION_BATCH),
        # one carrier breaching alone: fixable, floor yields
        ([OUTBOUND_IMAGE_LIMIT + 5], 0, None, 0, 1),
        # four 10-block carriers: retire until it fits, not everything
        ([10, 10, 10, 10], 0, None, 0, 2),
        # uploads alone breach: unfixable, keep the floor
        ([1] * 5, OUTBOUND_IMAGE_LIMIT + 1, None, 0, 5 - OUTBOUND_IMAGE_FLOOR),
        # uploads alone breach with fewer carriers than the floor: nothing to gain
        ([1, 1], OUTBOUND_IMAGE_LIMIT + 5, None, 0, 0),
        # byte pressure from uploads: floor never shelters a 413
        ([1, 1, 1], 5, [3 * MB] * 3, 25 * MB, 3),
        # bytes bind before blocks: 2 MB frames, 13 of them
        ([1] * 13, 0, [2 * MB] * 13, 0, IMAGE_EVICTION_BATCH),
        # empty
        ([], 0, None, 0, 0),
    ],
)
def test_retire_count(blocks, reserved, sizes, reserved_bytes, expected):
    assert (
        outbound_image_retire_count(
            blocks, reserved, carrier_bytes_newest_first=sizes, reserved_bytes=reserved_bytes
        )
        == expected
    )


@pytest.mark.parametrize("n", [OUTBOUND_IMAGE_LIMIT + 1, 30])
def test_pre_shrunk_screenshots_do_not_trip_the_many_image_trigger(n):
    """Invariant 1 (#133999): a request whose images are all within the many-image per-side cap
    crosses no Anthropic constraint at image #21, so no prefix may be rewritten.

    Hermes embeds its own tool images at 1568 px (vision_analyze / browser) and 1456 px
    (computer_use), i.e. comfortably under the 2000 px cap the >20 rule would tighten. Retiring a
    batch here buys nothing and costs a full prompt-cache rewrite per batch -- plus the cache
    invalidation of every later thinking block on the preserved-thinking models. Red on base:
    8 rewrites at n=21, 16 at n=30.
    """
    shrunk = _png_data_url(1568, 882)
    assert evict_stale_outbound_tool_images(_screenshot_tool_messages(n, shrunk)) == 0

    # The wire pass reaches aux/MoA requests without the compressor's pass, so it must hold
    # the same invariant alone.
    wire = _wire_tool_results(n, shrunk)
    _evict_old_screenshots(wire)
    assert _wire_image_blocks(wire) == n


def test_one_oversized_image_keeps_the_strict_trigger():
    """Invariant 2 (#133999): the stricter cap is per request, so a single image over the cap --
    here a native-size user upload -- must keep the strict 20-block trigger for the whole request,
    byte-identical to the pre-#133999 behaviour."""
    shrunk = _png_data_url(1568, 882)
    oversized = _png_data_url(MANY_IMAGE_CAP_PX + 1, 8)

    upload = {
        "role": "user",
        "content": [
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": oversized}},
        ],
    }
    outbound = [upload, *_screenshot_tool_messages(OUTBOUND_IMAGE_LIMIT, shrunk)]
    assert evict_stale_outbound_tool_images(outbound) == IMAGE_EVICTION_BATCH
    # The upload itself is reserved and never rewritten.
    assert upload["content"][1]["image_url"]["url"] == oversized

    # Same request on the wire shape: 20 pre-shrunk carriers plus the oversized reserved upload.
    wire = _wire_tool_results(OUTBOUND_IMAGE_LIMIT, shrunk)
    wire[0]["content"].append(
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": oversized.split(",", 1)[1]}}
    )
    _evict_old_screenshots(wire)
    assert _wire_image_blocks(wire) == OUTBOUND_IMAGE_LIMIT - IMAGE_EVICTION_BATCH + 1


def test_dimension_classification_reads_headers_and_fails_closed():
    """The ceiling may only relax on dimensions this module can actually read.

    Imported inside the test so the two invariants above still exercise the pre-change module.
    """
    from agent.image_eviction_policy import (
        MANY_IMAGE_DIMENSION_LIMIT,
        image_max_side_from_data,
        image_within_many_image_limit,
        images_all_within_many_image_limit,
    )

    shrunk = _png_data_url(1568, 882)
    assert MANY_IMAGE_DIMENSION_LIMIT == MANY_IMAGE_CAP_PX
    assert image_max_side_from_data(shrunk) == 1568
    assert image_within_many_image_limit(_png_data_url(MANY_IMAGE_CAP_PX, 1)) is True
    assert image_within_many_image_limit(_png_data_url(MANY_IMAGE_CAP_PX + 1, 1)) is False
    # Unreadable, absent, or non-image sources must keep the strict trigger.
    assert image_within_many_image_limit(None) is False
    assert image_within_many_image_limit("data:image/png;base64,not-a-png") is False
    assert image_within_many_image_limit("https://example.com/shot.png") is False
    assert images_all_within_many_image_limit([shrunk, shrunk]) is True
    assert images_all_within_many_image_limit([shrunk, "https://example.com/shot.png"]) is False


def test_quantum_shrinks_to_the_fit_window_for_heavy_carriers():
    """Three-block carriers: a batch of eight is wider than the window, so the retire count
    must still be a step function (hold for window - floor turns), never ``total - floor``."""
    window = OUTBOUND_IMAGE_LIMIT // 3
    counts = [outbound_image_retire_count([3] * n, 0) for n in range(window + 1, window + 10)]
    assert all(c > 0 for c in counts)
    assert all(3 * (n - c) <= OUTBOUND_IMAGE_LIMIT for n, c in zip(range(window + 1, window + 10), counts))
    moves = sum(a != b for a, b in zip(counts, counts[1:]))
    assert moves <= len(counts) // (window - OUTBOUND_IMAGE_FLOOR)
