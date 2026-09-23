# Best Recent Frame Selection — SDK 0.1.3

Purpose: avoid relying only on the exact shutter-time frame when finger/device motion makes that frame slightly worse than a recent eligible frame.

Default behavior:

- retain only preview frames marked `capture_allowed=True` by Golden FP2
- bounded in-memory buffer (default 12 frames / 96 MiB, whichever limit is reached first)
- select within the most recent 750 ms window
- reject geometrically inconsistent candidates when document position changes too much
- rank using existing Golden quality diagnostics
- emphasize blur and OCR-critical resolution more strongly than glare
- apply only a small recency preference
- write no image to disk

## Integration requirement

Push the **full** preview result:

```python
preview = gate.analyze_preview(frame, guide, timestamp=t)
selector.push(frame, preview, timestamp=t)
```

Do not push the compact public result because the selector needs quality/localization diagnostics.

## Timestamps

Within one session:

- timestamps must be finite
- push timestamps must strictly increase
- trigger timestamp must not be older than the latest push

For a new session/document:

```python
selector.clear()
```

## Final handoff

The selected frame is still a full camera frame.

Required flow:

```text
select recent best full frame
        ↓
run final quality analysis on that exact frame
        ↓
ACCEPT
        ↓
extract passport page
        ↓
OCR crop
```

The selector itself does not crop or write images.
