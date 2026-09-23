# Best Recent Frame Selection

Purpose: avoid using the exact shutter-time frame when finger movement introduces a small amount of motion blur.

Default behavior:

- retain only frames already marked `capture_allowed=True` by Golden FP2;
- keep a bounded in-memory buffer (default 12 frames / 96 MiB, whichever is hit first);
- consider frames within the most recent 750 ms;
- reject an old candidate when its detected document geometry differs strongly from the current document position;
- rank using existing Golden metrics, with blur and OCR-critical resolution weighted most strongly;
- use glare only as a small ranking signal because FP2 glare has known limitations;
- add only a small recency preference so a clearly sharper older frame can beat the shutter-time frame.

All values are configuration defaults, not mobile/server requirements. No image is written to disk by the selector.
