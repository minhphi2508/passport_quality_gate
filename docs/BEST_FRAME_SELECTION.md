# Best Recent Frame Selection — SDK 0.1.4

`BestFrameSelector` keeps only recent capture-allowed ROI frames in RAM. Default limits are approximately 750 ms, 12 frames and 96 MiB. It writes nothing to disk.

Push the **exact ROI pixels** and full preview result:

```python
selector.push(roi_bgr, preview_result, timestamp=t)
selected = selector.select_recent(trigger_timestamp=t_click)
```

Final analysis and passport extraction must use `selected.frame` when a selection exists. Clear the selector on a new session/document or meaningful pause/resume.
