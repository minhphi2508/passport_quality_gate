# Deployment Notes

The reference package is deployment-neutral.

- `device="auto"`: CUDA if PyTorch reports CUDA available, otherwise CPU.
- explicit device values supported by the underlying YOLO runtime may be supplied.
- camera resolution, camera API, preview FPS and shutter implementation are owned by the host application.
- choose/install the PyTorch build appropriate for the target environment rather than hard-coding the laptop CPU environment.

Potential later targets include server GPU inference, ONNX/TensorRT, or mobile-specific runtimes. Those are deployment phases, not assumptions of SDK 0.1.0.
