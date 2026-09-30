# Deployment Notes — SDK 0.1.4

The Python package is a reference runtime. `device="auto"` uses CUDA when the installed PyTorch reports CUDA available, otherwise CPU.

The host owns camera API, preview FPS/resolution, threading around the camera loop, mobile wrappers, networking/storage and target-device optimization. Potential ONNX/TensorRT/CoreML/TFLite work is a later deployment phase and is not part of this release.

Profile the complete app on target hardware; the SDK does not claim a universal FPS/thermal budget.
