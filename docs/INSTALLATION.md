# Installation — SDK 0.1.4

Python reference runtime: Python 3.11 or 3.12.

For normal integration, install the provided wheel once in the dev environment. Reinstalling dependencies is not required for every code/test iteration. Editable source development can use the same existing virtual environment.

The wheel declares NumPy, OpenCV-headless, PyYAML and Ultralytics. For CUDA deployment, install a PyTorch build appropriate for the target environment before the SDK wheel when necessary.

The detector weights, default thresholds and `capture_viewport` profile are packaged in the wheel; installed usage does not depend on repository-relative config paths.

`examples/webcam_capture.py` uses `cv2.imshow`, so a GUI-capable OpenCV build may be needed for that example. UI/camera code is not part of the SDK contract.
