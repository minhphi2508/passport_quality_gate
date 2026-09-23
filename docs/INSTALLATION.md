# Installation / deployment notes

## Python reference environment
Recommended for the current reference build: Python 3.11 or 3.12.

The wheel declares the detector runtime (`ultralytics`) plus NumPy/OpenCV/PyYAML. For GPU/server deployment, install the PyTorch build matching the target CUDA stack first, then install the SDK wheel; pip will reuse a compatible installed PyTorch. For a plain CPU acceptance environment, installing the wheel normally is sufficient for dependency resolution from the configured package index.

The source-tree SDK can be installed editable with `pip install -e .`. A wheel is also produced by the release builder. Default model/config assets are packaged inside the wheel, so installed usage does not depend on the repository directory layout.

## OpenCV note
The SDK runtime uses OpenCV image operations and defaults to the headless package for server-friendly installation. The webcam examples call `cv2.imshow`; a local GUI-capable OpenCV build may be required for those examples. Camera/UI code is not part of the SDK contract.

## Hardware
`device="auto"` resolves to CUDA when PyTorch reports CUDA available, otherwise CPU. Callers can explicitly request a device instead. Benchmarking and production sizing must be done on the target deployment hardware.


## Fresh-environment acceptance
A release should be tested from a new virtual environment, not from the source repository. Verify that `passport_quality_gate.__file__` points into that environment's `site-packages`, instantiate `PassportQualityGate(device="auto")`, inspect `gate.metadata` or `gate.runtime_info()`, run one preview/final image, and run the RAM-only best-frame selector. Full-frame preview/final calls require a normalized guide box.
