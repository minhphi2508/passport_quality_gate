# Output and Storage — SDK 0.1.4

Core SDK behavior is in-memory. `PassportQualityGate`, `BestFrameSelector`, and `extract_passport_page` do not persist passport images.

Recommended production path:

```text
selected ROI in RAM → final ACCEPT → passport crop in RAM → OCR/VLM
```

Disk recording in examples is opt-in debug behavior only. The host application owns retention, privacy, encryption and transport policy.
