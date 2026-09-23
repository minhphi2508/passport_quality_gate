import json
from passport_quality_gate.diagnostics import runtime_diagnostics

print(json.dumps(runtime_diagnostics(), indent=2))
