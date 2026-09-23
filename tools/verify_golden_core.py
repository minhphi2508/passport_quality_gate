from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
LOCK=ROOT/'GOLDEN_CORE_SHA256.json'

def main():
    data=json.loads(LOCK.read_text(encoding='utf-8'))
    hashes=data['files']
    problems=[]
    for rel, expected in hashes.items():
        p=ROOT/rel
        if not p.is_file():
            problems.append(f'MISSING {rel}')
            continue
        actual=hashlib.sha256(p.read_bytes()).hexdigest()
        if actual != expected:
            problems.append(f'CHANGED {rel}\n  expected {expected}\n  actual   {actual}')
    if problems:
        print('\n'.join(problems))
        raise SystemExit(2)
    print(f"Golden core OK: {len(hashes)} files match {data['quality_policy']}")

if __name__=='__main__': main()
