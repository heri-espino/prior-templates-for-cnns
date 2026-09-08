"""Verify byte-for-byte imported evidence without ML dependencies."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
entries=json.loads((root/'.ai_handoff/import_manifest.json').read_text())
errors=[]
for item in entries:
 p=root/item['path']
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:errors.append(item['path'])
assert not errors, 'Imported files missing/changed: '+str(errors[:20])
study=root/'studies/cnn_causal_milestone'
assert len(list((study/'runs').glob('*/*/block*/*/model.pt')))==400
assert json.loads((study/'audit.json').read_text())['run_count']==400
print(f'PASS: {len(entries)} imported files match; 400 main checkpoints preserved.')
