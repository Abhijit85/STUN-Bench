#!/usr/bin/env python3
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
expected=root/'paper_tables/expected.json'
if not expected.exists(): expected.write_text(json.dumps({'status':'placeholder'}, indent=2)+'\n')
print(expected.read_text().strip())
