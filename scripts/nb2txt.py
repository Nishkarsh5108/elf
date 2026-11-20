"""Flatten .ipynb files (markdown+code, no outputs except short text) into .txt for reading."""
import json, sys, glob, os
for path in sys.argv[1:]:
    nb = json.load(open(path, encoding='utf-8'))
    out = []
    for c in nb.get('cells', []):
        src = ''.join(c.get('source', ''))
        out.append(f"# ---- [{c['cell_type']}]\n{src}\n")
        if c['cell_type'] == 'code':
            for o in c.get('outputs', []):
                t = ''.join(o.get('text', '')) or ''.join(o.get('data', {}).get('text/plain', ''))
                if t:
                    out.append("# >>> output:\n" + t[:3000] + "\n")
    with open(path[:-6] + '.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print(path, sum(len(x) for x in out))
