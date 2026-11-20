"""Fetch Kaggle competition pages (overview, evaluation, rules, ...) into research/pages/."""
import os, sys
from kaggle.api.kaggle_api_extended import KaggleApi

comp = sys.argv[1] if len(sys.argv) > 1 else "santa-2026"
out = os.path.join(os.path.dirname(__file__), "..", "research", "pages")
os.makedirs(out, exist_ok=True)
api = KaggleApi(); api.authenticate()
pages = api.competition_list_pages(comp)
for p in pages:
    d = {k: getattr(p, k) for k in dir(p) if not k.startswith("_") and not callable(getattr(p, k))}
    name = d.get("name") or "page"
    content = d.get("content") or ""
    with open(os.path.join(out, f"{name}.md"), "w", encoding="utf-8") as f:
        f.write(content)
    print(name, len(content), [k for k in d if k not in ("content",)])
