"""Standalone HTML fallback that renders Mermaid code in the browser."""

from __future__ import annotations

import html
from pathlib import Path

from mictotext.config import RenderConfig

_TEMPLATE = """<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ background: #ffffff; margin: 0; padding: 24px; font-family: "Segoe UI", Arial, sans-serif; }}
  .mermaid {{ background: #ffffff; }}
</style>
</head>
<body>
<pre class="mermaid">
{code}
</pre>
<script src="{script_src}"></script>
<script>
  mermaid.initialize({{ startOnLoad: true, securityLevel: "strict" }});
</script>
</body>
</html>
"""


def write_html(mermaid_code: str, path: Path, cfg: RenderConfig, title: str = "Schema") -> Path:
    local_js = cfg.local_mermaid_js
    # A local copy keeps the page fully offline; otherwise only the library is fetched from the CDN.
    script_src = local_js.resolve().as_uri() if local_js.is_file() else cfg.mermaid_cdn_url
    path.write_text(
        _TEMPLATE.format(title=html.escape(title), code=html.escape(mermaid_code), script_src=script_src),
        encoding="utf-8",
    )
    return path
