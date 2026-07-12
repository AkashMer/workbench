import os
import sys
from pathlib import Path

# Walks the output dir (Nole-Nexus Website/Workbench/) for rendered .md files
# Rewrites relative figure paths to absolute paths rooted at /Workbench/{slug}/
# so Quartz can resolve them when building from Nole-Nexus Website/ as site root
#
# Guard: only run when explicitly rendering into the private site repo (production
# render passes --output-dir into Nole-Nexus Website/Workbench/). Local preview/render
# never sets an output dir pointing there, so skip the rewrite - otherwise every local
# preview gets its figure links rewritten to a /Workbench/ path that doesn't exist locally.
output_dir_env = os.environ.get('QUARTO_PROJECT_OUTPUT_DIR')
if not output_dir_env or 'Nole-Nexus Website' not in output_dir_env:
    sys.exit(0)

output_dir = Path(output_dir_env)

for md_file in output_dir.rglob('*.md'):
    slug = md_file.stem
    content = md_file.read_text(encoding='utf-8')
    if f'{slug}_files/' in content:
        content = content.replace(f'{slug}_files/', f'/Workbench/{slug}/{slug}_files/')
        md_file.write_text(content, encoding='utf-8')
