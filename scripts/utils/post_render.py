import os
from pathlib import Path

# Walks the output dir (Nole-Nexus Website/Workbench/) for rendered .md files
# Rewrites relative figure paths to absolute paths rooted at /Workbench/{slug}/
# so Quartz can resolve them when building from Nole-Nexus Website/ as site root
output_dir = Path(os.environ.get('QUARTO_PROJECT_OUTPUT_DIR', '.'))

for md_file in output_dir.rglob('*.md'):
    slug = md_file.stem
    content = md_file.read_text(encoding='utf-8')
    if f'{slug}_files/' in content:
        content = content.replace(f'{slug}_files/', f'/Workbench/{slug}/{slug}_files/')
        md_file.write_text(content, encoding='utf-8')
