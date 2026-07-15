import os
import sys
from pathlib import Path

# Only run when rendering into the private site repo
output_dir_env = os.environ.get('QUARTO_PROJECT_OUTPUT_DIR')
if not output_dir_env or 'Nole-Nexus Website' not in output_dir_env:
    sys.exit(0)

output_dir = Path(output_dir_env)

# Walk the rendered .md files and rewrite relative figure paths to absolute
# paths rooted at /Workbench/{slug}/ so Quartz can resolve them from
# Nole-Nexus Website/ as site root
for md_file in output_dir.rglob('*.md'):
    slug = md_file.stem
    content = md_file.read_text(encoding='utf-8')
    target = f'/Workbench/{slug}/{slug}_files/'
    # Skip files already rewritten so reruns don't compound paths in older posts
    if f'{slug}_files/' in content and target not in content:
        content = content.replace(f'{slug}_files/', target)
        md_file.write_text(content, encoding='utf-8')
