import os
import shutil
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

# Posts with _quarto.yml above the qmd (e.g. renv/rig at the slug root)
# mirror a scripts/ subfolder into the output dir. Flatten it so the post
# lands directly in output_dir, matching every other post's layout
scripts_subdir = output_dir / 'scripts'
if scripts_subdir.is_dir():
    for item in scripts_subdir.iterdir():
        shutil.move(str(item), str(output_dir / item.name))
    scripts_subdir.rmdir()
