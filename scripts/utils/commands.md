# Useful commands

# Final render
quarto render data-slug.qmd --output-dir "$env:NOLE_NEXUS\Nole-Nexus Website\Workbench"

# Generate the environment history yml (Out-File -Encoding utf8, not >, to avoid UTF-16)
mamba env export --from-history | Out-File -Encoding utf8 environment.yml
# Generate personal envrionement file
mamba env export --no-builds | Out-File -Encoding utf8 my_environment.yml
# Merge a --from-history environment.yml with versions from my_environment.yml (full export)
# Run from the workbench root
mamba run python scripts/utils/merge_env_versions.py --slug post-slug --environment environment.yml --full my_environment.yml --out environment.yml
