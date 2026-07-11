# Useful commands

# Final render
quarto render data-slug.qmd --output-dir "$env:NOLE_NEXUS\Nole-Nexus Website\Workbench"

# Generate the environment history yml
mamba env export --from-history > environment.yml
# Generate personal envrionement file
mamba env export --no-builds > my_environment.yml
# Merge a --from-history environment.yml with versions from my_environment.yml (full export)
mamba run python scripts/utils/merge_env_versions.py --environment environment.yml --full my_environment.yml --out environment.yml

