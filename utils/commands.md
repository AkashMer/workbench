# Useful commands

# Final render (posts without a Dockerfile)
quarto render data-slug.qmd --output-dir "$env:NOLE_NEXUS\Nole-Nexus Website\Workbench"

# Run from inside post-slug
# Generate the environment history yml (Out-File -Encoding utf8, not >, to avoid UTF-16)
mamba env export --from-history | Out-File -Encoding utf8 environment.yml
# Generate personal envrionement file
mamba env export --no-builds | Out-File -Encoding utf8 my_environment.yml
# Merge a --from-history environment.yml with versions from my_environment.yml (full export)
# From the workbench root
mamba run python utils/merge_env_versions.py --slug post-slug --environment environment.yml --full my_environment.yml --out environment.yml
# From post-slug/scripts
mamba run python ../../utils/merge_env_versions.py --environment environment.yml --full my_environment.yml --out environment.yml

# Docker build (run from inside post-slug/scripts)
docker build -t -post-slug- .

# Final render (posts with a Dockerfile, run from the workbench root)
docker run --rm `
  -v "${PWD}\post-slug:/workbench" `
  -v "$env:NOLE_NEXUS\Nole-Nexus Website:/Nole-Nexus Website" `
  -post-slug-
