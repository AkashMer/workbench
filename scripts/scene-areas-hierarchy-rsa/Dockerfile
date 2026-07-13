FROM nipreps/fmriprep:25.2.0

# Install a minimal conda distribution
RUN curl -L -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh

# Build the post env from the environment spec
COPY environment.yml /tmp/environment.yml
RUN /opt/conda/bin/mamba env create -f /tmp/environment.yml

# Make the scene-areas-hierarchy-rsa env's own binaries (python, etc.) the default on PATH
ENV PATH="/opt/conda/envs/scene-areas-hierarchy-rsa/bin:/opt/conda/bin:$PATH"

# Register the env's ipykernel as "fmri"
RUN python -m ipykernel install --name fmri --display-name fmri

# Install Quarto CLI version: 1.9.38
RUN curl -L -o /tmp/quarto.tar.gz https://github.com/quarto-dev/quarto-cli/releases/download/v1.9.38/quarto-1.9.38-linux-amd64.tar.gz \
    && mkdir -p /opt/quarto \
    && tar -xzf /tmp/quarto.tar.gz -C /opt/quarto --strip-components=1 \
    && rm /tmp/quarto.tar.gz
ENV PATH="/opt/quarto/bin:$PATH"

WORKDIR /workbench

# Override the base image's fmriprep entrypoint so the script below runs directly
ENTRYPOINT []

# Copy scripts/ into the container's own filesystem, render there, then copy
# the output back out to the mounted scripts/ folder
RUN mkdir -p /build/scripts
COPY render_entrypoint.sh /usr/local/bin/render_entrypoint.sh
RUN chmod +x /usr/local/bin/render_entrypoint.sh
CMD ["/usr/local/bin/render_entrypoint.sh"]
