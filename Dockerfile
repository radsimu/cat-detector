FROM debian:trixie-slim

# OS deps. ffmpeg for RTSP decoding, libgl/libglib for cv2, plus python.
# curl/gnupg/ca-certificates needed to add the Pi apt repo.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Add Pi apt repo for the Hailo runtime + Python bindings. Userspace only —
# the kernel driver (h10-hailort-pcie-driver) lives on the host; the
# container reaches the chip via /dev/hailo0 mounted in compose.
# We use the OpenPGP keyring file copied from the host (newer apt rejects
# the legacy SHA1 key shipped at archive.raspberrypi.com/debian/raspberrypi.gpg.key).
COPY raspberrypi-archive-keyring.pgp /usr/share/keyrings/raspberrypi-archive-keyring.pgp
RUN echo 'deb [signed-by=/usr/share/keyrings/raspberrypi-archive-keyring.pgp] http://archive.raspberrypi.com/debian trixie main' \
        > /etc/apt/sources.list.d/raspi.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        h10-hailort python3-h10-hailort \
 && rm -rf /var/lib/apt/lists/*

# Use a venv with system-site-packages so apt-installed hailo_platform
# remains importable while we pip-install everything else (PEP 668 in
# Trixie blocks pip directly into /usr).
RUN python3 -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# CPU-only torch first so ultralytics' install (in -e .) doesn't pull
# NVIDIA CUDA wheels (~2GB of dead weight on aarch64). torch is only used
# by the ultralytics fallback path; the Hailo path doesn't need it.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

# pyproject + source first → cacheable layer for code-only changes.
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# Project assets that don't need a rebuild trigger from src/
COPY templates/ ./templates/
COPY static/ ./static/

# Where ultralytics caches downloaded weights — points at a mounted volume
# so we don't redownload yolov8x.pt every container restart.
ENV YOLO_CONFIG_DIR=/app/data/ultralytics

EXPOSE 8000

# Healthcheck: HTTP-probe the root endpoint. Catches the "process alive
# but event loop frozen" case (Pi 5 Wi-Fi hangs, Hailo PCIe wedges, etc.)
# that restart: unless-stopped doesn't catch because the process didn't
# actually exit. willfarrell/autoheal in compose restarts on unhealthy.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:8000/', timeout=8); sys.exit(0)" || exit 1

CMD ["python", "-m", "cat_detector.main"]
