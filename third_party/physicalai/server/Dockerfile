ARG OPENVINO_VERSION_MAJOR=2026.2.1
ARG OPENVINO_VERSION_FULL=2026.2.1.21919.ede283a88e3
ARG UBUNTU_VERSION=24.04

# Intel GPU driver versions. https://github.com/intel/compute-runtime/releases
ARG IGC_VERSION=v2.36.3
ARG IGC_VERSION_FULL=2_2.36.3+21719
ARG COMPUTE_RUNTIME_VERSION=26.22.38646.4
ARG COMPUTE_RUNTIME_VERSION_FULL=26.22.38646.4-0
ARG IGDGMM_VERSION=22.10.0

# Intel NPU driver versions. https://github.com/intel/linux-npu-driver/releases
ARG NPU_DRIVER_VERSION=v1.33.0
ARG NPU_DRIVER_FULL=v1.33.0.20260529-26625960453
ARG LIBZE1_VERSION=1.27.0-1~24.04~ppa2

## Build Image
FROM docker.io/ubuntu:${UBUNTU_VERSION}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        gnupg \
        wget \
        curl \
        git \
        cmake \
        ninja-build \
        build-essential \
        libtbb12 \
        libssl-dev \
        ocl-icd-opencl-dev \
        opencl-headers \
        opencl-clhpp-headers \
        intel-opencl-icd \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        python3-pip \
        python-is-python3 && \
    rm -rf /var/lib/apt/lists/*

# OpenVINO toolkit and GPU/NPU drivers are cached via BuildKit cache mounts to avoid re-downloading on rebuilds.
# Install OpenVINO for Ubuntu 24.04.
ARG OPENVINO_VERSION_MAJOR
ARG OPENVINO_VERSION_FULL
RUN --mount=type=cache,target=/var/cache/openvino,sharing=locked \
    mkdir -p /opt/intel && \
    TGZ=/var/cache/openvino/openvino_toolkit_ubuntu24_${OPENVINO_VERSION_FULL}_x86_64.tgz && \
    if [ ! -f "$TGZ" ]; then \
        wget -O "$TGZ" https://storage.openvinotoolkit.org/repositories/openvino/packages/${OPENVINO_VERSION_MAJOR}/linux/openvino_toolkit_ubuntu24_${OPENVINO_VERSION_FULL}_x86_64.tgz; \
    fi && \
    tar -xf "$TGZ" -C /opt/intel/ && \
    mv /opt/intel/openvino_toolkit_ubuntu24_${OPENVINO_VERSION_FULL}_x86_64 /opt/intel/openvino_${OPENVINO_VERSION_MAJOR} && \
    cd /opt/intel/openvino_${OPENVINO_VERSION_MAJOR} && \
    echo "Y" | ./install_dependencies/install_openvino_dependencies.sh && \
    cd - && \
    ln -s /opt/intel/openvino_${OPENVINO_VERSION_MAJOR} /opt/intel/openvino

ENV OpenVINO_DIR=/opt/intel/openvino

RUN apt-get update \
    && apt-get install -y libgomp1 libtbb12 curl wget ffmpeg ocl-icd-libopencl1 \
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete

# Install GPU drivers
ARG IGC_VERSION
ARG IGC_VERSION_FULL
ARG COMPUTE_RUNTIME_VERSION
ARG COMPUTE_RUNTIME_VERSION_FULL
ARG IGDGMM_VERSION
RUN --mount=type=cache,target=/var/cache/intel-gpu,sharing=locked \
    set -eux; \
    cd /var/cache/intel-gpu; \
    for url in \
        https://github.com/intel/intel-graphics-compiler/releases/download/${IGC_VERSION}/intel-igc-core-${IGC_VERSION_FULL}_amd64.deb \
        https://github.com/intel/intel-graphics-compiler/releases/download/${IGC_VERSION}/intel-igc-opencl-${IGC_VERSION_FULL}_amd64.deb \
        https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/intel-ocloc_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.deb \
        https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/intel-opencl-icd_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.deb \
        https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/libigdgmm12_${IGDGMM_VERSION}_amd64.deb \
        https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/libze-intel-gpu1_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.deb ; do \
        f=$(basename "$url"); \
        [ -f "$f" ] || wget -q -O "$f" "$url"; \
    done; \
    apt-get update; \
    apt-get install -y --no-install-recommends ./*.deb; \
    rm -rf /var/lib/apt/lists/*

# Install NPU userspace drivers
ARG NPU_DRIVER_VERSION
ARG NPU_DRIVER_FULL
ARG LIBZE1_VERSION

RUN --mount=type=cache,id=intel-npu-${NPU_DRIVER_VERSION},target=/var/cache/intel-npu,sharing=locked \
    set -eux; \
    \
    NPU_CACHE=/var/cache/intel-npu; \
    NPU_ARCHIVE="linux-npu-driver-${NPU_DRIVER_FULL}-ubuntu2404.tar.gz"; \
    NPU_URL="https://github.com/intel/linux-npu-driver/releases/download/${NPU_DRIVER_VERSION}/${NPU_ARCHIVE}"; \
    TGZ="${NPU_CACHE}/${NPU_ARCHIVE}"; \
    \
    LIBZE_ARCHIVE="libze1_${LIBZE1_VERSION}_amd64.deb"; \
    LIBZE_URL="https://snapshot.ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu/20260324T100000Z/pool/main/l/level-zero-loader/${LIBZE_ARCHIVE}"; \
    LIBZE_DEB="${NPU_CACHE}/${LIBZE_ARCHIVE}"; \
    \
    mkdir -p "${NPU_CACHE}"; \
    \
    # Delete and redownload the NPU archive if it is missing or corrupted.
    if ! tar -tzf "${TGZ}" >/dev/null 2>&1; then \
        echo "NPU archive is missing or corrupted; downloading again"; \
        rm -f "${TGZ}" "${TGZ}.tmp"; \
        wget \
            --https-only \
            --tries=5 \
            --timeout=30 \
            --progress=dot:giga \
            -O "${TGZ}.tmp" \
            "${NPU_URL}"; \
        test -s "${TGZ}.tmp"; \
        tar -tzf "${TGZ}.tmp" >/dev/null; \
        mv "${TGZ}.tmp" "${TGZ}"; \
    fi; \
    \
    # Delete and redownload libze1 if it is missing or not a valid Debian package.
    if ! dpkg-deb --info "${LIBZE_DEB}" >/dev/null 2>&1; then \
        echo "libze1 package is missing or corrupted; downloading again"; \
        rm -f "${LIBZE_DEB}" "${LIBZE_DEB}.tmp"; \
        wget \
            --https-only \
            --tries=5 \
            --timeout=30 \
            --progress=dot:giga \
            -O "${LIBZE_DEB}.tmp" \
            "${LIBZE_URL}"; \
        test -s "${LIBZE_DEB}.tmp"; \
        dpkg-deb --info "${LIBZE_DEB}.tmp" >/dev/null; \
        mv "${LIBZE_DEB}.tmp" "${LIBZE_DEB}"; \
    fi; \
    \
    INSTALL_DIR="$(mktemp -d)"; \
    tar -xzf "${TGZ}" -C "${INSTALL_DIR}"; \
    cp "${LIBZE_DEB}" "${INSTALL_DIR}/"; \
    \
    cd "${INSTALL_DIR}"; \
    test -n "$(find . -maxdepth 1 -name '*.deb' -print -quit)"; \
    apt-get update; \
    apt-get install -y --no-install-recommends ./*.deb; \
    \
    cd /; \
    rm -rf "${INSTALL_DIR}" /var/lib/apt/lists/*

# Install Python packages into the system Python environment.
RUN python3 -m pip install \
        --no-cache-dir \
        --break-system-packages \
        "numpy==2.4.6" \
        "physicalai==0.1.1" \
        "pyzmq" \
        "msgpack==1.2.1" && \
    python3 -m pip install \
        --no-cache-dir \
        --break-system-packages \
        --upgrade \
        --force-reinstall \
        --no-deps \
        "openvino==2026.2.1" \
        "openvino-tokenizers==2026.2.1.0"

COPY . /workspace/
WORKDIR /workspace

CMD ["python3", "run_server.py"]
