# Harness image for contest.harness.workspace.v2 (contest_harness SDK 1.0.4).
# Build from the repository root:
#   docker build -f docker/Dockerfile -t zhixun-harness:1.0 .
# Everything is installed at build time; the image needs no network at run time. The model is reached
# only through the host gateway (rnd.model); no key, endpoint or external client is inside the image.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    BASELINE_CPU_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    AGENT_HARNESS_BASELINE=/opt/agent-harness/vendor/cpu-baseline

# The official baselines pin numpy 1.26.4; the same version keeps checkpoints byte-comparable.
RUN pip install numpy==1.26.4 PyYAML==6.0.3

COPY third_party/contest-harness-sdk-1.0.4 /opt/contest-harness
RUN pip install /opt/contest-harness

# agent_harness is pure Python: copied into site-packages, so this layer builds without network.
COPY src/agent_harness /usr/local/lib/python3.12/site-packages/agent_harness
COPY vendor /opt/agent-harness/vendor

COPY third_party/contest-harness-sdk-1.0.4/template/harness.py /app/harness.py
COPY submission/contest_submission.py /app/contest_submission.py
COPY submission/model_config.txt /app/model_config.txt

# Model identity of this submission. The PyTRiO key is NOT in the image: it is entered in the platform's
# key field and used by the host gateway only.
LABEL org.contest.team="AgentModel" org.contest.model="Qwen/Qwen3.8-27B" org.contest.sdk="contest_harness 1.0.4"
ENV AGENT_HARNESS_MODEL=Qwen/Qwen3.8-27B

WORKDIR /app
USER 65534:65534
ENTRYPOINT ["python3", "/app/harness.py"]
