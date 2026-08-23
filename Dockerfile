# syntax=docker/dockerfile:1

# Two stages, and only because the wheels earn it. Installing them wants uv, a
# resolver cache and a few hundred megabytes of scratch, none of which has any
# business in the image that ships; what ships is the resulting venv.
FROM python:3.12-slim-bookworm AS build

COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_PYTHON=python3.12 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies before source, so editing a module does not re-download torch.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev


FROM python:3.12-slim-bookworm

# torch's CPU kernels want libgomp at load time. Nothing else is missing,
# because opencv is the headless build -- which is why pyproject pins that one.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 aoi

WORKDIR /app

COPY --from=build /app/.venv /app/.venv
# The project is installed editable against /app/src, and DeepPCB's default
# root is resolved relative to that layout, so the source tree has to survive
# into this stage rather than being folded into site-packages.
COPY src/ src/
COPY scripts/ scripts/

# Nothing heavy is baked in. The store, the checkpointer, the Chroma index, the
# patches, the dataset and the trained weights are all built by scripts and are
# gitignored; they arrive on these two mounts. Unmounted, the station starts
# against an empty queue, which is the honest failure rather than a crash.
RUN mkdir -p /app/data /app/models && chown -R aoi:aoi /app/data /app/models

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1

USER aoi
EXPOSE 8000

# The CLI defaults to 127.0.0.1, which from outside a container is nowhere.
CMD ["python", "-m", "aoi_agent", "station", "--host", "0.0.0.0", "--port", "8000"]
