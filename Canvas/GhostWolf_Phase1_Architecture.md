### GhostWolf Voxelated Ray Inference Engine — Phase 1 Architecture

## Overview
GhostWolf is a local-first Windows 11 application that replaces transformer self-attention with a 4D voxel raytracing engine across space-time slices. It voxelizes open-weight LLMs into sparse grids and performs inference by traversing voxel volumes with time-sliced rays (past/present/future), collapsing 120 perspective rays into a mean token path. The system runs on AMD AI 9 370HX (CPU + GPU + NPU), packaged as a signed .exe with a desktop shortcut.

## Goals and Non-Goals
- **Goals**:
  - Local, offline inference; no network dependencies.
  - 4D voxelized inference: (x, y, z) + time slice (t).
  - 120-way ray split across past/present/future; mean collapse to output token.
  - Efficient CPU/GPU/NPU scheduling and prefetch.
  - Polished Windows 11 GUI (Orange/Black/White).
  - Support GGUF, ONNX, safetensors models.

- **Non-Goals**:
  - Training or fine-tuning.
  - Exact replication of AMD GAIA internals.
  - Cloud services.

## System Architecture (High-Level)
```
[GUI (Qt/QML)]  <->  [C++ Host App]  <->  [Voxel Engine Core (C++/Rust)]
                                  \        /             \
                                   [Embedded Python Orchestrator]  <->  [Model Adapters (GGUF/ONNX/safetensors)]
                                                                                      |
                                                                            [Voxelizer Pipeline]
                                                                                      |
                                                                                  [Voxel Grids]
                                                                                      |
                                          [NPU Ray Prefetch] <-> [GPU Voxel Math & Raster] <-> [CPU Orchestration]
```

## Hardware Mapping
- **CPU**: App host, orchestration, control loop, memory management, IO, tokenizer, scheduling, logging.
- **GPU** (DirectX 12 compute or Vulkan): Voxel math kernels, sparse grid ops, reductions, rasterization of ray segments, pre/post transforms.
- **NPU** (DirectML via ONNX Runtime EP): Async voxel neighborhood prefetch, ray traversal primitives, line integrals, gather/scatter, MLP-style post-projections.
- **Memory**: Unified orchestration; pinned host buffers; device-local pools; double/triple buffering for rays.

## Software Stack
- **Core Engine**: C++20 (optionally Rust for kernels), DX12 compute; DirectML for NPU; ONNX Runtime (DirectML EP); optional Vulkan compute backend.
- **Orchestrator**: Python 3.11; `pybind11`/`nanobind` bridge to engine; tokenization and model loading.
- **GUI**: Qt 6 + QML (Fluent-like styling); Orange highlights, Black background, White text.
- **Packaging**: CMake (engine), Python wheels/venv embed, NSIS/WiX installer; `signtool` for signing.

## Module Breakdown
- **Orchestrator** (`ghostwolf.orchestrator` in Python)
  - Model discovery and loading (GGUF, ONNX, safetensors).
  - Tokenizer integration (SentencePiece/BPE).
  - Voxelizer control and pipeline staging.
  - Generation loop control (step/stream/cancel).

- **Voxelizer** (Engine + Python front)
  - Maps embeddings, attention weights, and layer semantics into sparse 3D voxel grids with a 4th time dimension.
  - Grid structure: Hashed sparse grid with chunked bricks (e.g., 32³) + t slices; compress to half/uint8 with scale.
  - Quantization and projection of high-D embeddings into 3D coordinates.

- **4D Raytracing Engine** (Engine)
  - Ray synthesis: 120 rays per token step, split into 3 groups of 40 (past/present/future).
  - Traversal: Space-time voxel sampling with trilinear + t-slice interpolation.
  - Aggregation: Weighted line integrals → per-ray token logits → mean collapse.
  - Prefetch: NPU queues voxel neighborhoods ahead of GPU math.

- **Scheduler/Runtime** (Engine)
  - Job graph for steps: tokenize → ray synthesis → prefetch → traverse → reduce → sample.
  - Device placement and overlap (CPU prep, NPU prefetch, GPU math).
  - Priority lanes for UI responsiveness.

- **Memory Manager** (Engine)
  - Pools for device-local voxel chunks and ray buffers.
  - Lock-free rings for prefetch queues.
  - Mapped files for large model weights.

- **Model Adapters**
  - GGUF: via llama.cpp loader in Python; export tensors to voxelizer.
  - ONNX: ONNX Runtime for weight access; DirectML EP configured for NPU.
  - safetensors: Python loader; zero-copy into voxelizer buffers where possible.

- **GUI** (Qt/QML)
  - Panels: Model selector, Voxelizer status, Inference controls, System monitor.
  - Live charts: CPU/GPU/NPU utilization; TPS; latency histograms.
  - Theme: Futuristic flat UI with Orange accents on Black, White text.

- **Telemetry/Logging**
  - Local structured logs; session artifacts; no network.
  - Optional perf captures (ETW) for advanced profiling.

- **Config**
  - `ghostwolf.toml`: model paths, cache sizes, device prefs, theme overrides.

- **Installer**
  - NSIS/WiX script: bundles engine, embedded Python, runtimes.
  - Desktop shortcut “GhostWolf Engine”; Start menu entry.
  - Optional elevation for driver/runtime installation.

## Data Flow (Per Request)
1. Input prompt → tokenizer (Python) → token IDs.
2. Orchestrator sets inference params (ray depth, time splits, temperature, max tokens).
3. Engine voxelizer ready-state: ensures grids in device memory and cache warm.
4. Step loop:
   - Ray Synthesis (CPU): build 120 rays grouped by t-perspective.
   - Prefetch (NPU): enqueue voxel neighborhoods based on predicted ray segments.
   - Traverse (GPU+NPU): sample voxels, integrate features, compute per-ray logits.
   - Collapse (GPU): average mean path across 120 rays into a single logit vector.
   - Sample (CPU): produce next token; update context; shift time window.
5. Stream token to GUI; repeat until stop condition.

## Voxelization Pipeline
- **Embedding Projection**:
  - Learnable or PCA-initialized projection \( W \in \mathbb{R}^{d \times 3} \) mapping embeddings to (x, y, z).
  - Coordinate normalization to grid bounds; quantize to voxel indices; store residuals as features.

- **Layer/Weight Encoding**:
  - Attention/value tensors folded into per-voxel feature channels using spatial hashing with per-layer offsets.
  - Sparse brick structure with per-brick scale/zero for quantized features (FP16/INT8).

- **Time Dimension (t)**
  - Discrete slices represent past (negative), present (0), future (positive).
  - Sliding window in t with ring-buffered slices; future slices bootstrap from short-horizon predictors or prior step extrapolations.

- **Caches**
  - Hot bricks near current ray tips pinned device-local.
  - Read-mostly weights memory-mapped; demand-paged into GPU/NPU pools.

## 4D Ray Inference (Algorithm)
- **Ray Split**:
  - 120 rays = 40 past, 40 present, 40 future; equal group weighting.
  - Within group, rays fan via stratified directions sampled from a deterministic low-discrepancy set.

- **Traversal**:
  - Each ray traverses through 3D bricks and across a t-slice schedule (group-specific: past drifts t-, present t≈0, future drifts t+).
  - Sampling uses trilinear interpolation in space and linear interpolation in time.

- **Aggregation**:
  - Per-ray integral: sum of sampled features weighted by distance falloff and positional embedding gates.
  - Map integrated feature to logits via small MLP (NPU) or GEMV (GPU).

- **Collapse**:
  - Compute mean logit vector across 120 rays:
    - Group weights: \( w_{\text{past}}=w_{\text{present}}=w_{\text{future}}=\frac{1}{3} \).
    - Uniform within-group averaging; final logits are the weighted mean.
  - Temperature/top-k/top-p applied on CPU to sample next token.

- **Coherence**:
  - Exponential moving average on directional fields to stabilize ray orientations between steps.
  - Optional contrastive gating to suppress self-intersections.

## Scheduling & Concurrency
- **Double-buffering** for:
  - Prefetch queues (NPU), traversal batches (GPU), collapse buffers (GPU).
- **Overlap**:
  - t: step n prefetch runs while GPU computes step n-1 traversal; CPU samples step n-2.
- **Work Graph**:
  - Small DAG executor assigns device, stream, and events; back-pressure via lock-free queues.

## Memory Management
- **Voxel Storage**:
  - Hashed grid → bucketized into 32³ bricks × T slices; per-brick compression.
  - Index tables placed in CPU and mirrored in GPU/VRAM; NPU retains working set via DML resources.

- **Buffers**:
  - Pinned host buffers for DMA; circular rays buffer; scratch for reductions.
  - Heuristic cache size targets based on VRAM/NPU SRAM capacities.

## APIs and Interfaces
- **Engine C ABI** (hosted by C++; consumed by Python via `pybind11`):
```c
// C-style for stable FFI
typedef void* GW_Handle;

typedef enum { GW_GGUF, GW_ONNX, GW_SAFETENSORS } GW_ModelFormat;

GW_Handle gw_create();
int gw_load_model(GW_Handle h, const char* path, GW_ModelFormat fmt);
int gw_voxelize(GW_Handle h, const char* config_toml);
int gw_infer_begin(GW_Handle h, const char* prompt_json);
int gw_infer_step(GW_Handle h, /*out*/ int* token_id, /*out*/ float* logprob);
void gw_cancel(GW_Handle h);
void gw_destroy(GW_Handle h);
```

- **Python Orchestrator**:
```python
class GhostWolfOrchestrator:
    def load_model(self, path: str, fmt: str) -> None: ...
    def voxelize(self, config: dict) -> None: ...
    def generate(self, prompt: str, settings: dict):  # yields tokens
        ...
```

- **GUI Bridge**:
  - Qt C++ wrapper exposes `Q_INVOKABLE` methods calling engine; signals stream tokens and perf metrics to QML.

## Performance Targets (initial)
- Startup (warm cache): < 5 s on supported hardware.
- Prefetch latency (p95): < 2 ms chunk fetch to NPU for typical rays.
- Traversal throughput: ≥ 120 rays per token step with overlap; ≥ 10–20 tok/s for 7B-class models (target, tune later).
- VRAM footprint: adaptive; aim for ≤ 6–8 GB for 7B-class with compressed bricks.

## Theme & UX Notes
- Fluent-like QML; Orange accent (`#FF6A00`), Black surfaces (`#0E0E10`), White text (`#FFFFFF`).
- Live utilization dials for CPU/GPU/NPU; streaming token output panel; voxelizer progress bar; ray parameters panel.

## Packaging & Deployment
- CMake builds engine and Qt app; embeds Python runtime and site-packages.
- Installer (NSIS/WiX):
  - Installs runtimes (VC++ Redist, DX12/DML), creates desktop shortcut “GhostWolf Engine”.
  - Registers file associations for `*.ghostwolfproj`.
  - Signs binaries with `signtool`.

## Security & Privacy
- No network calls; offline by design.
- Local logs only; optional redact mode for prompts.
- Models stored under `%PROGRAMDATA%\GhostWolf\models` or user-selectable folder.

## Risks & Mitigations
- **NPU API variability**: use ONNX Runtime DirectML EP with capability probing; GPU fallback for kernels.
- **VRAM pressure**: aggressive sparse bricks + LRU eviction; streaming loads.
- **Numeric drift from voxelization**: per-layer calibration; small MLP heads trained offline (optional) or rule-based scaling.
- **GUI-Engine deadlocks**: strict async messaging + bounded queues.

## MVP Scope (Phase 2–5 impact)
- Single 7B-class model; GGUF and ONNX first; safetensors next.
- Basic voxelizer (embedding + key layers) with INT8/FP16.
- Core 120-ray traversal, mean collapse, streaming tokens.
- Qt GUI with essential panels; basic perf monitor.
- NSIS installer; unsigned in dev, signed for release.

## Future Extensions
- Multimodal voxel fields (vision/audio).
- Learned ray controllers (RL-based).
- Advanced brick compression (tensor-cores friendly).
- Project save/restore and prompt graphs.

## Open Questions
- Exact NPU operator coverage for 4D interpolation and custom reductions?
- Best default grid resolution and T-slice count per model size?
- Strategy for future rays initialization beyond short-horizon predictors?