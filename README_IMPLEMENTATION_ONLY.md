# HRI2027 Scene Graph Implementation

This branch contains implementation code for the OpenVLA/LIBERO scene-graph auxiliary learning experiments.
Experiment result files, generated artifacts, checkpoints, datasets, and paper reports are intentionally excluded.

## Included

- `graph_internalization/`
  - Graph auxiliary wrapper, teacher loader, graph loss integration, sidecar dataset utilities.
- `scene_graph_generator/`
  - Graph Generator models, losses, decoding, metrics, token selection, feature extraction, and training/evaluation scripts.
- `scene_graph_gt_v0/`
  - LIBERO scene-graph ground-truth generation utilities and relation rules.
- `scripts/`
  - Rollout collection, OpenVLA feature caching, Graph Generator training/evaluation, scene-graph generation, and validation scripts.
- `configs/`
  - Training/evaluation configuration files.
- `tests/`
  - Lightweight tests for token layout, graph losses, graph schema, rollout manifests, and scene-graph rules.

## Excluded

- `data/`
- `outputs/`
- `reports/`
- `artifacts/`
- `checkpoints/`
- `runs/`
- generated videos/images
- paper drafts and result summaries

## OpenVLA Core Changes

OpenVLA model-code changes are kept in the companion repository:

```text
https://github.com/shin111mura-blip/openvla-hri2027/tree/bbox-scene-graph-10pct
```

Important files in that repository include:

- `prismatic/extern/hf/modeling_prismatic.py`
- `prismatic/vla/token_layout.py`
- `vla-scripts/finetune.py`
- `prismatic/models/scene_graph_heads.py`
- `prismatic/models/bbox_token_encoder.py`

