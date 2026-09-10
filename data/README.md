# data/ (git 제외)

```
data/<dataset_id>/
  raw/            E57, mp4, 설계 중심선 — 로컬에만 존재. 파드에 올리지 않는다.
  dataset/        데이터셋 계약 (docs/ARCHITECTURE.md §4). 파드로 가는 유일한 디렉토리.
  runs/<run_id>/  ckpt, point_cloud/*.ply (LOCAL_METRIC), log, run.json
  eval/<eval_id>/ mesh, geometry.json, sections.json, volume.json, eval.json
  export/<id>/    .spz (.splat 은 legacy)
```

합성 데이터셋으로 계약을 확인하려면:

```
minegs dataset synthetic data/synthetic_tunnel --length-m 120
minegs dataset validate data/synthetic_tunnel/dataset
```
