# configs/

CLI 가 읽는 YAML. 모든 파일은 `schema_version` 을 가지며 `minegs.core.config` 의
migration registry 로 상위 버전으로 올라간다.

| 경로 | 스키마 | 용도 |
|---|---|---|
| `dataset/e57.yaml` | `E57IngestConfig` | E57 → dataset (§6.1) |
| `dataset/video.yaml` | `VideoIngestConfig` | 일반 영상 → dataset (§6.2) |
| `dataset/video360.yaml` | `VideoIngestConfig` | 360 영상 → 링 크롭 + rig → dataset (§6.2) |
| `eval/geometry_holdout.yaml` | `EvalConfig` | 형상·단면·체적 평가 (§5, §11) |
| `runner/local.yaml` | `RunnerConfig` | `LocalRunner` (§8.2) |
| `runner/runpod.yaml` | `RunnerConfig` | `RunPodRunner` (§8.2) |

학습 프로파일(light/heavy)은 패키지 안 `minegs/train/profiles/` 에 있다 (§8.3).
