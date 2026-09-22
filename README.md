# minegs

지하 갱도 **metric** Gaussian Splatting 연구 프레임워크.
TLS(E57)·영상(일반/360) 두 입력 경로 → 하나의 데이터셋 계약 → gsplat 학습(로컬/RunPod 동일 이미지) →
정합·형상·단면·체적·change 평가 → Viser 시각화. 설계 문서는 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## 원칙 (요약)

| # | 원칙 | 코드에서 |
|---|---|---|
| 1 | 데이터셋 계약 하나 | `minegs/core/manifest.py` — 입력 경로가 무엇이든 `dataset/` 은 같은 모양 |
| 2 | 코드 한 벌, 목적별 런타임 | `docker/Dockerfile.gpu`(digest 로 pin) / `Dockerfile.cpu`(선택) |
| 3 | CLI 가 진실의 원천 | `minegs/cli/` 는 파이썬 API 의 얇은 껍데기 |
| 4 | 원본은 로컬에만 | `train/runner/sync.py` 는 `dataset/` 만 push, `raw/` 는 거부 |
| 5 | 좌표 프레임 3계층 | `core/frames.py` — TLS_GLOBAL(m) ↔ LOCAL_METRIC(m) 는 SE3, BACKEND_INTERNAL 은 어댑터가 역변환 |
| 6 | 평가 누수는 계약 수준에서 차단 | `eval/protocol.py` — manifest 의 split/initialization 으로 주장 가능 범위를 판정·거부 |
| 7 | 가우시안 중심은 표면이 아니다 | `eval/surface/` → `eval/geometry/` 순서 |
| 8 | 모든 산출물에 계보 | `core/provenance.py` — sha256·config_hash·git SHA·도구 버전·부모 ID |

## 설치

```bash
pip install -e ".[dev]"          # 코어 + 테스트 (CPU, 순수 numpy/scipy)
pip install -e ".[e57]"          # pye57 — E57 읽기      (§6.1)
pip install -e ".[pdal]"         # PDAL 대용량 타일링 — PDAL C++ 라이브러리 별도 필요 (§6.1)
pip install -e ".[video]"        # OpenCV, pycolmap       (§6.2, COLMAP ≥ 4.0 바이너리 별도)
pip install -e ".[viz]"          # Viser 뷰어             (§12)
pip install -e ".[train]"        # torch + gsplat — 보통은 docker/Dockerfile.gpu 사용
```

## Phase 0A 게이트 — 합성 데이터셋

```bash
minegs dataset synthetic data/synthetic_tunnel --length-m 120
minegs dataset validate  data/synthetic_tunnel/dataset --strict
minegs dataset info      data/synthetic_tunnel/dataset
minegs eval    protocol  data/synthetic_tunnel/dataset
# → protocols: ['novel_view', 'geometry_holdout']

# 단면 간격·슬랩 두께·각도 bin 은 점밀도에 맞춰야 한다. 출력의 "valid n/N" 로 확인할 것
minegs eval sections data/synthetic_tunnel/raw/tls_full.ply data/synthetic_tunnel/dataset \
       --interval-m 2 --thickness-m 0.5 --angle-bins 72 --out sections.json
# 원시 PLY 로 자른 단면은 raw_cloud 로 기록되어 volume_accuracy 를 만들 수 없다 (§1C)
minegs eval volume   sections.json data/synthetic_tunnel/dataset --design-radius-m 2.4 --diagnostic
# claim 을 담는 geometry 는 surface artifact 를 요구한다 (§1.7). 원시 PLY 는 --diagnostic 전용
minegs eval geometry <surface_dir> data/synthetic_tunnel/dataset --tls-ply data/synthetic_tunnel/raw/tls_full.ply
minegs train command data/synthetic_tunnel/dataset --profile light     # 실행할 gsplat 커맨드 확인
pytest                                                                 # CI 와 동일
```

합성 갱도(반경 2.5 m, 곡선, UTM 급 오프셋)에서 단면 면적·∫A(s)ds·여굴 체적이 해석값과 **0.2 % 이내**로
일치하는 것을 `tests/test_eval.py` 가 게이트(`rel=0.002`)로 강제한다. 실제 오차는 시드와 무관하게 0.128 % 이며,
이는 72각형 내접 다각형의 면적비 (n/2π)·sin(2π/n) 에서 오는 이산화 오차다. `reconstruction` 프로토콜 데이터셋에 형상 정확도를
요청하면 CLI 는 exit code 3 으로 거부한다(`--diagnostic` 은 주장 없는 수치만 허용).

## 디렉토리

```
minegs/
  core/      config(schema_version + migration) · manifest · frames(SE3/Sim3) · centerline · chunking · provenance · pointcloud(PLY) · synthetic
  ingest/
    common/  geometry(PanoConvention) · equirect(링 크롭) · colmap_io(rigs.txt/frames.txt 포함)
    e57/     _nodes(단일 pye57 seam) · inventory+models+exceptions(0B.1) · images+mapping(0B.2 증거 기반 매핑) · extract(0B.3 추출·마스크·manifest) · scan_split(deprecated) · tiles(PDAL) · pose_to_colmap · pano/{E57Embedded,ExternalJpeg,VendorExport}
    video/   frames(ffmpeg) · dedup_blur · masks · rig(360 → COLMAP rig) · sfm/{COLMAPIncremental,COLMAPGlobal,GLUEMAP(exp)}
  train/     staging(쓰기 가능 복사본 + max_images 서브셋 + init_points→points3D)
    backends/  base(BackendCapabilities + capability_notes) · gsplat(executable contract)
    runner/    base · local(docker) · runpod(Phase 6, fail-closed) · sync(rclone)
    profiles/  light.yaml · heavy.yaml
  eval/      protocol · register(Sim3 → ICP → diagnostics) · surface(render=gsplat depth · depth 역투영 → surface artifact) · geometry(양방향) · sections(A(s) + section artifact) · volume(∫A ds gap-safe, 설계대비, coverage) · change · render(PSNR/SSIM/LPIPS)
  viz/       viewer(Viser) · overlay(규약 캘리브레이션 = 골든 게이트) · compare · export(.spz/.splat)
  cli/       ingest / dataset / train / eval / viz / sync
docker/      Dockerfile.gpu · Dockerfile.cpu · entrypoint.sh
configs/     dataset/{e57,video,video360}.yaml · eval/geometry_holdout.yaml · runner/{local,runpod}.yaml
docs/        ARCHITECTURE.md(invariant) · ROADMAP.md(Phase·Gate·DoD)
data/        (git 제외) <dataset_id>/{raw,dataset,runs,eval,export}
```

## 실제 E57 검사 · 매핑 · 추출 (Phase 0B)

E57 파일이 실제로 무엇을 담고 있는지 **점군을 읽지 않고** 조사한다.

비용은 두 부분으로 나뉜다. **메타데이터 파싱은 O(scan 수)** 이고 점 배열을 전혀 적재하지 않으므로
수십 GB 파일에서도 메모리를 쓰지 않는다. 반면 **provenance 용 SHA-256 은 O(파일 크기)** 다 —
1 MB 씩 스트리밍하므로 메모리는 일정하지만, 50 GB 스캔이면 50 GB 를 읽는 시간이 든다.
빠르게 훑어보려면 `--no-hash` 를 쓴다(리포트에 건너뛴 사실이 기록된다).

```bash
pip install -e ".[e57]"          # pye57 만 있으면 된다 (휠 제공, 네이티브 빌드 불필요)

# Linux / macOS
minegs ingest e57 inventory /data/scan/tunnel.e57

# Windows (PowerShell / cmd)
minegs ingest e57 inventory D:\scan\tunnel.e57

# 결과를 JSON 으로도 저장 (매핑·추출로 넘길 입력)
minegs ingest e57 inventory D:\scan\tunnel.e57 --json inventory.json

# SHA-256(O(파일 크기))을 건너뛰고 메타데이터만 빠르게 훑어보기 — 리포트에 건너뛴 사실이 남는다
minegs ingest e57 inventory D:\scan\tunnel.e57 --no-hash
```

리포트는 scan 마다 다음을 그대로 보여준다: 결정적 식별자(`scan_000` / 후보 station `S000`),
점 개수, Cartesian·spherical·RGB·intensity·row/column 존재 여부, pose 상태, header bounds.
파일 수준 문제(예: 유효하지 않은 pose 를 선언한 scan)와 관찰(예: pose 가 전부 identity)을
구분해 마지막에 모아 출력한다.

**이 명령이 하지 않는 것**: 파노라마를 꺼내거나 station 에 연결하지 않고(`pano-map`),
점군을 추출하지 않으며(`extract`), E57 좌표를 TLS_GLOBAL 이라고 선언하지 않는다(Phase 0C).
pose 는 파일 자신의 `SOURCE` 프레임에 있는 `T_source_from_scan` 으로 보고된다.

### station ↔ 파노라마 매핑 (Phase 0B.2)

```bash
minegs ingest e57 pano-map tunnel.e57 --json pano_mapping.json
minegs ingest e57 pano-map tunnel.e57 --mapping mapping.csv --json pano_mapping.json
minegs ingest e57 pano-map tunnel.e57 --images-dir ./panoramas --mapping mapping.csv
```

매핑은 **증거가 있을 때만** 만들어진다. E57 자신의 `associatedData3DGuid` 가 정확히 한 scan 을
가리키면 `confirmed`, 캡처 소프트웨어가 만든 vendor index 도 `confirmed`, 사람이 쓴 매핑 파일은
`manual` 이다. scan/image index 가 같다거나, 개수가 같다거나, 파일 순서·이름이 비슷하다는 것은
근거로 쓰지 않는다 — 힌트로 출력될 수는 있어도 매핑이 되지는 않는다. 틀린 매핑은 오류를 내지
않고 학습도 되고 수렴까지 하는, 의미 없는 재구성이 되기 때문이다.

근거가 없으면 `unmapped`, 하나의 근거가 여러 scan 을 가리키면 `ambiguous`, 근거가 가리키는
scan 이 없으면 `orphan`, 사람이 쓴 매핑이 파일 자체 근거와 다르면 (덮어쓰기가 아니라) `conflict`
로 보고한다. E57 이 선언한 GUID 가 이 파일에 없을 때도 마찬가지다 — target 이 없다는 것은 파일의
진술이 해석 불가능하다는 뜻이지 진술이 없다는 뜻이 아니라서, 그 경우에도 매핑 파일이 이기지 않고
`conflict` 가 된다.

결과를 바꾸는 입력은 전부 hash 되어 리포트에 남는다: E57, 매핑 파일, vendor manifest, 외부
이미지 각각. 같은 E57 을 다른 CSV 로 매핑하면 다른 결과이므로 E57 만 적힌 provenance 로는 둘을
구분할 수 없다. `--no-hash` 는 전부에 일관되게 적용되고 이유를 남긴다.

매핑 파일은 헤더가 있는 CSV 또는 JSON 이고, 이미지 열
(`image_id`/`image_name`/`image_guid`) 과 대상 열(`scan_id`/`scan_guid`/`station_id`) 을 하나씩
갖는다:

```csv
scan_id,image_id
scan_000,image_002
scan_001,image_004
```

### scan · 이미지 추출 (Phase 0B.3)

```bash
minegs ingest e57 extract tunnel.e57 work/              # SOURCE 프레임, pose 필요
minegs ingest e57 extract tunnel.e57 work/ --voxel 0.01 --scan scan_000
minegs ingest e57 extract tunnel.e57 work/ --raw        # SCANNER 프레임, unregistered
```

```
work/
  inventory.json  pano_mapping.json  extraction_manifest.json
  scans/scan_000.ply  scan_000.pose.json
  images/image_000.jpg
```

**이것은 dataset 이 아니다.** 점은 아직 E57 자신의 `SOURCE` 프레임(또는 `--raw` 의 scan 별
`SCANNER` 프레임)에 있다. TLS_GLOBAL 선언, LOCAL_METRIC origin, train/test 분할,
`init_points.ply`, dataset manifest 는 전부 Phase 0C 다.

기본값(registered)은 모든 대상 scan 에 쓸 수 있는 pose 를 요구하고, 없으면 `--raw` 를 알려주며
거부한다. `--raw` 는 scanner 프레임 그대로 쓰고 `unregistered` 로 표시하며, 없는 pose 를 identity
로 채우지 않는다. 읽을 수 없거나 유효하지 않은 pose 는 두 경로 모두 거부한다.

invalid-state 마스크는 좌표·RGB·intensity·row/column 에 **동일하게** 적용된다(길이가 다른 컬럼은
하드 실패). 색 범위는 scan 의 `colorLimits` 에서 읽어 변환 사실을 manifest 에 남기고, 선언이
없으면 8-bit 라고 가정하지 않고 raw 로 보존한다. 임베디드 이미지는 spherical·cylindrical·pinhole
을 꺼내며(쓰는 것은 blob 바이트 복사이고, pinhole 의 *해석* 은 Phase 0C 다) 바이트가 선언한 코덱과
맞는지 확인한다 — visual-reference 등 의미가 선언되지 않은 representation 은 이유와 함께
`skipped_images` 에 기록되고 다른 projection 으로 재해석되지 않는다.

추출은 전부 성공했을 때만 publish 된다. run 은 `.<name>.minegs-partial` 임시 트리에 쓰이고
마지막에 `work_dir` 로 rename 되므로, 40개 중 12번째 scan 에서 실패해도 "scan 12개 + manifest
없음" 같은 완성처럼 보이는 디렉토리가 남지 않는다. `--overwrite` 는 이전 트리를 지우고 시작하는
것이 아니라 옆으로 옮겨 두었다가 새 run 이 성공한 뒤에 치우므로, 실패한 재실행이 직전의 정상
결과를 파괴하지 않는다. 이 추출기가 쓰지 않은 파일이 하나라도 있는 디렉토리는 `--overwrite`
여부와 무관하게 거부한다. inventory·pano_mapping·extraction_manifest 세 산출물은 한 번만 계산한
같은 source digest 를 공유한다 — `scan_000` 은 특정 파일 안의 index 라서, 어느 바이트를 읽었는지
말할 수 없는 산출물은 자기 ID 가 무엇을 가리키는지도 말할 수 없다.

pye57 에는 chunked reader 가 없어 scan 을 통째로 읽는다. 큰 scan 은 예상 peak memory 를 note 로
알려주고 `--max-scan-points` 로 fail-closed 할 수 있다. production 규모 타일링은
`minegs ingest e57 tiles` 의 PDAL 경로가 담당한다(PDAL C++ 라이브러리 필요; 없으면 fail-closed).

## Metric dataset 과 Golden Gate (Phase 0C)

0B staging tree 를 dataset 계약으로 materialize 하고, frame/camera convention 이 맞는지 **증명**한다.

```bash
minegs dataset calibrate-camera work/ --out camera_convention.json     # 24 축 규약 × 3 station, RGB residual
minegs dataset build-config-example > build.yaml                      # 편집: source_frame, split, centerline, holdout
minegs dataset from-e57 work/ data/tunnelA/dataset --config build.yaml
minegs dataset validate data/tunnelA/dataset --strict
minegs eval protocol data/tunnelA/dataset
minegs dataset golden-gate data/tunnelA/dataset --staging work/ --out data/tunnelA/golden_gate
minegs viz view data/tunnelA/dataset --golden-gate data/tunnelA/golden_gate
```

핵심 규칙 (docs/ROADMAP.md §Phase 0C):

* **SOURCE 는 TLS_GLOBAL 이 아니다.** `source_frame.mode: explicit_identity` (이 파일의 registered
  frame 을 survey 기준으로 채택한다는 *선언*) 또는 `explicit_transform` (4×4 SE(3)) 이 필요하다.
  선언 없이 identity 를 고르지 않는다. `TLS_GLOBAL` 은 측지 좌표계(UTM/EPSG) 를 뜻하지 않는다 —
  "여러 scan/camera 를 하나의 metric survey 좌표계로 표현하는 평가 기준 프레임" 이다.
* **LOCAL_METRIC** 은 translation-only (`R=I`, scale 1), 원점은 station centroid 를 0.1 m 로 round.
* **Pinhole intrinsics 는 E57 이 선언한 값에서만** (`fx = focalLength / pixelWidth`); 축 규약은
  `calibrate-camera` 가 **RGB 있는 scan 3 station 이상**에서 측정하거나 config 에 명시한다
  (Matterport 증거: `cam(+X,-Y,-Z)` = `diag(1,-1,-1)`). calibration artifact 는 source digest 로
  staging tree 에 묶인다.
* **Provenance 는 소비한 바이트를 기술한다**: 읽는 scan/image 마다 extractor 의 digest 와 대조하고
  불일치면 거부한다. `--overwrite` 는 이 도구가 쓴 dataset 만 교체한다.
* **Split 은 요청될 때만**, **geometry holdout 은 centerline 이 있을 때만**. holdout 구간 point 는 실제
  좌표를 centerline 에 투영해 `init_points.ply` 와 `points3D.txt` 양쪽에서 제거되고, publish 전에 PLY 를
  다시 읽어 확인한다.
* `golden-gate` 의 `structural_result` 는 수치 검사 결과이고 `real_data_validation_status` 는 항상
  `pending_human_inspection` 이다. G2 는 사람이 overlay 와 Viser 를 본 뒤에만 PASS 다.

## 데이터셋 계약 (§4)

```
dataset/
  images/            pinhole 이미지
  sparse/0/          cameras.txt · images.txt · points3D.txt · (rigs.txt · frames.txt)   — LOCAL_METRIC
  init_points.ply    초기 가우시안 위치, LOCAL_METRIC (헤더 comment 에 frame 기록)
  masks/             선택
  manifest.json      schema_version · dataset_id · coordinate_frames · capture_groups · split · initialization · provenance (+ 선택 항목)
```

`capture_groups` 는 광학중심/궤적 단위(`tls_station`, `trajectory_segment`, …). 렌더 평가는 그룹 단위,
**형상 홀드아웃은 chainage 구간 단위**(`split.geometry_holdout.chainage_ranges_m`) 이며
`initialization.excluded_chainage_ranges_m` 가 이를 덮지 않으면 형상 주장이 거부된다.

## 학습 (§8)

```bash
# 로컬 GPU (docker, digest 로 pin 된 이미지) — configs/runner/local.yaml
minegs train run data/<id>/dataset --profile light --runner local --config configs/runner/local.yaml
minegs train command data/<id>/dataset --profile light   # 실행할 커맨드만 확인 (GPU·trainer 불필요)
minegs train status  data/<id>/runs/<run_id>             # run.json: backend, digest, dataset_hash, staged, provenance
```

프로파일은 backend 플래그가 아니라 **capability** 를 요청한다 (`requests: {antialiasing: true, ...}`).
백엔드가 그 capability 를 *제공할 수 없으면* 이유와 함께 거부한다 (`capability_notes`).

**실행 계약 (gsplat v1.5.3 upstream 기준)**
- trainer 는 PyPI wheel 에 없고 저장소의 `examples/simple_trainer.py` 다. `docker/Dockerfile.gpu` 가 wheel 과 같은
  태그를 `/opt/gsplat` 에 체크아웃하고 `examples/requirements.txt` 를 설치한 뒤 `MINEGS_GSPLAT_TRAINER` 로 경로를 준다.
  스크립트를 못 찾으면 어댑터가 커맨드 생성을 거부한다(`minegs train command` 는 dry-run 이라 예외).
- 러너는 `dataset/` 을 read-only 로 마운트하고 `runs/<run_id>/staged/` 에 **스테이징 복사본**을 만든다
  (`minegs/train/staging.py`): 프로파일의 `max_images` 를 train 이미지에서 균등 추출하고, `init_points.ply` 를
  `sparse/0/points3D.txt` 로 써서 TLS 점으로 초기화한다. gsplat 파서가 `images_<factor>_png` 를 `data_dir` 안에 쓰기
  때문에 read-only 데이터셋을 직접 넘길 수 없다. 스테이징 내용(이미지 수·서브셋 여부·init 출처·sha256)은 `run.json` 에 기록된다.
- `absgrad` 는 `--strategy.absgrad`(default 전략 전용). `run.json` 의 `dataset_hash` 는
  manifest·sparse·init_points·**images·masks**·centerline 을 모두 덮는다.

**지금 거부되는 것 (fail-closed, 전체 목록은 [docs/ROADMAP.md](docs/ROADMAP.md) §6)**

| 요청 | 결과 | 이유 | 해제 |
|---|---|---|---|
| `normalize_world_space: true` | `ContractError` (exit 2) | BACKEND_INTERNAL = LOCAL_METRIC 이 baseline 계약. 재구현한 정규화는 upstream 과 equivalence 미검증이고, docker 경로에서는 host 가 볼 수 없는 경로로 변환을 계산하게 된다 | Phase 4 (equivalence test 후) |
| `depth_loss: true` | `ContractError` (exit 2) | upstream depth supervision 은 COLMAP image→point track 을 쓰는데 TLS 스테이징이 그 track 을 비운다 | Phase 4 (depth supervision 재설계) |
| **`--profile heavy`** | 위와 같은 이유로 `ContractError` | heavy 가 `depth_loss` 를 필수로 요청한다 | Phase 4 |
| `--runner runpod` | `NotYetImplementedError` (exit 4) | 미구현. 필요한 단계는 `runner/runpod.py` docstring | Phase 6 |

light 프로파일은 영향을 받지 않는다: `--no-normalize_world_space`, `--depth_loss` 없음,
`T_local_from_internal` 은 항등이다.

## Metric surface artifact (Phase 1A) · metric depth rendering (Phase 1B)

> 단면·체적까지의 증거 경계는 아래 [Section / volume evidence boundary (Phase 1C)](#section--volume-evidence-boundary-phase-1c) 에 이어진다.

**가우시안 중심은 표면이 아니다** (원칙 7). `runs/<run_id>/point_cloud/*.ply` 는 볼류메트릭 방사장의
파라미터이지 갱도 벽면의 샘플이 아니므로, 그것을 TLS 와 비교하면 "복원이 얼마나 정확한가" 가 아니라
"옵티마이저가 프리미티브를 어디에 두었는가" 를 재게 된다. 그래서 surface 는 **유도**하고, 유도 사실을
artifact 로 남긴다.

```bash
# Phase 1B — 학습된 run 에서 metric depth 를 직접 렌더링한다 (GPU 필요, CPU fallback 없음)
minegs eval render-depth data/<id>/runs/<run_id> data/<id>/dataset
# → runs/<run_id>/depth/{depth_manifest.json, <image stem>.npy, ...}

# Phase 1A — depth 를 LOCAL_METRIC surface sample 로 융합한다
minegs eval surface-depth data/<id>/runs/<run_id>/depth data/<id>/dataset \
       --run-dir data/<id>/runs/<run_id> --stride 2 --max-depth 30
# → runs/<run_id>/surface/depth_v001/{surface.json, surface_points.ply}

# manifest 가 검증되면 claim-bearing, 외부 depth 라면 --diagnostic 필요
minegs eval geometry data/<id>/runs/<run_id>/surface/depth_v001 data/<id>/dataset \
       --tls-ply data/<id>/raw/tls_full.ply
```

* **depth map 계약**: `<depth_dir>/<image stem>.npy`, 값은 **카메라 z 방향 미터**. PNG/EXR 는 후속 작업.
* **누락은 실패**: 카메라 view 수와 depth map 수가 다르면 `ContractError`. 조용히 빠진 view 는
  completeness·단면·체적에서 미복원 형상과 구별되지 않는다.
* **필터**: NaN·Inf·`z <= 0`·`> --max-depth` 는 제외. 결과는 항상 `LOCAL_METRIC`, 단위 m.
* **publish 는 원자적**: `.<name>.minegs-partial` 에 쓰고 성공 후 rename. 중단된 fusion 이 더 얇은
  surface 로 읽히지 않는다.
* **surface.json 은 정확도를 주장하지 않는다.** 어느 run·어느 dataset(`dataset_hash`)·어떤 depth
  (`depth_source`, `depth_sha256`)·어느 method 로 만들었고 어떤 점들(`point_sha256`)인지만 기록한다.
* **record 는 surface 가 아니다.** claim 경로는 `point_file` 을 다시 읽어 `point_sha256`·point
  count·frame·유한성을 확인한다. 이게 없으면 "surface 인가?" 는 "옆에 surface.json 이라는 파일이
  있는가?" 로 퇴화하고, Gaussian PLY 에 JSON 만 씌우면 claim 경로로 들어온다.
* **surface 라는 것만으로는 accuracy claim 이 안 된다.** `depth_source` 가 판단 기준이다.
  Phase 1A 가 받을 수 있는 것은 전부 `external_unverified` — 디렉터리에 놓인 `.npy` 는 그것이
  기록된 run 에서 나왔다는 증거를 하나도 갖고 있지 않고, 같은 depth 를 아무 성공한 run 에 붙여도
  동일한 artifact 가 나온다. 따라서 **Phase 1A surface 로는 `geometry_accuracy` 를 만들 수 없다.**
  `minegs_render` 는 Phase 1B renderer 가 run 과 묶인 depth 를 낼 때 쓰는 값이고, gate 는 지금
  써 둔다. 나중에 검사를 추가하는 게 아니라 증거를 만들어서 여는 구조다.
* **geometry gate 정리**

  | 입력 | `--diagnostic` 없이 | `--diagnostic` |
  |---|---|---|
  | 원시 PLY | `ContractError` — "Gaussian centres are not surfaces" | 경고 + `geometry_diagnostic` |
  | surface artifact, `external_unverified` | `ContractError` — depth 가 run 과 묶여 있지 않음 (Phase 1B) | 경고 + `geometry_diagnostic` |
  | surface artifact, `minegs_render` 인데 재유도 불가 | `ContractError` — 증거가 없다 | 경고 + `geometry_diagnostic` |
  | surface artifact, `minegs_render` 이고 재유도됨 | `geometry_accuracy` | `geometry_accuracy` |

  **`depth_source` 는 읽지 않고 다시 유도한다** (Phase 1C 에서 추가). 그 값은 fusion 시점에 한 번
  정해져 record 에 쓰이고, `check_surface` 는 점군 digest 만 다시 본다 — 그래서 `surface.json` 의
  문자열 한 줄을 고치면 임의의 PLY 가 claim 에 도달했다. 적대적 감사가 `raw/tls_full.ply` 자신을
  surface 로 위장해 `geometry_accuracy` 0.0 mm 를 얻는 것으로 재현했다. 이제 claim 경로는 surface 가
  기록한 depth/run 디렉터리에 대해 Phase 1B promotion 을 다시 돌리고, 돌아온 값을 쓴다.

  여기에 더해 **`--no-holdout-only` 는 claim 을 내리지 않는다**: `geometry_accuracy` 는 정의상
  holdout TLS 에 대한 주장이고, holdout 구간을 벗어난 수치는 run 이 초기화·학습에 쓴 형상을
  다시 재는 것이다. 그래서 경고와 함께 `geometry_diagnostic` 으로 보고한다.

### Phase 1B — 렌더링된 depth 가 증거가 되는 조건

`minegs eval render-depth` 는 gsplat rasterizer 를 `render_mode="ED"` 로 돌려 view 마다 카메라 +z
방향 metric depth 를 내고, 같은 패스에서 `depth_manifest.json` 을 쓴다.

* **manifest 가 곧 증거다**: `run_id · dataset_id · dataset_hash · backend · checkpoint(file,
  sha256, step) · renderer(name, version, settings) · frame · unit · view 마다 (image/camera id,
  해상도, `sha256`, `valid_ratio`, min/max)` · provenance.
* **승격 규칙**: `surface-depth` 는 depth 디렉터리에 manifest 가 있으면 그것을 **검증한다** — run,
  dataset hash, camera 해상도, 파일별 digest 가 전부 맞아야 `depth_source = minegs_render` 다.
  하나라도 어긋나면 강등이 아니라 **거부**한다. manifest 가 없으면 `external_unverified` 그대로다.
* **NaN 정책**: ray 가 `--min-alpha`(기본 0.5) 만큼 불투명도를 쌓지 못한 픽셀은 거리값이 없으므로
  **NaN** 이다. 0 을 쓰면 렌즈 위치에 표면이 생기고, far plane 을 쓰면 없는 벽을 만든다.
  `backproject_depth` 가 이미 NaN 을 버린다.
* **검증한 bytes = 소비한 bytes**: manifest 의 `file` 은 depth naming contract 가 정하는 이름과
  같아야 한다. 아니면 거부한다 — 한 파일의 digest 를 검사하고 다른 파일을 역투영하는 순간
  provenance 경계가 이름만 남는다.
* **checkpoint identity 도 검증한다**: manifest 의 `checkpoint.file`·`step` 이 run.json 의
  `final_checkpoint`·`checkpoint_step` 과 같아야 하고, 파일이 아직 있으면 digest 도 대조한다.
  같은 run 의 이전 checkpoint 로 렌더한 manifest 는 다른 모델을 기술하면서 나머지 검사를 전부
  통과한다.
* **renderer 가 재현하지 않는 run 은 거부한다**: `pose_opt`(학습 중 카메라 pose 를 갱신하므로
  dataset pose 는 더 이상 모델이 맞춰진 pose 가 아니다) 와 `antialiased`(opacity 누적 방식이
  달라져 expected depth 가 달라지고, equivalence 를 측정한 적이 없다). 증인 넷 — 실행된 argv,
  옆에 남아 있다면 trainer 자신의 `cfg.yml`, profile 의 capability requests, 그리고
  `backend_args` — 중 **하나라도** 해당하면 거부하고, checkpoint 에 `pose_adjust` 가 들어 있으면
  그것만으로도 거부한다. `normalize_world_space` 를 재구현 대신 거부한 것과 같은 이유다:
  equivalence 미검증 재현은 claim 경로의 증거가 될 수 없다.
* **`backend_args` 는 allowlist 로 본다**: profile 의 `backend_args` 는 trainer 에 그대로
  전달되므로(`build_command`), 알려진 나쁜 이름 목록으로는 `camera_model`·`with_ut`·`far_plane`
  같은 것을 놓친다. 그래서 *reasoned about 하지 않은 키는 거부*한다
  (`RENDER_NEUTRAL_BACKEND_ARGS`). light profile 은 전부 neutral 이라 그대로 렌더된다.
* **승격 시 재확인한다**: manifest 에는 metric frame 도 재현 가능 여부도 기록되지 않으므로,
  `build_depth_surface` 가 promotion 직전에 두 검사를 run record 에 대해 다시 돌린다.
  가드가 없던 빌드가 만든 depth 가 manifest 만으로 승격되지 않는다.
* **reference cloud 의 frame 도 검사한다**: `--tls-ply` 가 `TLS_GLOBAL` 이 아니면
  (`UNKNOWN` 포함) claim 경로에서 거부한다. `dataset/init_points.ply` 는 LOCAL_METRIC 이고
  `raw/tls_full.ply` 바로 옆에 있다 — 그 비교는 두 표면이 아니라 두 좌표계의 거리를 잰다.
* **checkpoint 는 `weights_only=True` 로 읽는다**: run artifact 는 GPU 호스트에서 가져오므로,
  계약 검사 *전에* pickle 이 코드를 실행하게 두지 않는다.
* **왜곡 있는 camera model 도 거부한다**: rasterizer 는 pinhole 로 투영하므로 `SIMPLE_RADIAL`
  등은 왜곡이 조용히 빠진 채 모든 ray 가 틀어진다. `PINHOLE`/`SIMPLE_PINHOLE` 만 지원.
* **image 실측 크기와 intrinsics 크기가 다르면 거부한다**: `fx,fy,cx,cy` 는 픽셀 값이므로
  `sparse/0` 와 `images/` 의 해상도가 다르면 그 intrinsics 는 그 이미지를 기술하지 않는다.
* **rasteriser 버전도 증거다**: manifest 의 `renderer.version` 이 `PINNED_GSPLAT` 과 다르면
  (`"not installed"` 포함) 승격하지 않는다. manifest 의 `backend` 도 run 의 것과 대조한다.
* **training backend 버전도 같은 pin 에 묶는다**: manifest 와 run 이 서로 일치하는 것만으로는
  부족하다 — 둘 다 gsplat 1.4 이고 renderer 만 1.5.3 이면 나머지 검사를 전부 통과한다.
  `train` extra 가 `gsplat>=1.4` 라 실제로 가능한 조합이므로, render 진입과 promotion 양쪽에서
  `run.backend["version"] == PINNED_GSPLAT` 을 확인한다.
* **checkpoint 내부 `step`** 도 run.json 의 `checkpoint_step` 과 대조한다.
* **fail closed**: run != succeeded · dataset id/hash 불일치 · checkpoint 없음/경로 깨짐 ·
  `T_local_from_internal` 이 항등이 아님(= backend 단위가 미터라고 보장 못 함) · backend 가
  `depth_render` 미선언 · 지원하지 않는 backend · CUDA 없음 · torch/gsplat 없음 · view 누락/중복/
  유령 view · image stem 충돌 · 해상도 불일치 · Inf 또는 음수 depth · 전 픽셀 empty ·
  출력 디렉터리 존재. **CPU fallback 은 없다.**

**아직 아닌 것**: `GsplatDepthRenderer` 자체는 이 저장소에서 **한 번도 실행된 적이 없다** — CI 에는
CUDA 도 gsplat 도 없다. 주변의 계약 검증은 전부 테스트되지만 rasterizer 호출은 아니다.
TSDF(`eval/surface/tsdf.py`)·mesh 재구성은 여전히 미구현이고, 실제 갱도 데이터의 과학적 검증도
수행하지 않았다. 구조적 검증만이다 (`tests/test_surface.py` T1–T10, `tests/test_depth_render.py`).

## Section / volume evidence boundary (Phase 1C)

Phase 1A·1B 가 `run → rendered depth → surface` 까지 닫았지만, `eval volume` 은 여전히 dataset
manifest 의 `judge()` 만 통과하면 `volume_accuracy` 를 줬다. manifest 는 *데이터셋* 이 그 주장을
지탱할 수 있다고 말할 뿐, 누가 넣은 어떤 구름인지에 대해서는 아무 말도 하지 않는다. 그래서
`raw/tls_full.ply` 로 자른 단면이 복원으로 자른 단면과 같은 주장에 도달했다. Phase 1C 는 체인을
끝까지 연장하고, 결측 구간을 가로지르는 적분을 금지한다.

```bash
# surface artifact 를 자르면 provenance 가 붙은 section artifact 가 나온다
minegs eval sections data/<id>/runs/<run_id>/surface/depth_v001 data/<id>/dataset \
       --interval-m 1 --thickness-m 0.5 --angle-bins 72 --out sections.json
minegs eval volume   sections.json data/<id>/dataset --design-radius-m 2.4
# → [volume_accuracy] V = ... m³ over 20.0-26.0 m
#    integrated 6.00 of 6.00 m (100.0%) in 1 segment(s); gaps: none
```

### Section artifact

`SectionRecord` (schema 1.0) 는 하나의 시계열에 대해 다음에 답한다 — **어느 dataset 의, 어느
verified surface 에서, 어느 run 에서, 어떤 depth provenance 로, 어떤 reference axis 를 따라,
어떤 parameters 로** 잘랐는가.

`section_id · dataset_id · dataset_hash · source{kind, surface_id, run_id, depth_source,
point_sha256, point_path} · reference_axis · reference_axis_sha256 · frame · unit · series ·
parameters · provenance`.

기록한 것은 전부 다시 대조한다 (`check_section_record`, `--diagnostic` 여부와 무관하게 실행):

| 기록 | 무엇과 대조하는가 |
|---|---|
| `dataset_id` · `dataset_hash` | 지금의 dataset |
| `reference_axis` | manifest 가 지금 선언하는 축 문자열 |
| `reference_axis_sha256` | 축 CSV 파일 자체의 digest |
| `series` 의 chainage 격자 | 지금의 centerline 과 기록된 parameters 로 다시 만든 station 격자 |
| `source.surface_id` · `run_id` · `point_sha256` · `depth_source` | surface artifact 가 아직 디스크에 있다면 그것 (`check_surface` 포함) |
| `source` 의 필드 존재 자체 | `kind` 가 요구하는 필드가 모두 채워져 있는가 (없으면 검사가 조용히 no-op 이 된다) |
| (claim 경로만) `series` 의 면적·반경·`empty_bins`·`n_points` | 검증된 surface 에서 **다시 잘라** 재현되는가 |
| (claim 경로만) `parameters.point_count` | 그 surface 가 실제로 가진 점 개수 |
| `radii_m` 의 길이 | `angle_bins` (없으면 claim 경로에서 numpy `ValueError` 로 터진다) |
| (claim 경로만) surface 의 `depth_source` | surface 가 기록한 depth/run 디렉터리에서 **다시 유도**되는가 |
| `series.frame` | record 의 `frame` (TLS_GLOBAL) |

축 digest 가 따로 필요한 이유: `DATASET_HASH_PATTERNS` 는 dataset 루트의 `centerline.csv` 만
덮는다. manifest 가 다른 경로를 가리키면 dataset hash 는 축 편집을 보지 못하고, chainage 는 다른
polyline 위에서는 다른 뜻이 된다.

### volume_accuracy 게이트

다음이 **동시에** 성립해야 한다. 하나라도 빠지면 거부, `--diagnostic` 이면 `geometry_diagnostic`.

1. dataset protocol 이 `VOLUME_ACCURACY` 를 허용한다;
2. 입력이 bare series 가 아니라 section artifact 다;
3. record 가 지금의 dataset·축·surface 와 일치한다 (위 표, 항상 검사);
4. `source.kind == "surface"` 이고 `depth_source == "minegs_render"` 다;
5. **series 를 그 surface 에서 다시 잘라 재현할 수 있고**, 슬랩이 station 간격보다 넓지 않다;
6. 적분이 선언된 geometry holdout 구간으로 제한된다;
7. 그 구간의 coverage 가 완전하다.

### 면적은 claim 경로에서 다시 계산된다

identity 검사는 record 를 옳은 dataset·옳은 축·옳은 surface 에 묶지만, **면적이 그 surface 에서
나왔다는 것은 말하지 않는다** — series 가 record 안에 들어 있으므로 `area_m2` 를 편집하거나,
invalid station 하나를 그럴듯한 값으로 뒤집어 gap 을 메워도 identity 검사는 전부 통과한다.

그래서 claim 경로에서는 surface 점군을 `check_surface` 로 검증해 읽고, TLS_GLOBAL 로 옮긴 뒤
record 가 기록한 parameters 로 **다시 자르고** 결과를 대조한다 — station 별 valid 여부, 면적,
그리고 각도 bin 별 반경까지. 재현 슬랙은 1e-9 상대오차 (부동소수 표현용이지 기하 허용치가
아니다).

* **surface 의 `depth_source` 도 다시 유도한다.** Phase 1A/1B 는 그 값을 fusion 시점에 한 번
  정하고 record 에 쓴다. `check_surface` 는 점군 digest 를 다시 확인하지만 이 필드는 확인한
  적이 없어서, `surface.json` 의 문자열 한 줄을 `external_unverified` → `minegs_render` 로
  고치면 승격되고 section record 가 그것을 그대로 물고 내려온다. 그래서 claim 경로에서는
  surface 가 기록한 depth 디렉터리와 run 디렉터리에 대해 Phase 1B promotion
  (`_depth_provenance` — manifest↔run, checkpoint, renderer, pin, view 집합, 파일별 digest)
  을 **다시 돌리고** manifest id 까지 대조한다. 두 디렉터리가 없으면 claim 은 거부다.
* diagnostic 경로에서는 하지 않는다. 전체 재추출 비용이 들고, diagnostic 수치가 생산자의
  선언이라는 것이 바로 "diagnostic" 의 뜻이다.
* surface 가 사라졌으면 claim 은 **거부**한다 (강등이 아니다). 검증할 수 없는 증거 위의 주장은
  주장이 아니다. `--diagnostic` 은 여전히 동작한다.

| 입력 | `--diagnostic` 없이 | `--diagnostic` |
|---|---|---|
| bare `SectionSeries` JSON (1C 이전) | `ContractError` — provenance 없음 | 경고 + `geometry_diagnostic` |
| `raw_cloud` section artifact | `ContractError` — 어떤 복원에 대한 증거도 아님 | 경고 + `geometry_diagnostic` |
| `external_unverified` surface 기반 | `ContractError` — depth 가 run 과 묶여 있지 않음 | 경고 + `geometry_diagnostic` |
| `minegs_render` 기반, holdout coverage 불완전 (관측 쌍 있음) | `ContractError` — 빠진 구간을 이름으로 보고 | 경고 + holdout 에 대한 partial volume, `geometry_diagnostic` |
| `minegs_render` 기반, holdout 에 연속 관측 station 2개 미만 | `ContractError` | `ContractError` — holdout 에 대한 partial volume 자체가 없다. `--no-holdout-only` 로 자른 구간의 diagnostic 을 낸다 |
| `minegs_render` surface 기반, coverage 완전 | `volume_accuracy` | `volume_accuracy` |
| 다른 dataset/축/변경된 surface | `ContractError` | `ContractError` (플래그로 면제되지 않음) |

`eval geometry` 와 같은 이유로 **`--no-holdout-only` 는 claim 을 내린다**: `volume_accuracy` 는
정의상 holdout 에 대한 주장이고, 전 구간 적분은 run 이 학습에 쓴 형상을 다시 재는 것이다.

### Gap-safe 적분

```
chainage  0   1   2   3   4
area     10  10   -  10  10      →  V = 10 + 10 = 20 m³   (40 이 아니다)
                                    integrated [0,1] ∪ [3,4], missing [1,3]
```

* 연속된 관측 station 의 run 안에서만 적분한다. gap 은 적분 경계이고, 그 구간의 체적은
  추정하지도 보간하지도 않는다. 따라서 coverage 가 불완전하면 수치는 항상 **과소** 추정이다.
* 서로 다른 holdout 구간 사이도 같은 이유로 절대 잇지 않는다. 구간이 겹치거나 맞닿으면 먼저
  union 으로 합친 뒤 각각 독립 적분한다.
* 관측 station 이 하나뿐인 run 은 길이가 0 이므로 적분에 기여하지 않는다. 그 station 은 여전히
  "관측됨" 으로 세고, 주변 구간은 여전히 missing 이다. 둘 다 참이다.
* coverage 는 **요청한 구간** 기준이다. 시계열이 holdout 중간에서 끝나면 coverage 는 절반이지
  "짧은 갱도의 완전 coverage" 가 아니다.
* **station 이 아예 빠진 것도 gap 이다.** NaN 규칙이 보는 것은 "있는데 invalid" 뿐이라, 행을
  지우면 이웃이 붙어 사다리꼴이 구멍을 가로지르고 리포트는 아무것도 빠지지 않았다고 말한다.
  시계열이 스스로 선언한 station 간격보다 넓은 step 은 gap 으로 본다. `diff_sections` 의
  "한쪽 epoch 에만 있는 station" 도 같은 규칙으로 처리된다.
* `integrate_sections` · `compare_to_design` · `diff_sections` 가 모두 같은 helper
  (`eval/volume/coverage.py`)를 쓴다. 셋이 "어느 구간이 관측되었는가" 에 대해 갈라지지 않는다.
* `VolumeReport` 에 `coverage{requested_intervals_m, integrated_intervals_m, missing_intervals_m,
  requested_length_m, covered_length_m, coverage_fraction, sampled_length_m, sampled_fraction,
  valid/missing_section_count, missing_chainages_m}` 와 `segments[]`, `section_parameters`,
  `source` 가 실린다. `volume_m3` 는 segment 합과 정확히 같고, `mean_area_m2` 는 적분 구간
  길이 가중 평균이라 `mean_area_m2 × covered_length_m == volume_m3` 이다.
  `start/end_chainage_m` 은 적분 구간의 **외곽** 이지 적분된 길이가 아니다 (구간이 둘이면
  그 사이는 포함되지 않는다).
* **임의 임계값을 만들지 않는다.** 이 PR 은 80 %/90 % 같은 coverage threshold 를 도입하지
  않는다. claim 은 "요청한 holdout 을 전부 적분할 수 있어야 한다" 로 두고, 실측 검증으로 허용
  가능한 최소 coverage 가 정해지면 그때 별도 결정으로 완화한다.
* **슬랩이 station 간격보다 넓으면 claim 경로에서 거부한다.** `--thickness-m > --interval-m`
  이면 이웃 슬랩이 겹쳐, 자기 형상이 없는 station 이 이웃의 점으로 채워져 valid 가 된다
  (gappy 표면에서 1 m 간격 6 m 슬랩은 holdout 의 구멍을 전부 메운다). 데이터 품질 임계값이
  아니라, `A(s_i)` 가 `s_i` **에서의** 측정이기 위한 조건이다. diagnostic 에서는 허용한다.
* **coverage 는 station 격자에 대한 진술이고, 격자는 사용자가 고른다.** 6 m 떨어진 두 station
  은 그 사이 6 m 를 사다리꼴 규칙으로 완전히 적분한다 — 통상적인 관행이다 — 그래서 어느
  station 도 걸리지 않는 구멍은 `coverage_fraction` 에 보이지 않는다. 대신
  `sampled_length_m` / `sampled_fraction` 에 보인다: 적분된 구간 중 슬랩이 실제로 들여다본
  길이의 비율이다. 위 예시에서 1 m 격자는 거부되고 6 m 격자는 통과하지만, 통과한 쪽은
  `sampled 0.50 m of that (8.3%)` 로 보고된다. **게이트가 아니라 읽을 수 있는 수치다** —
  claim 에 필요한 최소 해상도는 최소 coverage 와 같은 성격의 미결 결정이고, 여기서 숫자를
  지어내지 않는다. `section_parameters` 가 volume.json 에 함께 실리는 이유이기도 하다.
* **격자가 부풀릴 수 없는 수치도 함께 낸다.** claim 경로는 surface 를 다시 읽으므로, 점군
  자체에서 `max_point_gap_m` — 주장 구간 안에서 복원된 점이 하나도 없는 가장 긴 구간 — 을
  계산해 리포트와 CLI 에 싣는다. `coverage_fraction` 은 "station 이 구간을 덮는가" 를 말하고,
  이것은 "그 아래에 무엇이라도 있는가" 를 말한다. 위 6 m 격자 예시는 `coverage 100%` 이면서
  `max_point_gap_m > 2 m` 로 보고된다. 역시 게이트가 아니라 수치다.
* **각도 방향 보간도 보고한다.** `extract_sections` 는 빈 angle bin 을 `angle_bins//10` 까지
  이웃에서 보간하고도 section 을 valid 로 둔다 — chainage 축에서 거부하는 바로 그 보간이다.
  section geometry 알고리즘 재설계는 이 PR 범위 밖이므로, 적분된 station 들에 대한
  `interpolated_bin_fraction` 을 coverage 블록에 실어 얼마나 되는지 말한다.

### 알려진 한계

* **claim 이 아닌 경로의 면적은 생산자의 선언이다.** claim 경로는 surface 에서 다시 잘라
  대조하지만, diagnostic 수치와 surface 가 사라진 record 는 그렇지 않다. 그리고 재현 대조도
  서명은 아니다: surface 점군 자체를 바꾸고 record 를 그에 맞춰 다시 만들면 전부 일관된다.
  `DepthManifest` 와 같은 경계 — 사고를 막는 경계이지 서명이 아니다.
* `minegs eval change` 는 dataset 을 인자로 받지 않으므로 대부분을 대조하지 않는다. 그래서
  결과는 영구히 `geometry_diagnostic` 이다 (pair protocol 은 Phase 7). 다만 dataset 없이도
  대조 가능한 하나 — 두 시계열의 `reference_axis` — 는 검사하고, 다르면 거부한다. 서로 다른
  polyline 위의 chainage 를 빼는 것은 변화가 아니라 좌표계 차이다.
* `--start-m`/`--end-m` 없이 자른 격자는 축 시작점부터 `interval_m` 간격이다. holdout 경계가
  그 격자에 떨어지지 않으면 coverage 는 완전해질 수 없고, 거부 메시지가 재단(re-section)을
  안내한다.

## E57 end-to-end workflow (Phase 2)

Phase 0B–1C 가 닫아 둔 계약을 E57 하나로 관통시키는 thin orchestration 이다. 계약은
[docs/PHASE2_CONTRACT.md](docs/PHASE2_CONTRACT.md), 실제 GPU 장비에서의 실행 절차는
[docs/PHASE2_E57_G2.md](docs/PHASE2_E57_G2.md).

```bash
# 전부
minegs e2e run workflow.yaml --work-dir work/wf

# 끊어서 (각 stage 가 몇 시간짜리일 수 있다)
minegs e2e run workflow.yaml --work-dir work/wf --through dataset
minegs e2e status --work-dir work/wf            # 어디까지 됐고, 지금 돌리면 무엇이 거부되는가
minegs e2e run workflow.yaml --work-dir work/wf --rebuild-from depth

# 이미 있는 artifact 에서 문서만 다시 만든다 (stage 를 하나도 실행하지 않는다)
minegs e2e report --work-dir work/wf --out report/ep1
```

stage 는 `ingest → dataset → train → depth → surface → geometry → sections_volume → report`
이고, 순서는 `minegs/e2e/models.py` 의 `STAGE_ORDER` 하나로만 선언된다.

**재사용 규칙.** stage 마다 자기가 실제로 의존한 identity 를 **세상에서 다시 읽어** digest 하고,
앞 stage 의 fingerprint 를 사슬로 엮어 함께 기록한다. 그래서 E57 교체·build config 변경·dataset
hash 변경·run 교체·surface 교체 중 하나만 있어도 그 뒤가 전부 stale 이 된다. stale 한 stage 는
조용히 재사용하지도, 조용히 다시 돌리지도 않는다 — 무엇이 움직였는지 말하고 멈춘다.
`--rebuild-from` 이 명시적 답이고, 무엇이 다시 만들어졌는지는 원장에 남는다.

이것은 **training resume 이 아니다.** 이미 성공한 TRAIN stage 를 다시 돌리지 않고 그 run
artifact 에서 DEPTH 부터 이어 간다는 뜻이다 (gsplat training 자체의 재개는 Phase 0D.3, 여전히
fail closed).

**paired TLS validation.** 복원과 held-out TLS 를 **같은 section grid** 에서 station 단위로
pair 하고, 체적은 **공통 적분 구간** 에서만 비교한다. 서로 다른 coverage 에서 잰 100 m³ 와
102 m³ 를 빼는 것은 오차가 아니라 서로 다른 갱도 구간에 대한 두 숫자이고, 복원이 덜 덮을수록
작아진다. 결측을 가로지르는 사다리꼴은 없다 (Phase 1C helper 재사용, 새 적분 구현 없음).

**report.** `phase2_report.json` 이 machine-readable source of truth, `phase2_report.md` 가 그
표현이다. report 는 집계만 한다 — 다시 계산하지 않고, 못 채운 값은 **null + 이유**이며,
maturity status 를 올리지 않는다.

**CLI 에 trainer/renderer 를 대체하는 flag 는 없다.** 그 seam 은 Python 에서 structural gate 가
잡고, 잡았다는 사실을 stage 와 report 양쪽에 기록한다 (`real_gpu_execution`,
`real_renderer_execution`).

### 지금 검증된 것과 아닌 것

`tests/test_e2e_gate.py` 는 **테스트 시점에 쓴 실제 E57 파일**에서 inventory·추출·camera
convention 측정·dataset·golden gate·run 검증·depth manifest·surface promotion·geometry·section
record·paired validation·report 까지 실제 production 함수를 관통한다. 대체되는 것은 선언된 두
hardware seam — **trainer 와 renderer** — 뿐이다.

그래서 이것은 **control path 와 evidence path 의 검증이지 G2 가 아니다.** 합성 renderer 결과를
G2 라고 부르지 않는다.

> Phase 2 E57 end-to-end workflow is implemented and structurally tested.
> Real-data scientific validation remains NOT VALIDATED.
> Phase 2 G2 remains PENDING.

## Image / 360 독립 재구성 (Phase 3)

TLS 초기 형상 없이 영상만으로 만든 재구성을, 측정된 Sim(3) 하나로 측량 좌표에 넣고, **같은**
학습·depth·surface·단면·체적 경로로 평가한다. 계약은 [docs/PHASE3_CONTRACT.md](docs/PHASE3_CONTRACT.md).

```bash
# 1. 프레임 집합 — SfM 이 무엇을 먹는지가 디렉토리 레이아웃이 된다
minegs ingest video frameset drift.mp4 work/fs --kind video360 --fps 2 \
    --n-yaw 8 --fov-deg 90 --size 1600 \
    --pano-source Configured --pano-az-sign 1 --nadir-el-deg -35

# 2. 재구성 — 그 프레임 집합에서만
minegs ingest video sfm work/fs work/sfm --backend colmap --mapper global

# 3. 정합 — 측정이다. 대응이 없으면 거부한다
minegs eval register work/sfm work/reg --basis known_target \
    --targets targets_sfm.csv --targets-tls targets_tls.csv \
    --reference-ply raw/tls_full.ply \
    --max-rmse-m 0.05 --min-inlier-ratio 0.6 --min-correspondences 6

# 4. dataset — init 은 이 재구성 자신의 점
minegs dataset from-sfm work/fs work/sfm work/reg data/driftA/dataset \
    --dataset-id driftA_v360 --source video360 --holdout-m 30:38 --centerline design.csv

# 5. image-only Golden Gate — station scan 이 없으므로 다른 증거로 본다
minegs dataset golden-gate data/driftA/dataset --out gg --reference-ply raw/tls_full.ply

# 6. 두 경로 비교 — 같은 grid, 같은 holdout, 둘 다 관측한 구간에서만
minegs eval compare-paths data/tls/dataset data/driftA/dataset \
    --tls-sections a.json --tls-reference-sections b.json \
    --image-sections c.json --image-reference-sections d.json --out cmp
```

읽는 법:

* **`SFM_INTERNAL` 은 `SOURCE` 가 아니다.** 재구성 좌표는 scale 까지 임의이고, `TLS_GLOBAL` 로
  나가는 유일한 출구는 **측정된** Sim(3) 다. `SOURCE` 문은 선언이고 SE(3) 만 돌려주므로 임의 scale
  재구성을 통과시키면 "이 좌표가 곧 측량 기준" 을 측정 없이 선언하게 된다 — 그래서 거부된다.
* **정합 support 는 ① 초기 대응 ∪ ② ICP target 이다.** `--basis known_target` 은 *scale* 의 출처를
  말하지 pose 의 출처를 말하지 않는다. 타깃으로 scale 을 잡고 전체 TLS 로 pose 를 refine 하면 결과는
  **diagnostic** 이다 — 전체와 겹치지 않는 holdout 은 없기 때문이다. support 범위를 기록하지 않으면
  holdout 과의 겹침을 판정할 수 없고, 판정할 수 없으면 거부다.
* **threshold 가 없으면 claim 도 없다.** 숫자 없는 게이트는 전부 통과시킨다. 실측 전에 발명한 숫자는
  아무 데이터도 근거하지 않은 threshold 라 더 나쁘다.
* **claim 은 읽는 것이 아니라 다시 도출된다.** `claim_allowed` 는 결론이므로 `registration.json`
  을 다시 열 때마다 품질 게이트와 함께 **재계산**해 기록된 값과 대조한다. manifest 가 들고 있는
  사본도 record 와 전부 대조한다 — protocol judge 가 읽는 것은 manifest 쪽이다.
* **init 이 재구성에서 왔다는 것은 원본과 대조해 증명된다.** dataset 안의 두 cloud 를 서로
  비교하는 것은 같은 주장의 사본 두 개일 뿐이다. 선택된 SfM model 이
  `provenance/phase3/sfm_model/` 에 번들되고, 거기서 측정된 Sim(3) → local origin → holdout
  제외를 다시 밟아 `init_points.ply`·`sparse/0`·카메라 중심을 대조한다.
* **holdout 은 선언이 아니라 사실이다.** image-only 경로에서 init cloud 는 재구성 그 자체이므로,
  `init_points.ply` 와 `sparse/0` **양쪽**에서 holdout chainage 의 점을 뺀다.
  `--holdout-images-excluded`(외삽 시험) 를 켜면 holdout 과 겹치는 capture group 이 빌드 시점에
  `train_groups` 에서 빠지고, chainage 를 잴 수 없는 그룹이 있으면 거부한다. 기본값은
  **꺼짐**(복원 시험) 이다 — 플래그를 안 줬다는 이유로 더 강한 실험이라고 기록되면 안 된다.
* **`golden-gate` 는 dataset 의 source 를 보고 갈라진다.** image-only 판은 `gate_kind:
  image_sfm_registered` 를 적고, `real_data_validation_status` 는 언제나
  `pending_human_inspection` 이다 — 사람이 볼 때까지.
* **`compare-paths` 는 먼저 거부하고 나중에 비교한다.** 같은 grid 에서 잘리지 않았거나 서로 다른
  holdout 으로 평가된 두 경로는 비교하지 않는다. 체적은 둘이 **함께 관측한** 구간에서만 적분되고,
  빠진 구간은 보고된다.
* **`real_execution` 은 경로별로 판정된다.** TLS 쪽과 image 쪽의 필수 stage 목록이 다르므로
  따로 본다. flag 를 합치면 한쪽의 `true` 가 다른 쪽의 `false` 를 덮어쓰고, 목록에 없는 stage 는
  아무도 보고하지 않은 채 통과한다. 없는 key 는 "모른다" 이고, 모르면 거짓이다.
* **CLI 에 SfM·프레임 추출을 대체하는 flag 는 없다.** trainer/renderer 와 같은 규칙이다.

### 지금 검증된 것과 아닌 것

`tests/test_phase3_gate.py` 가 합성 갱도 하나를 스캐너와 파노라마 양쪽으로 재구성하고, 실제 선별·
crop·mask·프레임 집합 검사·실제 정합·실제 image-only 빌더·실제 validator·실제 protocol judge 를
관통한 뒤 두 경로를 비교한다. 대체되는 것은 네 hardware seam — **SfM·프레임 추출·trainer·
renderer** — 이고 전부 기록된다. T1–T30 이 거부를 고정한다.

**REAL COLMAP EXECUTION: NOT PERFORMED.** 이 저장소에서 COLMAP 은 한 번도 실행되지 않았다.

> Phase 3 image/360 independent reconstruction path is implemented and structurally tested.
> Real-data scientific validation remains NOT VALIDATED.
> Phase 3 G2 remains PENDING.


## 평가 주장 게이트 (§5)

`minegs eval protocol <dataset>` 이 manifest 의 `split`/`initialization` 만 보고 주장 가능 범위를 판정한다.

- `reconstruction` 데이터셋에 형상 정확도를 요청하면 exit 3 으로 거부한다 (`--diagnostic` 은 주장 없는 수치만).
- **`change_volume` 은 단일 manifest 에서 절대 나오지 않는다.** 아키텍처상 change 는 *두 epoch* 의 같은 구간
  비교이므로, `capture_epoch` 가 있어도 `judge()` 는 `Protocol.CHANGE` 도 `Claim.CHANGE_VOLUME` 도 부여하지 않는다.
  `minegs eval change` 는 두 단면 시계열의 산술 차이이며 `geometry_diagnostic` 으로 표시된다. epoch 호환성
  (프레임·scale basis·reference axis·누수 구간)을 검증하는 pair protocol `judge_change(a, b)` 는 Phase 7 이다.

## 단계 · 상태 (§13)

**source of truth: [docs/ROADMAP.md](docs/ROADMAP.md)** — Phase 범위, validation gate(G0–G3), DoD.
아래는 요약이며 *implemented*(코드 있음)와 *validated*(실데이터 Gate 통과)를 구분한다.

| Phase | 상태 |
|---|---|
| 0A Foundation & Contract Freeze | **implemented + G1 통과** |
| 0B Real E57 ingest | **0B.1–0B.3 implemented** (inventory·증거 기반 매핑·추출), **not validated** — 실제 E57 필요 |
| 0C Metric dataset golden gate | **implementation complete, G1 structurally tested** — 합성 staging → `from-e57` → 재투영 Golden Gate 가 CI 에서 돈다. **G2: DEFERRED / NOT VALIDATED** (실제 E57 미실행) |
| 0D Local GS baseline | **0D.1 resume safety contract** + **0D.2 local GPU baseline execution contract: implemented + structurally tested** — gsplat v1.5.3 training resume 은 unsupported 이고 fail closed; 성공한 run 은 checkpoint·PLY·step 진행·frame invariant 를 모두 통과한 것만 기록된다. **실제 GPU baseline 미실행** → 0D 전체 **NOT COMPLETE** (ROADMAP §Phase 0D) |
| 1 Metric surface & evaluation | **1A metric surface artifact + depth fusion**, **1B metric depth rendering**, **1C section/volume evidence boundary: implemented + structurally tested** — 학습된 run → 렌더 depth + manifest → 검증된 surface artifact → section artifact → gap-safe 체적 → claim. 검증된 manifest 가 있을 때만 `minegs_render` 이고, 외부 depth·원시 PLY·bare series 는 diagnostic 전용이다. 결측 구간을 가로지르는 적분은 없다. **실제 GPU rendering 미실행** (CI 에 CUDA·gsplat 없음), **TSDF/mesh: NOT IMPLEMENTED**, 실측 데이터 과학적 검증: **NOT VALIDATED** |
| 2 E57 end-to-end MVP (v0.1) | **implemented + structurally tested** — `minegs e2e run` / `status` / `report` 가 E57 한 개를 ingest→dataset→train→depth→surface→geometry→단면/체적→report 로 관통한다. stage 마다 입력 identity 를 다시 읽어 대조하므로 움직인 입력 위에 조용히 쌓지 않는다. 복원과 held-out TLS 를 같은 grid 에서 pair 하고 공통 구간에서만 체적을 비교한다. structural gate 는 테스트 시점에 쓴 실제 E57 에서 돌지만 **trainer·renderer 는 대체**되고 그 사실이 report 에 남는다. **real E57 G2: NOT RUN**, 실측 과학적 검증: **NOT VALIDATED** ([runbook](docs/PHASE2_E57_G2.md)) |
| 3 Image/360 독립 재구성 | **implemented + structurally tested** — 영상/360 → 프레임 집합 → SfM(`SFM_INTERNAL`, 임의 scale) → 측정된 Sim(3) 정합 → image-only dataset → 기존 학습·depth·surface·단면·체적 → TLS-assisted 대비 공통 구간 비교. 초기화는 재구성 자신의 점이고 holdout 구간은 실제로 빠진다. **실제 COLMAP·GPU·렌더러·실측 영상 미실행** — 네 seam 모두 대체되고 그 사실이 artifact 와 report 에 남는다. **Phase 3 G2: PENDING**, 실측 과학적 검증: **NOT VALIDATED** |
| 4 Advanced GS / heavy | 미착수 — `depth_loss`·`normalize_world_space` 를 여기서 설계 |
| 5 장거리 갱도 · 청킹 | 예약만 (manifest.chunks) |
| 6 RunPod | **미구현, fail-closed** |
| 7 Multi-epoch change | **미구현** — single manifest 는 change claim 불가 |
| 8 Viewer / Export / Web | 부분 — Viser·export 구현, FastAPI 미착수 |

이 저장소는 아직 실제 갱도 데이터에서 검증된 결과를 주장하지 않는다. 합성 데이터셋 계약과
contract 강제만 CI 로 보장된다.
