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
minegs eval volume   sections.json data/synthetic_tunnel/dataset --design-radius-m 2.4
minegs eval geometry <pred.ply> data/synthetic_tunnel/dataset --tls-ply data/synthetic_tunnel/raw/tls_full.ply
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
  eval/      protocol · register(Sim3 → ICP → diagnostics) · surface · geometry(양방향) · sections(A(s)) · volume(∫A ds, 설계대비) · change · render(PSNR/SSIM/LPIPS)
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
| 1 Metric surface & evaluation | 부분 — 양방향 지표·단면·체적 구현, surface 추출(depth/TSDF) 미구현 |
| 2 E57 end-to-end MVP (v0.1) | 미착수 |
| 3 Image/360 독립 재구성 | 부분 — 커맨드 빌더·rig·Sim3 정합 구현, 미검증 |
| 4 Advanced GS / heavy | 미착수 — `depth_loss`·`normalize_world_space` 를 여기서 설계 |
| 5 장거리 갱도 · 청킹 | 예약만 (manifest.chunks) |
| 6 RunPod | **미구현, fail-closed** |
| 7 Multi-epoch change | **미구현** — single manifest 는 change claim 불가 |
| 8 Viewer / Export / Web | 부분 — Viser·export 구현, FastAPI 미착수 |

이 저장소는 아직 실제 갱도 데이터에서 검증된 결과를 주장하지 않는다. 합성 데이터셋 계약과
contract 강제만 CI 로 보장된다.
