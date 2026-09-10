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
pip install -e ".[e57]"          # pye57, PDAL            (§6.1)
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

minegs eval sections data/synthetic_tunnel/raw/tls_full.ply data/synthetic_tunnel/dataset \
       --interval-m 1 --thickness-m 0.2 --out sections.json
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
    e57/     inventory · scan_split · tiles(PDAL) · pose_to_colmap · pano/{E57Embedded,ExternalJpeg,VendorExport}
    video/   frames(ffmpeg) · dedup_blur · masks · rig(360 → COLMAP rig) · sfm/{COLMAPIncremental,COLMAPGlobal,GLUEMAP(exp)}
  train/
    backends/  base(BackendCapabilities) · gsplat(정규화 역변환 포함)
    runner/    base · local(docker, CUDA 없으면 RunPod 제안) · runpod · sync(rclone)
    profiles/  light.yaml · heavy.yaml
  eval/      protocol · register(Sim3 → ICP → diagnostics) · surface · geometry(양방향) · sections(A(s)) · volume(∫A ds, 설계대비) · change · render(PSNR/SSIM/LPIPS)
  viz/       viewer(Viser) · overlay(규약 캘리브레이션 = 골든 게이트) · compare · export(.spz/.splat)
  cli/       ingest / dataset / train / eval / viz / sync
docker/      Dockerfile.gpu · Dockerfile.cpu · entrypoint.sh
configs/     dataset/{e57,video,video360}.yaml · eval/geometry_holdout.yaml · runner/{local,runpod}.yaml
docs/        ARCHITECTURE.md
data/        (git 제외) <dataset_id>/{raw,dataset,runs,eval,export}
```

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
minegs train run data/<id>/dataset --profile light --runner local --config configs/runner/local.yaml --wait
# heavy 프로파일을 로컬 GPU 에서 (RunPod 러너는 Phase 1, 미구현)
minegs train run data/<id>/dataset --profile heavy --runner local --config configs/runner/local.yaml
minegs train status data/<id>/runs/<run_id>      # run.json: backend, digest, dataset_hash, T_local_from_internal, provenance
```

프로파일은 backend 플래그가 아니라 **capability** 를 요청한다 (`requests: {depth_loss: true, ...}`).

**실행 계약 (gsplat v1.5.3 upstream 기준)**
- trainer 는 PyPI wheel 에 없고 저장소의 `examples/simple_trainer.py` 다. `docker/Dockerfile.gpu` 가 wheel 과 같은
  태그를 `/opt/gsplat` 에 체크아웃하고 `examples/requirements.txt` 를 설치한 뒤 `MINEGS_GSPLAT_TRAINER` 로 경로를 준다.
  스크립트를 못 찾으면 어댑터가 커맨드 생성을 거부한다(`minegs train command` 는 dry-run 이라 예외).
- 러너는 `dataset/` 을 read-only 로 마운트하고 `runs/<run_id>/staged/` 에 **스테이징 복사본**을 만든다
  (`minegs/train/staging.py`): 프로파일의 `max_images` 를 train 이미지에서 균등 추출하고, `init_points.ply` 를
  `sparse/0/points3D.txt` 로 써서 TLS 점으로 초기화한다. gsplat 파서가 `images_<factor>_png` 를 `data_dir` 안에 쓰기
  때문에 read-only 데이터셋을 직접 넘길 수 없다. 스테이징 내용(이미지 수·서브셋 여부·init 출처·sha256)은 `run.json` 에 기록된다.
- 어댑터는 기본 `--no-normalize_world_space` 로 BACKEND_INTERNAL = LOCAL_METRIC 을 유지하고, 정규화를 켜면 gsplat 의
  정규화(Sim3)를 재계산해 출력 `.ply` 를 LOCAL_METRIC 으로 되돌린다. `absgrad` 는 `--strategy.absgrad`(default 전략 전용).
- `run.json` 의 `dataset_hash` 는 manifest·sparse·init_points·**images·masks**·centerline 을 모두 덮는다.
- **RunPod 러너는 Phase 1 이며 아직 실행되지 않는다.** `--runner runpod` 은 외부 호출 없이 `NotYetImplementedError`(exit 4)
  로 끝난다. 필요한 단계는 `minegs/train/runner/runpod.py` docstring 에 적혀 있다.

## 단계 (§13)

| Phase | 상태 |
|---|---|
| 0A Foundation — 패키지·config·manifest v1+migration·CLI·tests/CI | **완료** (합성 데이터셋 통과) |
| 0B E57 ingest — inventory·scan split·PDAL·pose·PanoSource | 인터페이스 + 구현, 실제 E57 검증 필요 |
| 0C Dataset — 링 크롭·COLMAP export·init PLY·프레임 | 구현, **골든 게이트**(재투영 오버레이) 실데이터 확인 필요 |
| 0D GS baseline — gsplat 어댑터·LocalRunner·light | 어댑터·스테이징·러너 구현(upstream 계약 확인), GPU 실행 검증 필요 |
| 1 RunPod | 미구현 — `submit` 이 명시적으로 거부, 계획은 docstring |
| 2 Image SfM | 커맨드 빌더·rig·Sim3 정합 구현, GLUEMAP 보류 |
| 3 Metric geometry | 단면·체적·change·양방향 지표 구현, TSDF/PGSR 미구현 |
| 4 Web (FastAPI) | 미착수 |
