# minegs 아키텍처 v2

지하 갱도 metric Gaussian Splatting 연구 프레임워크. TLS(E57) 와 영상(일반·360)
두 입력 경로, 로컬/RunPod 이중 실행, 정합·형상·체적 평가, 웹 기반 시각화.

## 1. 원칙

1. **데이터셋 계약 하나.** 입력 경로가 무엇이든 `dataset/` 은 같은 모양.
   학습·평가·시각화는 이 계약만 읽는다.
2. **코드 한 벌, 목적별 런타임.** GPU 학습 이미지는 로컬과 RunPod 에서
   digest 까지 동일하게 쓴다. CPU 인제스트는 네이티브 파이썬 또는 CPU 컨테이너를
   허용한다. 거대 단일 컨테이너는 만들지 않는다.
3. **CLI 가 진실의 원천.** UI 는 CLI 가 노출하는 파이썬 API 의 얇은 껍데기.
4. **원본은 로컬에만.** E57·원본 영상은 파드에 올리지 않는다. `dataset/` 만 간다.
5. **좌표 프레임 3계층.** 평가·공간 산출물은 TLS_GLOBAL(m) 로 환원 가능해야 한다.
   학습은 수치안정성을 위해 LOCAL_METRIC(m) 을 쓴다. 프레임 간 변환은 manifest 에
   반드시 명시한다 (§3).
6. **평가 누수는 계약 수준에서 차단.** 어떤 TLS 데이터가 초기화·학습에 들어갔는지
   manifest 가 선언하고, 평가 모듈은 그 선언에 맞지 않는 주장을 거부한다 (§5).
7. **가우시안 중심은 표면이 아니다.** 형상 평가는 항상
   GS → 깊이/메시/표면 표현 → TLS 비교 순서를 따른다. 이 경계는 계약으로 강제된다:
   surface 는 `SurfaceRecord` 를 남기는 명시적 artifact 이고 (`eval/surface/models.py`),
   claim 을 담는 `eval geometry` 는 그 artifact 없이는 실행되지 않는다. 그리고 record 를
   믿는 것이 아니라 점들을 다시 읽어 digest 로 대조하며, artifact 라는 것만으로는 부족하다 —
   `depth_source` 가 이 프로젝트가 렌더한 depth 를 가리킬 때만 accuracy claim 이 열린다
   (Phase 1A 의 외부 depth 는 diagnostic 전용).
8. **모든 산출물에 계보.** raw → dataset → run → eval → export 각 단계가
   입력 해시·설정 해시·git SHA·도구 버전·부모 ID 를 기록한다 (§9).

## 2. 디렉토리

```
minegs/
  minegs/
    core/        config 스키마(pydantic, schema_version + migration), manifest,
                 frames(SE3/Sim3), centerline, chunking, provenance
    dataset/     Phase 0C materialization: build_config(strict), staging_input(0B 계약 reader),
                 frames(SOURCE→TLS→LOCAL), cameras(E57 pinhole→COLMAP, 축 규약), calibrate,
                 materialize(groups·split·holdout·leak-free init·transactional publish),
                 reprojection, golden_gate(report·overlay·Viser sample)
    ingest/
      common/    geometry, equirect, colmap_io               ← 두 경로가 공유
      e57/       _nodes(단일 pye57 seam), inventory+models+exceptions (SOURCE 프레임 계약),
                 images+mapping(증거 기반 station↔image), extract(추출·마스크·manifest),
                 scan_split(deprecated), tiles(PDAL), pose_to_colmap,
                 pano/  PanoSource 어댑터: E57Embedded · ExternalJpeg · VendorExport
      video/     frames(ffmpeg), dedup_blur, masks,
                 sfm/   SfMBackend: COLMAPIncremental · COLMAPGlobal · (exp) GLUEMAP
                 rig.py 360 크롭 → COLMAP rig 정의
    train/     staging (쓰기 가능 복사본 · max_images 서브셋 · init_points→points3D)
      backends/  base.py(BackendCapabilities), gsplat.py
                 (later) splatfacto.py, pgsr.py — INRIA 는 외부 호출만, 저장소 미포함
      runner/    base.py, local.py, runpod.py, sync.py(rclone)
      profiles/  light.yaml, heavy.yaml
    eval/        register/ (initial_alignment, sim3, rigid_icp, diagnostics)
                 surface/  (models=surface·depth manifest 계약 · render=gsplat metric depth · depth 역투영 · TSDF·mesh 추출)
                 geometry/ (accuracy, completeness, chamfer, 분위수)
                 sections/ (중심선 기준 단면 A(s))
                 volume/   (∫A(s)ds, 메시 체적, 설계 대비 여굴·미굴)
                 change/   (epoch 간 차분)
                 render/   (홀드아웃 그룹 PSNR·SSIM·LPIPS)
    viz/         viewer(Viser), overlay(규약 검증), compare(히트맵·단면), export(.spz)
    cli/         typer: ingest / dataset / train / eval / viz / sync
  docker/        Dockerfile.gpu (CUDA, PyTorch, gsplat, COLMAP ≥4.0)
                 Dockerfile.cpu (PDAL, pye57, ffmpeg, COLMAP CPU) — 선택
  configs/
  tests/
  data/                              (git 제외)
    <dataset_id>/
      raw/           E57, mp4, 설계 중심선
      dataset/       ← 계약 (§3)
      runs/<run_id>/     ckpt, point_cloud/*.ply (LOCAL_METRIC), log, run.json
      eval/<eval_id>/    mesh, geometry.json, sections.json (SectionRecord), volume.json, eval.json
      export/<id>/       .spz (.splat 은 legacy, 선택)
```

## 3. 좌표 프레임

```
SCANNER        개별 스캔 자신의 좌표             ← 0B.3 --raw 추출 (unregistered)
   │  T_source_from_scan (E57 pose)
   ▼
SOURCE        원본 파일 자신의 좌표 (스캐너 로컬, 이미 m)   ← 0B 추출 산출물
   │  Phase 0C 의 선언 (SE(3), scale 없음)
   │
   │            SFM_INTERNAL   독립 SfM 자신의 좌표 — 임의 scale, 임의 원점
   │                  │  측정된 Sim(3) (Phase 3 §7, registration artifact 에 근거와 품질)
   ▼                  ▼
TLS_GLOBAL     실제 계측 좌표, m        ← 평가·보고
   │  SE(3), 병진·회전만
   ▼
LOCAL_METRIC   갱도 또는 청크 중심 원점, m ← 학습 입력·출력
   │  백엔드 내부 정규화 (gsplat scene_scale 등)
   ▼
BACKEND_INTERNAL                        ← 어댑터가 반드시 역변환해서 .ply 를
                                           LOCAL_METRIC 으로 내보낸다
```

* LOCAL_METRIC 도 1 unit = 1 m. 스케일은 건드리지 않는다.
* UTM 급 좌표(10⁶ m)를 float32 에 넣으면 유효 정밀도가 수십 cm 로 떨어진다.
  체적 계측에서 이 하나로 결과가 무의미해진다.
* `T_tls_from_local` 은 manifest 필수 항목. run 이 청크 단위면 청크마다 하나.
* **`SFM_INTERNAL` 은 `SOURCE` 가 아니다 (Phase 3 AD-1).** 독립 image/360 SfM 의 출력은 scale 까지
  임의다. `SOURCE` → `TLS_GLOBAL` 의 문(`SourceFrameConfig`)은 `explicit_identity` /
  `explicit_transform` 두 모드의 **선언**이고 `se3()` 를 돌려주므로 구조적으로 scale 을 담을 수 없다.
  임의 scale 재구성을 `SOURCE` 라 부르면 `explicit_identity` 한 줄로 "이 좌표가 곧 측량 기준" 이라고
  **측정 없이** 선언할 수 있게 된다. 그래서 `SFM_INTERNAL` 을 별도로 두고, 그 프레임에서 나가는
  유일한 출구를 **측정된 Sim(3)** 로 고정한다. 선언은 이 경로에서 거부된다. 계약 전문은
  [docs/PHASE3_CONTRACT.md](PHASE3_CONTRACT.md) §3 AD-1, 정합 규칙은 같은 문서 §7.
  구현됨: `Frame.SFM_INTERNAL`, `require_metric_frame()`(`minegs/core/frames.py`), 그리고
  `resolve_tls_from_source()` 가 이 프레임을 거부한다(`minegs/dataset/frames.py`).
* `SOURCE` 와 `SCANNER` 는 Phase 0B ingest 전용이며 dataset 계약에 등장하지 않는다. E57 의
  좌표가 `TLS_GLOBAL` 인지는 파일이 말해 주지 않으므로, 그 선언은 dataset materialization
  (Phase 0C, `minegs/dataset/`) 이 명시적으로 한다 — `source_frame.mode: explicit_identity`
  또는 `explicit_transform`. 선언 없는 identity 는 없다. Phase 0B 산출물에는
  `TLS_GLOBAL`/`LOCAL_METRIC` 이라는 문자열이 주석으로도 등장하지 않는다 — 산출물을 grep 했을 때
  나오면 그것은 진짜 주장이어야 한다.
* **`TLS_GLOBAL` 의 GLOBAL 은 측지/global CRS 를 뜻하지 않는다.** 여러 scan 과 camera 를 하나의
  metric survey 좌표계로 표현하는 *평가 기준 프레임* 이라는 뜻이다. Matterport E57 의 registered
  survey frame 을 명시적으로 TLS_GLOBAL 로 채택할 수 있지만, 그것이 UTM/EPSG 좌표라는 주장은 아니다.
* LOCAL_METRIC ← TLS_GLOBAL 은 baseline 에서 translation-only 다 (`R = I`, scale 1). 원점은
  deterministic policy(station centroid, 0.1 m round) 또는 명시값이며 build config 에 기록된다.

## 4. 데이터셋 계약

```
dataset/
  images/              pinhole 이미지
  sparse/0/            cameras.txt · images.txt · points3D.txt · (rigs.txt)
  init_points.ply      초기 가우시안 위치, LOCAL_METRIC
  masks/               선택 — 삼각대·작업자·나다르
  manifest.json
```

### manifest.json — v1 필수 6 + 선택

```jsonc
{
  "schema_version": "1.0",                          // 필수
  "dataset_id": "kigam_tunnelA_ep1_v003",
  "coordinate_frames": {                            // 필수
    "evaluation": "TLS_GLOBAL", "training": "LOCAL_METRIC", "unit": "m",
    "T_tls_from_local": [[1,0,0,318300.0],[0,1,0,4012300.0],[0,0,1,120.0],[0,0,0,1]]
  },
  "capture_groups": {                               // 필수 — 광학중심/궤적 단위
    "S07":  {"type": "tls_station",        "members": ["S07_f00.jpg", ...],
             "chainage_m": 63.2},
    "V013": {"type": "trajectory_segment", "members": ["v_000412.jpg", ...],
             "chainage_range_m": [58.0, 71.5]}
  },
  "split": {                                        // 필수
    "train_groups": ["S01","S02","S03","S05","S06","S07"],
    "test_groups":  ["S04","S08"],
    "geometry_holdout": {                           // 스테이션이 아니라 구간으로
      "chainage_ranges_m": [[38.0, 46.0], [94.0, 102.0]],
      "points_excluded": true,                      // init_points 에서 제외
      "images_excluded": false                      // 사진은 남김 = 복원 시험
    }                                               // true 면 외삽 시험 (다른 질문)
  },
  "initialization": {                               // 필수
    "source": "tls | sfm_sparse | random",
    "file": "init_points.ply",
    "groups": ["S01","S02","S03","S05","S06","S07"],
    "excluded_chainage_ranges_m": [[38.0, 46.0], [94.0, 102.0]]
  },
  "provenance": {                                   // 필수
    "minegs_version": "0.1.0", "git_commit": "...", "config_hash": "...",
    "source_assets": [{"path": "raw/tunnelA.e57", "sha256": "..."}]
  },

  "source": "tls | video | video360",               // 이하 선택
  "capture_epoch": {"id": "ep1", "date": "2026-10-14"},
  "scale": {"basis": "tls_pose | sim3_to_tls | known_target", "factor": 1.0},
  "registration": {"method": "sim3+icp", "scale": 1.0032, "rmse_m": 0.018,
                   "inlier_ratio": 0.91, "transform": [...]},
  "pano_convention": {"az_sign": 1, "el_flip": false, "az_offset": 0.0,
                      "source": "E57Embedded", "vendor": "Leica"},
  "centerline": {"file": "raw/centerline.csv", "source": "design | extracted"},
  "chunks": {"basis": "centerline_chainage", "length_m": 80, "overlap_m": 15,
             "list": [{"id": "C01", "range_m": [0, 80]}, ...]}
}
```

* `capture_groups` 가 `stations` 를 대체한다. 타입은 `tls_station`,
  `trajectory_segment`, 향후 `camera_rig`, `mobile_mapping_segment`.
* 렌더 평가(PSNR 등)는 그룹 단위 분할. **형상 홀드아웃은 chainage 구간 단위.**
  TLS 스캔은 인접 스테이션과 겹치므로 스테이션 단위로는 형상 누수를 못 막는다.
* schema_version 은 첫날부터. `core/manifest.py` 에 migration 함수를 둔다.
* `scale.basis` 없이 평가 모듈은 실행을 거부한다.

## 5. 평가 프로토콜

| 프로토콜 | 초기화 | 학습 이미지 | 평가 | 주장 가능 범위 |
|---|---|---|---|---|
| `reconstruction` | TLS 전부 | 전부 | 없음 | 실무용 최고품질. 성능 주장 불가 |
| `novel_view` | train 그룹 | train 그룹 | test 그룹 이미지 | PSNR/SSIM/LPIPS |
| `geometry_holdout` | 홀드아웃 구간 제외 | 설정에 따름 | 홀드아웃 구간 TLS | 형상·체적 정확도 |
| `change` | epoch 별 위 중 하나 | | 두 epoch 의 동일 구간 | 차분 체적 |

평가 모듈은 manifest 의 `split`·`initialization` 을 읽어 프로토콜을 판정하고,
`reconstruction` run 에 대해 형상 정확도 수치를 내는 요청을 거부한다.

단, **`change` 는 pair-level 프로토콜이다.** 단일 manifest 를 보는 `judge(manifest)` 는
`Protocol.CHANGE` 도 `Claim.CHANGE_VOLUME` 도 절대 생성하지 않는다 (`capture_epoch` 가 있어도
마찬가지). epoch 호환성(서로 다른 epoch id, 프레임·scale basis, 공통 reference axis, 겹치는
평가 구간, 누수 없는 초기화)을 검사하는 pair evaluator `judge_change(a, b)` 는 Phase 7
(ROADMAP.md). 그 전까지 두 단면 시계열의 차분은 `geometry_diagnostic` 이다.

## 6. 입력 경로

### 6.1 E57 (TLS)
inventory(pye57, 헤더만) → 증거 기반 image↔scan 매핑 → 추출(staging, SOURCE 프레임)
→ **Phase 0C** (`minegs/dataset/`): SOURCE→TLS_GLOBAL 선언 → LOCAL_METRIC 원점 → 카메라 →
COLMAP → leak-free `init_points.ply` → 재투영 **골든 게이트** → Viser.

두 camera path:

* **pinhole** (primary — Matterport Pro3 E57 은 scan 당 6 pinhole face). intrinsics 는 E57
  `pinholeRepresentation` 선언값만 (`fx = focalLength / pixelWidth`). E57 image frame ↔ COLMAP
  camera 축 규약 `R_e57cam_from_cam` 은 `calibrate-camera` 가 24 개 proper rotation 을 TLS RGB
  재투영으로 채점해 3 station 이상에서 일관되게 이길 때만 채택하거나, config 에 명시한다.
  `T_local_from_cam = T_local_from_tls @ T_tls_from_source @ T_source_from_e57cam @ R_e57cam_from_cam`.
* **spherical**: `PanoConvention` → ring crop → COLMAP (합성 K). cylindrical 은 거부.

* E57 에 파노라마가 반드시 있다고 가정하지 않는다. 요구하는 것은
  `station_id ↔ panorama_id` 매핑 계약뿐이다.
* PDAL `readers.e57` 은 내부 클라우드를 병합해 읽고 공통 dimension 만 취하며
  spherical 은 미지원 → 인벤토리·스캔 분리는 pye57, 대용량 타일링만 PDAL.

### 6.2 영상 · 360
ffmpeg → 블러·중복 제거 → (360: equirect → 링 크롭, K 는 합성이므로 정확히 알려짐,
**같은 프레임의 크롭은 COLMAP rig 로 등록**) → 마스킹 → SfM → 희소점 → `register` (§7) →
metric dataset (`init_points.ply` 는 등록된 SfM 희소점에서). 순서와 증거 요구는
[docs/PHASE3_CONTRACT.md](PHASE3_CONTRACT.md) §6–§8 이 정한다 — SfM 희소점은 등록 전까지
`SFM_INTERNAL` 이므로 그 상태로 dataset 에 들어가지 않는다.

```yaml
sfm:
  backend: colmap          # ≥ 4.0
  mapper: global | incremental   # 반복 패턴·저텍스처면 incremental 폴백 유지
  fix_intrinsics: true     # 360 크롭·보정된 카메라
  rig: auto                # 360 이면 자동 생성
  experimental: gluemap    # 저텍스처·저겹침 특화, 의존성 무거움 — Phase 2 이후
```

지하 조명 변동은 백엔드 capability 로 처리 (§8).

## 7. 정합 (register)

영상 SfM 은 similarity geometry. 좌표계가 전혀 다르니 ICP 부터 넣지 않는다.

```
알려진 타깃 / 스테이션 대응
      ↓  initial_alignment
   Sim(3)  (scale + R + t)
      ↓  robust refine
   SE(3) ICP
      ↓
   diagnostics → manifest.registration (scale, rmse_m, inlier_ratio, transform)
```

TLS 경로는 항등. 품질 지표 없는 정합 결과는 평가에 쓸 수 없다.

image-only 경로의 추가 규칙(정합 support 와 evaluation holdout 의 분리, ICP target 까지 support 에
포함, 측정된 Sim(3) 만 허용)은 [docs/PHASE3_CONTRACT.md](PHASE3_CONTRACT.md) §3 AD-2 와 §7 이
source of truth 다.

## 8. 학습 엔진

### 8.1 백엔드
v0.1 은 **gsplat 하나**. `BackendCapabilities(appearance_embedding, bilateral_grid,
depth_loss, ...)` 를 어댑터가 선언하고 프로파일은 capability 로 옵션을 요청한다 —
gsplat 버전이 바뀌어도 프로파일 계약이 깨지지 않는다.
Phase 0 은 pinned `simple_trainer` 래퍼, 장기적으로 gsplat 라이브러리 위의 얇은
`MineGSTrainer`. splatfacto 는 rasterizer 만 같고 실행 계약이 다르므로 별도 어댑터
(later). PGSR/2DGS 는 Phase 3. INRIA 원본은 non-commercial 라이선스 → 저장소·Docker
미포함, 베이스라인 비교 시 외부 호출.

어댑터 책임: `(dataset, profile) → 커맨드`, 그리고 **출력 .ply 를
LOCAL_METRIC 으로 역변환**해 `runs/<id>/` 규약으로 정규화.

### 8.2 러너
```
Runner.submit(run_config) -> RunHandle
RunHandle.status() / .logs() / .fetch_artifacts()
```
* `LocalRunner` — `docker run --gpus device=<n> minegs:gpu@sha256:...`. CUDA 없으면 거부하고
  RunPod 저가 GPU 라우팅 제안.
* `RunPodRunner` — 파드 생성(네트워크 볼륨) → `sync.push`(dataset 만) → 엔트리 →
  폴링 → `sync.pull`(ply·로그) → 종료.
* 두 러너의 GPU 이미지 digest 는 동일. run.json 에 기록.
* **Training resume 은 구현되어 있지 않다.** `--resume-from` 은 명시적 옵션으로 존재하지만
  `Runner.prepare` 에서 항상 거부되며, 이는 trainer 실행 이전이자 run directory 생성 이전이다.
  gsplat v1.5.3 은 학습을 이어붙일 수 없고 (`--ckpt` = evaluation only), 진짜 resume 은 완전한
  training state 를 복원하는 checkpoint contract 를 요구한다 — Phase 0D.3, ROADMAP 참조.
  재시작은 "조금 느린 resume" 이 아니라 다른 실험이므로 fail closed 한다.

### 8.3 프로파일
| | light | heavy |
|---|---|---|
| 이미지 | ≤100 장 | 전체 (또는 청크) |
| 해상도 | 1/4 | 1/2 또는 원본 |
| iter | 7k | 30k |
| 기본 러너 | local | runpod |

## 9. 계보 (provenance)

```
raw ──▶ dataset ──▶ run ──▶ eval ──▶ export
 sha256   dataset_id   run_id    eval_id    export_id
          config_hash  config_hash + dataset_hash + git_commit
          git_commit   + docker_digest + backend{name, version}
```

run_id 예: `gsplat_20260910_a91f2c`. 6개월 뒤 "이 .ply 는 어느 E57·어느 크롭
설정·어느 commit 에서 나왔나"에 즉답할 수 있어야 한다.

**입력은 하나도 빠지지 않는다.** `source_assets` 에는 결과를 바꾸는 입력이 전부 들어간다 —
Phase 0B 라면 E57 뿐 아니라 매핑 파일, vendor manifest, 외부 이미지까지. 같은 E57 을 다른
매핑 CSV 로 돌리면 다른 결과이므로, E57 만 적힌 기록은 두 결과를 구분하지도 재현하지도
못한다. 한 산출물 트리 안의 여러 artifact 는 **한 번 계산한 같은 digest** 를 공유한다:
`scan_000`·`image_000` 은 특정 파일 안의 index 라서, 어느 바이트를 읽었는지 말할 수 없는
artifact 는 자기 ID 가 무엇을 가리키는지도 말할 수 없다. hash 를 생략했다면 **artifact 가 그
이유를 적는다** (`hash_skipped_reason`) — 값이 비어 있다는 것과 "왜 비었는지" 는 다른 사실이고,
`SourceAsset` 자체에는 이유를 적을 자리가 없으므로 그 진술은 artifact 수준과 입력별 레코드
(`MappingInput`·`ImageAsset`·`ImageOutput`) 에 남는다.

## 10. 청킹 · 중심선

장거리 갱도(100 m–1 km)는 단일 모델로 안 간다. 청킹 기준은 XYZ 격자가 아니라
**중심선 chainage**. 청크는 v1 에서 파일을 실제로 쪼개지 않아도 manifest 에 예약.

`centerline` 은 1급 산출물: 설계 중심선(DXF/측점표) 임포트가 기본, 없으면 TLS
클라우드에서 추출(`core/centerline.py`). 청킹·단면·체적·change 가 모두 이걸 참조한다.

## 11. 형상 · 체적 평가

* **형상**: 양방향 — TLS→GS(accuracy) 와 GS→TLS(completeness) 를 따로. 대칭 Chamfer,
  median, P90/P95, RMSE. 한 방향만 보면 큰 hole 을 놓친다.
* **단면**: 중심선 따라 일정 간격 단면 → A(s). 단면은 **artifact** 로 남긴다
  (`SectionRecord`): 어느 dataset 의, 어느 verified surface 에서, 어느 run 에서, 어떤
  depth provenance 로, 어느 reference axis 를 따라, 어떤 parameters 로 잘랐는가.
  기록한 것은 평가 직전에 전부 지금의 dataset·축·surface 에 대해 다시 대조한다.
* **체적**: 기본 ∫A(s)ds (여굴·미굴 산정 관행), 보조 닫힌 메시 체적.
  `volume.json` 필수 항목: `start_chainage, end_chainage, section_interval,
  valid_section_count, missing_section_count, reference_axis`, 그리고 coverage —
  `requested/integrated/missing_intervals_m`, `covered/requested_length_m`,
  `coverage_fraction`, `segments[]`.
* **결측은 적분하지 않는다**: 적분은 연속된 관측 station 의 run 안에서만 일어난다.
  결측 구간은 interval 로 보고하고 보간하지 않으므로, coverage 가 불완전한 체적은 항상
  과소 추정이다. 서로 다른 holdout 구간 사이도 잇지 않는다. `integrate_sections` ·
  `compare_to_design` · `diff_sections` 가 같은 helper 를 쓴다.
* **`volume_accuracy` 는 surface provenance 를 요구한다** (§5, §1.7): protocol 허용 +
  section artifact + 지금의 dataset/축/surface 와 일치 + `depth_source = minegs_render` +
  선언된 holdout 으로 제한된 적분 + 그 구간의 완전한 coverage. 임의 점군·외부 depth·
  bare series·불완전 coverage 는 diagnostic 으로만 계산한다.
* **설계 대비**: 설계 프로파일이 있으면 Design vs TLS vs 3DGS 를 동일 단면에서 비교해
  overbreak / underbreak / reconstruction error 를 분리한다. 여굴·미굴 체적도 같은
  integration segment 를 쓴다.
* **change**: 두 epoch 의 동일 chainage 구간 차분 → 차분 체적 가설 검증.
  단일 manifest 로는 주장할 수 없다 (§5, pair protocol = Phase 7).

## 12. 시각화

Viser(파이썬 API, WebGL) 를 연구용 UI 로. 프러스텀·초기 포인트·스플랫·TLS 토글·
거리 히트맵·단면 슬라이더·chainage 스크러버. 규약 검증용 2D `overlay`.
`export`: 연구 산출물 `.ply` → 배포 `.spz` (SuperSplat) → legacy `.splat` 선택.
다중 사용자가 필요해지면 FastAPI 를 CLI 위에 얹는다.

## 13. 단계 · 게이트

구현 순서, Phase 별 범위, validation gate(G0–G3), Definition of Done 은
[ROADMAP.md](ROADMAP.md) 가 source of truth 다. 이 문서는 invariant 만 다룬다.

큰 흐름: **0A** Foundation & Contract Freeze → **0B** 실제 E57 ingest → **0C** Metric dataset
golden gate → **0D** Local GS baseline → **1** Metric surface & evaluation → **2** E57
end-to-end MVP(v0.1) → **3** 영상·360 독립 재구성 → **4** Advanced GS / heavy →
**5** 장거리 청킹 → **6** RunPod → **7** Multi-epoch change → **8** Viewer/Export/Web.

두 가지만 여기서 못박는다.

* Phase 는 Gate 를 통과해야 완료다. 코드가 있다는 사실(implemented)과 실데이터에서
  확인됐다는 사실(validated)을 구분해 표기한다.
* 검증되지 않은 경로는 부분 지원하지 않고 fail-closed 한다 (§1.6). 현재 거부 목록은
  ROADMAP.md §6 에 있다.

## 14. 결정 이력

* 2026-09 — Phase 0A closeout: 단일 manifest 는 change claim 불가(§5 change 는 2 epoch),
  `depth_loss` 는 TLS staging 이 COLMAP observation track 을 제거하므로 거부(Phase 4 재설계),
  `normalize_world_space=true` 는 upstream equivalence 검증 전까지 거부. Phase/게이트 정의는
  ROADMAP.md 로 분리.
* 2026-09 — 리포 `e57gs` → `minegs`. `ingest/common/` 은 기존 모듈 그대로.
* 2026-09 — UI 는 웹(Viser → FastAPI). 3DGS 뷰어가 전부 WebGL, 데스크톱 패키징 비용 과다.
* 2026-09 — TLS 단독 경로 순환논증 → 영상 경로 Phase 2, 평가 프로토콜을 계약으로 승격.
* 2026-09 — v2 반영: TLS_GLOBAL/LOCAL_METRIC 분리, capture_groups, chainage 구간
  홀드아웃, COLMAP ≥4.0 global mapper(GLOMAP 독립 저장소 2026-03 아카이브), 360 rig,
  epoch/change, PanoSource 어댑터, v0.1 백엔드 gsplat 단일, INRIA 저장소 미포함,
  CPU/GPU 런타임 분리, Sim3→SE3, 양방향 형상 지표, provenance 계층, Phase 0 을 0A–0D 로 분할.
* 2026-09 — Phase 3 C0 (AD-1): 독립 SfM 출력은 `SOURCE` 가 아니라 **`SFM_INTERNAL`** 이고, 그
  프레임에서 `TLS_GLOBAL` 로 나가는 유일한 경로는 **측정된 Sim(3)** 다 (선언 금지). 함께 고정한
  것 — 정합 support 는 초기 Sim3 대응과 **ICP target 의 합집합**이며 evaluation holdout 과 겹치면
  claim 을 거부한다(scale 의 출처가 독립이어도 pose 가 평가 기준을 보고 최적화되면 leakage다);
  image-only dataset 은 같은 dataset 계약을 쓰되 `init_points` 가 SfM sparse 에서 왔음이 **검증**
  되어야 한다. 근거와 reality audit 은 [docs/PHASE3_CONTRACT.md](PHASE3_CONTRACT.md).
* 2026-09 — Phase 3 구현: image/360 경로가 Phase 2 의 workflow·ledger·report 를 **그대로** 쓴다.
  `INGEST` 가 frame set·재구성·registration, `DATASET` 이 image-only 빌더가 되고 그 뒤 단계는
  Phase 2 handler 그대로다 — 학습·depth·surface·sections·volume 의 구현이 하나뿐이어야 두 경로가
  갈라지지 않는다. 함께 고정한 것 — chainage holdout 은 `init_points.ply` 와 `sparse/0` **양쪽**
  에서 실제로 제외된다(선언만으로는 부족하고, 학습기가 둘 중 어느 쪽으로도 초기화될 수 있다);
  360 crop 이름은 view 를 품는다(depth map 이름이 image stem 하나당 하나이므로); frame set 검사는
  `images/` 를 열거해 record 에 없는 파일을 거부한다(SfM 은 목록이 아니라 디렉토리를 본다).
  §17 구현 기록 참조.
* 보류 — GLUEMAP: 갱도 조건에 특화되나 의존성 무거움. Phase 2 이후 experimental 백엔드.
