# Phase 3 — Image / 360 Independent Reconstruction 계약 (freeze)

이 문서는 Phase 3 구현 **전에** 고정하는 계약이다. 구현이 이 문서를 바꾸는 것이 아니라, 이 문서가
구현의 판정 기준이다.

source of truth: 범위와 gate 는 [docs/ROADMAP.md](ROADMAP.md) §3, 불변식은
[docs/ARCHITECTURE.md](ARCHITECTURE.md) §3·§6.2·§7. 이 문서는 그 둘 위에서 Phase 3 만 다룬다.
Phase 3 는 ROADMAP 의 Phase 3 정의를 바꾸지 않는다.

기준 commit: `main @ ad39f4d` (Phase 2 PR #12 merge).

> Phase 2 E57 end-to-end workflow is implemented and structurally tested.
> Real-data scientific validation remains NOT VALIDATED.
> Phase 2 G2 remains PENDING.

Phase 2 가 main 에 들어간 것과 real E57 G2 는 별개다. 이 문서 어디에서도 Phase 2 G2 를 PASS 로
바꾸지 않는다.

---

## 1. 목적과 non-goals

### 1.1 만드는 것

TLS point cloud 를 **초기 GS geometry 로 쓰지 않는** 독립 image/360 재구성 경로.

```
일반 영상 / 360 영상
 → frame 추출
 → blur / 중복 선별, mask
 → (360) 결정론적 ring crop + COLMAP rig
 → SfM (global / incremental)
 → 독립 sparse reconstruction        ← 여기까지 metric 이 아니다
 → metric registration (측정된 Sim3)
 → 기존 MineGS dataset 계약 (단일 계약)
 → GS 학습
 → Phase 1A/1B/1C 증거 사슬로 평가
 → TLS-assisted GS 와 image-only GS 의 동일 조건 비교
```

### 1.2 non-goals (Phase 3 에서 하지 않는다)

새 reconstruction 알고리즘, 새 SfM 알고리즘, 새 registration 알고리즘, 두 번째 dataset schema,
TSDF/mesh, 2DGS/PGSR, depth supervision, appearance embedding, RunPod, long-tunnel chunking,
viewer/web, multi-epoch, GLUEMAP backend(§14 에서 계속 보류), 임의의 accuracy/coverage 임계값
신설.

### 1.3 Phase 3 가 기존 계약을 바꾸지 않는 지점

* dataset 계약은 하나다. image-only 경로도 같은 `manifest.json` / `sparse/0` / `images/` /
  `init_points.ply` 를 만족해야 한다.
* `LOCAL_METRIC` 은 1 unit = 1 m. Phase 3 가 scale 을 도입하는 곳은 **SfM → TLS_GLOBAL 한 군데뿐**
  이고, 그것은 측정값이며 artifact 에 남는다.
* Gaussian center 는 surface 가 아니다. image-only 경로의 geometry/volume 주장도 Phase 1A/1B/1C 의
  `run → rendered depth → verified surface → sections → volume` 사슬을 그대로 통과해야 한다.

---

## 2. Reality audit — 2026-09-21, `main @ ad39f4d`

Phase 3 는 greenfield 가 아니다. 아래는 **파일 이름이나 docstring 이 아니라 실제 호출 경로**를 따라
확인한 현재 상태다. 분류는 다섯 단계다.

1. `placeholder only` — stub / `NotYetImplementedError` / 아무도 도달하지 않는 코드
2. `implemented not wired` — 로직은 있으나 CLI·파이프라인이 닿지 않음
3. `CLI-wired` — CLI 에서 끝까지 도달함
4. `structurally tested` — 구현을 stub 으로 바꾸면 실패하는 테스트가 있음
5. `production / real-data validated` — 실측 데이터로 검증됨

| capability | 상태 | 근거 (file:line) |
|---|---|---|
| ffmpeg frame 추출 | **CLI-wired** (command builder 만 테스트됨) | `cli/ingest.py:428` → `ingest/video/frames.py:34 extract_frames` 가 실제 `subprocess.run`; 테스트는 `tests/test_train_ingest_viz.py:334` 의 argv 문자열 검사뿐. CLI 는 `--fps` 만 노출한다 — `pattern`·`scale_width`·`start_s`·`duration_s` (`frames.py:16-19`) 는 도달 불가라 긴 4K 영상을 자르거나 줄일 수 없다. ffmpeg `-y` 라서 fps 를 바꿔 같은 디렉토리에 다시 추출하면 프레임이 말없이 섞인다 |
| ffprobe | **implemented not wired** | `frames.py:44 probe_command` — 호출자 없음 |
| blur / dHash 중복 선별 | **structurally tested** | `dedup_blur.py:20 blur_score`, `:27 dhash`; `cli/ingest.py:447` 이 `select_frames` 실행 후 결정을 JSON 으로 씀. 지표 자체는 `tests/test_train_ingest_viz.py:329` 가 실제로 계산 |
| 선별 결정의 identity | **없음** | `cli/ingest.py:458` 이 `dataclass.__dict__` 를 그대로 dump. 원본 video digest·fps·임계값·추출 설정 없음, record id 없음. `--out` 없이 쓰면 콘솔 한 줄 뒤에 JSON 이 붙어 stdout 이 파싱되지 않는다 |
| 선별 결과의 소비 | **없음** | `FrameDecision` 을 읽는 코드가 저장소에 없다. `select` 는 아무것도 지우지도 옮기지도 않고, `sfm` 은 디렉토리의 **모든** 이미지를 feature-extract 한다 — 현재 이 단계는 파이프라인에 아무 효과가 없는 보고서다 |
| 선별의 파일 탐색 | **조용한 누락** | `cli/ingest.py:458` 의 glob 은 비재귀·대소문자 구분(`*.png`, `*.jpg`)이다. `.jpeg`/`.JPG` 는 말없이 빠지고, 링 crop 이 만들 `p0y00/` 같은 하위 디렉토리 구조에서는 `kept 0 / 0` 을 찍고 **exit 0** 으로 끝난다 |
| equirect → pinhole crop 리샘플 | **structurally tested (E57 경로에서 production-wired)** | `ingest/common/equirect.py:74 crop_equirect`; 실제 사용처는 `dataset/materialize.py:347` (E57 파노라마) |
| **video 360 ring crop 생성 경로** | **없음** | `crop_equirect` 를 부르는 video 경로가 없다. `minegs ingest video` 에 crop 명령이 없음 |
| ring crop 기하 (`RingCropSpec`) | **structurally tested** | `equirect.py:23`, `:60 R_scanner_from_cam`; `core/synthetic.py:143`, `dataset/materialize.py:347` 이 사용 |
| COLMAP rig config 작성 | **CLI-wired + structurally tested** | `cli/ingest.py:467` → `video/rig.py:53 write_rig_config`; `tests/…:313` 이 `cam_from_rig_rotation` 구조를 검사 |
| crop ↔ orientation 결속 | **약함 (계약 부재)** | `rig.py:46` 이 `image_prefix = f"{view.name}/"` — 즉 디렉토리 이름이 회전을 가리킨다. rig_config 안의 회전은 명시적이지만, `p0y03/` 에 실제로 yaw index 3 으로 렌더된 crop 이 들어 있는지는 **아무도 검증하지 않는다** |
| mask 생성 (nadir / box) | **implemented not wired** | `video/masks.py:14, 23, 33` 은 배열을 만들고 `:40 write_mask` 가 파일을 쓰지만 **`write_mask` 호출자가 저장소에 하나도 없다**. `nadir_mask_for_crop` 은 테스트에서만 불린다 (`tests/…:322`) |
| mask → COLMAP 전달 | **CLI-wired** | `sfm/colmap_incremental.py:29` 이 `--ImageReader.mask_path` 를 붙이고 `cli/ingest.py:491 --masks` 로 전달됨. 즉 **소비 경로는 있고 생산 경로가 없다** |
| COLMAP feature/matcher/mapper 명령 구성 | **structurally tested (문자열만)** | `sfm/colmap_incremental.py:8 _common`, `:48`; `sfm/colmap_global.py:15`; 테스트 `tests/…:338` 은 `"global_mapper" in cmd` 수준의 문자열 검사 |
| COLMAP 실행 | **배선은 있으나 한 번도 실행된 적 없음** | `sfm/base.py:43 run` 이 `subprocess.run(..., check=True)` 로 실제 실행하고 `sfm.log` 에 명령을 남긴다(`ab` 모드라 재실행이 구분 없이 덧붙는다). CI·docker·이 작업 환경 어디에도 COLMAP 이 없어 `base.py:52` 이후 한 줄도 실행된 적이 없다. 첫 실제 실행에서 걸릴 것으로 보이는 두 곳: `base.py:53` 이 `work_dir` 만 만들고 mapper 의 `--output_path`(=`work_dir/sparse`) 를 만들지 않는다; `sequential_matcher` 에 `--SequentialMatching.loop_detection 1` 만 주고 vocab tree 경로가 없다 |
| `--mapper` 값 검증 | **fail-open** | `sfm/base.py:80-88 get_sfm_backend` 은 `"global"` 만 보고 나머지는 전부 incremental 로 떨어뜨린다. `--mapper globl` 이 경고 없이 incremental 을 돌린다 |
| `SfMOptions.mapper` | **dead field** | 어떤 backend 도 `opts.mapper` 를 읽지 않는다. 실제 선택은 `get_sfm_backend` 의 인자로 이뤄진다 (`cli/ingest.py:498`) |
| SfM 옵션 노출 범위 | **CLI 도달 불가** | `camera_model`, `camera_params`, `matcher`, `use_gpu`, `extra` (`base.py:18-24`) 에 CLI 옵션이 없다 — matcher 는 항상 sequential, `use_gpu` 는 `Dockerfile.cpu` 가 있어도 항상 `1` |
| GLUEMAP 도달성 | **CLI 도달 불가** | `cli/ingest.py:498` 이 backend 이름을 `"colmap"` 으로 고정한다. ROADMAP 의 fail-closed 행이 설명하는 거부를 사용자가 실제로 유발할 수 없다 |
| 다중 sparse model 처리 | **미정의 (fail-open)** | `base.py:58` 과 `colmap_global.py:43` 이 `sparse/0` 을 하드코딩한다. COLMAP 이 분리된 component 를 여러 개 낼 때 조용히 0 번을 집는다 |
| `SfMResult` | **휘발성 (증거 아님)** | `sfm/base.py:29` — `sparse_dir`, `n_registered`, `n_points`, `backend`, `log` 뿐인 in-memory dataclass. 디스크에 남는 것은 COLMAP 산출물과 `sfm.log` 뿐이고, backend 버전·옵션·입력 이미지 집합·모델 digest·frame 의미는 **아무 데도 기록되지 않는다** |
| GLUEMAP | **placeholder only** | `sfm/gluemap.py:14` → `NotYetImplementedError` |
| COLMAP model IO | **structurally tested** | `ingest/common/colmap_io.py`; `tests/test_pointcloud_colmap.py` |
| Umeyama Sim3 | **structurally tested** | `eval/register/sim3.py:11`; `tests/test_eval.py:87` 이 scale 1.02 복원을 검사 |
| RANSAC 대응 정합 | **structurally tested, fail-open 잔존** | `register/initial_alignment.py:35`; inlier 가 3 개 미만이면 `:60` 에서 **전체 점으로 되돌아가 fit** 한다 (조용한 degrade) |
| SE(3) ICP | **structurally tested** | `register/rigid_icp.py:24`; `umeyama(..., with_scale=False)` (`:58`) 이므로 **scale 을 바꾸지 않는다**. 다만 `converged=False` 여도 결과를 반환한다 (호출자가 판정해야 함) |
| registration diagnostics | **사실상 미검증** | `register/diagnostics.py:38 diagnose` 를 "완벽한 정합"(rmse 0, inlier 1.0) 을 돌려주는 상수 stub 으로 바꿔도 **640 개 테스트가 전부 통과한다**. `tests/test_eval.py:108` 의 단언(`inlier_ratio > 0.99`, `scale == 1.0`)이 상수로 충족되기 때문이다. 또 `rmse_m` 는 **inlier 에 대해서만** 계산되므로 (`:52`) inlier_ratio 없이는 의미가 없다 |
| **`minegs eval register` 자체** | **깨져 있음 (모든 호출이 실패)** | `cli/eval_cmd.py:180` 의 `T = res.T @ T0` 는 `SE3 @ Sim3` 다. `core/frames.py:182 SE3.__matmul__` 은 SE3 가 아니면 `NotImplemented` 를 돌려주고 `Sim3` 에는 `__rmatmul__` 이 없다 → **`TypeError` 로 종료, 출력 파일 없음**. 타깃을 주는 분기(`align_correspondences` → Sim3)와 안 주는 분기(`Sim3.identity()`)가 **둘 다** 여기로 온다. 이 저장소의 유일한 정합 CLI 는 한 번도 동작한 적이 없다 (실제 실행으로 확인: exit 1, `reg.json` 미생성) |
| registration → manifest | **수작업, 게다가 도달 불가** | `cli/eval_cmd.py:152` 의 도움말이 그대로 말한다: *"registration JSON (paste into manifest.registration)"*. 위 버그 때문에 붙여넣을 JSON 자체가 생성되지 않는다 |
| registration 의 **적용** | **없음** | `Registration.sim3` (`core/manifest.py:147`) 과 `Scale.factor` (`:135`) 는 **호출자가 0 건**이다. `manifest.registration` 은 게이트 토큰으로 읽히고(`protocol.py:75`, `manifest.py:320-322`) 보고될 뿐, **어떤 좌표도 이 변환으로 옮겨지지 않는다** |
| target 없는 registration | **fail-open** | `cli/eval_cmd.py:174` — 타깃이 없으면 노란 경고를 찍고 **`Sim3.identity()` 로 진행**한다. 즉 "이미 metric 이라고 가정" |
| registration support 기록 | **없음** | `diagnose` 는 `n_source` 만 남긴다. **어느 TLS 기하를 정합 근거로 썼는지 남지 않는다** |
| crop 레이아웃 정합성 | **모순, 검사 없음** | `rig.py:34` 의 기본 prefix 는 `p0y00/` 같은 **디렉토리**인데, 저장소의 유일한 crop 생산자(`dataset/materialize.py:348`)는 `{station}_{view}.png` 를 **한 디렉토리에 평평하게** 쓴다. 게다가 `colmap_incremental.py:20-21` 은 `--ImageReader.single_camera_per_folder 1` 을 항상 준다. 셋이 서로 맞지 않는데 아무도 확인하지 않는다 |
| `write_mask` 의 `images_dir` | **무시됨** | `video/masks.py:40` 의 시그니처에 있지만 본문(`:42-45`)이 쓰지 않는다. 호출자도 없는 함수의 무시되는 파라미터 |
| COLMAP 포맷 적합성 | **검증 없음** | 테스트의 모든 `cameras.txt`/`images.txt`/`rigs.txt` 는 `colmap_io.py:173 write_model` 이 쓰고 `:278 read_model` 이 되읽는다. 서로만 맞는 한 쌍이어도 통과한다 — 진짜 COLMAP 이 읽어 준 적은 없다 |
| video source 의 claim 차단 | **부분 검증** | `eval/protocol.py:75-83` — `source in ("video","video360")` 인데 `registration` 이 없거나 품질 지표가 0 이면 metric claim 거부. 테스트는 `reg is None` 분기(`tests/test_eval.py:81-84`)만 덮고, **품질 지표가 0 인 분기(`:81-83`)는 테스트가 없다** |
| image-only dataset 빌더 | **없음** | 유일한 실빌더는 `cli/dataset.py:258 from-e57`. `dataset/materialize.py:755` 이 `initialization.source="tls"` 를, `:768` 이 `source="tls"` 를 **하드코딩**한다 |
| SfM frame 의 표현 | **없음** | `core/frames.py:23 Frame` 은 `SOURCE / TLS_GLOBAL / LOCAL_METRIC / BACKEND_INTERNAL` 뿐. 감사 시점의 ARCHITECTURE §3 은 SOURCE 를 "스캐너 로컬 / SfM 임의" 로 설명했지만, SOURCE → TLS_GLOBAL 의 문 (`dataset/build_config.py:30 SourceFrameConfig`) 은 `se3()` (`:60`) 를 돌려준다 — **구조적으로 scale 을 담을 수 없다**. §3 AD-1 의 결정에 따라 이 C0 에서 ARCHITECTURE §3·§14 를 갱신했고, enum 값은 Phase 3A 에서 들어온다 |
| manifest 의 registration/scale 어휘 | **이미 존재** | `core/manifest.py:138 Registration`, `:32 DatasetSource = "tls"\|"video"\|"video360"`, `ScaleBasis = "tls_pose"\|"sim3_to_tls"\|"known_target"`, `InitSource = "tls"\|"sfm_sparse"\|"random"` |

### 2.1 재사용할 수 있는 것

* **frame 선별 지표**: `blur_score` / `dhash` / `hamming` (`dedup_blur.py`) — 실제 로직, 테스트 있음.
* **ring crop 기하 전체**: `RingCropSpec` / `CropView.R_scanner_from_cam` / `crop_equirect`
  (`equirect.py`) — E57 파노라마 경로에서 이미 production 으로 쓰인다. 360 영상은 **같은 코드를
  다른 입력에 연결**하는 문제이지 새 구현이 아니다.
* **COLMAP rig**: `rig_from_ring` / `rig_config_json` (`rig.py`).
* **COLMAP 명령 구성과 실행 골격**: `sfm/base.py`, `colmap_incremental.py`, `colmap_global.py`.
  mask 전달(`--ImageReader.mask_path`)과 rig 구성(`rig_configurator`)이 이미 들어 있다.
* **정합 수학**: `umeyama`, `align_correspondences`, `icp_point_to_point` — 실제 로직이고
  `tests/test_eval.py:87,97` 이 합성 데이터로 검증한다. Phase 3 는 여기에 **게이트와 provenance 를
  씌우는 것**이지 다시 구현하는 것이 아니다. **단 `diagnose` 는 예외다** — 상수 stub 으로 바꿔도
  테스트가 통과하므로, 재사용하되 먼저 실제 테스트를 붙여야 한다.
* **manifest 어휘**: `Registration`, `ScaleBasis`, `InitSource.sfm_sparse`, `DatasetSource.video*` —
  schema 를 새로 만들 필요가 없다.
* **claim 경계**: `eval/protocol.py:75` 의 video 규칙, 그리고 Phase 1A/1B/1C 의 surface/section/volume
  증거 사슬 전체.
* **구조 게이트 방식**: Phase 2 의 `tests/test_e2e_gate.py` 와 대체 seam(`renderer=`, `trainer=`)
  기록 방식.

### 2.2 placeholder / dead / 검증되지 않은 것

* `write_mask` — 호출자 없음. `probe_command` — 호출자 없음. `GLUEMAP` — `NotYetImplementedError`.
* `SfMBackend.run` — 실행 경로는 있으나 **한 번도 실행된 적이 없고** 테스트가 없다. COLMAP 4.x 플래그
  (`global_mapper`, `rig_configurator`, `--FeatureExtraction.use_gpu`, `--SequentialMatching.loop_detection`)
  는 **문자열로만** 검증되어 있다. 실제 COLMAP 과의 호환성은 Phase 3 에서 처음 확인된다. 이미 눈에
  보이는 두 곳(`--output_path` 디렉토리 미생성, vocab tree 없는 `loop_detection`)은 3A 에서 먼저 고친다.
* **`minegs eval register` 는 깨져 있다.** `SE3 @ Sim3` 타입 오류로 모든 호출이 실패한다
  (`cli/eval_cmd.py:180`). 실제 실행으로 확인했다. 즉 "정합 결과를 손으로 manifest 에 붙여 넣는다" 는
  현재 상태조차 성립하지 않는다 — 붙여 넣을 파일이 만들어지지 않는다.
* **정합 결과를 적용하는 코드가 없다.** `Registration.sim3` 과 `Scale.factor` 는 호출자가 0 건이다.
  오늘의 `manifest.registration` 은 claim 게이트의 토큰이지 좌표를 옮기는 변환이 아니다.
* `diagnose` 는 상수 stub 으로 바꿔도 전체 테스트가 통과한다. 테스트가 있는 것과 검증되는 것은 다르다.
* video → 360 crop 경로 자체가 없다.
* image-only dataset 빌더가 없다.
* registration 결과를 manifest 로 옮기는 자동 경로가 없다 (복사/붙여넣기).
* `sparse/0` 하드코딩, RANSAC 전체 fallback, target 없는 identity Sim3, 미수렴 ICP 반환 — 모두
  **조용한 degrade** 이고 Phase 3 에서 fail-closed 로 바꿔야 한다.

### 2.3 문서 정정

ARCHITECTURE §3 은 `SOURCE` 를 "스캐너 로컬 / SfM 임의" 로 정의하고 있었다. AD-1 이 그 절반을
떼어 내므로 §3 의 프레임 그림, §6.2·§7 의 포인터, §14 의 결정 이력을 이번 C0 에서 함께 고친다.
문서 변경뿐이고 코드는 건드리지 않는다.

ROADMAP §1 표의 Phase 3 행은 "부분 implemented (커맨드 빌더·rig·정합), 미검증" 이다. 방향은 맞지만
두 가지가 빠져 있어 사실관계만 보강한다: (1) SfM 실행 경로는 존재하되 **한 번도 실행된 적이 없고**
CI 에 COLMAP 이 없다, (2) **360 영상 crop 경로와 image-only dataset 빌더는 아예 없다**. 구현하지
않은 것을 implemented 로 바꾸지 않는다.

---

## 3. Architecture decisions

### AD-1 — SfM 출력 frame: `SFM_INTERNAL` 을 명시적으로 만든다

독립 SfM 의 출력은 **scale 까지 임의인 similarity frame** 이다. 이것을 처음부터 `TLS_GLOBAL` 이나
`LOCAL_METRIC` 이라고 부르지 않는다.

기존 `SOURCE` 를 재사용하지 **않는** 이유는 이름 취향이 아니라 문의 모양 때문이다. `SOURCE` →
`TLS_GLOBAL` 의 유일한 문인 `SourceFrameConfig` 는 `explicit_identity` / `explicit_transform` 두
모드이고 `se3()` 를 돌려준다 (`dataset/build_config.py:30-63`). 즉 **선언이며 rigid 다**. 임의
scale 의 재구성을 `SOURCE` 로 부르는 순간, `mode: explicit_identity` 한 줄로 "이 파일의 좌표가 곧
측량 기준" 이라고 **측정 없이** 선언할 수 있게 된다. 그것이 Phase 3 가 막아야 할 바로 그 사고다.

```
SFM_INTERNAL          독립 SfM 의 자기 좌표 — 임의 scale, 임의 원점
      │  측정된 Sim(3), registration artifact 에 근거와 품질이 함께 남는다
      ▼
TLS_GLOBAL            측량 좌표, m
      │  기존 규칙 그대로: SE(3), translation-only baseline, scale 1
      ▼
LOCAL_METRIC          학습 입력 좌표, 1 unit = 1 m
```

규칙:

* `SFM_INTERNAL` 은 `Frame` 에 값을 하나 더한다. 코드에서 frame 은 문자열로 다루고 `Frame.X` 참조가
  현재 0 건이므로 파급은 작다 — `core/frames.py:23` 과 frame 을 열거하는 소수의 `Literal` 뿐이다.
* **`SFM_INTERNAL` 클라우드는 metric 을 요구하는 어떤 소비자에도 들어갈 수 없다.** 기존 소비자들은
  `TLS_GLOBAL`/`LOCAL_METRIC` 을 요구하므로 별도 조치 없이 거부된다 (fail-closed by construction).
* `SFM_INTERNAL` 에서 나가는 유일한 출구는 **측정된 Sim3** 이다. `SourceFrameConfig` 의 선언 문은
  `SFM_INTERNAL` 을 받지 않는다.
* Phase 3A 산출물에는 `TLS_GLOBAL` / `LOCAL_METRIC` 문자열이 등장하지 않는다 (Phase 0B 와 같은 규칙).
* **ARCHITECTURE §3 은 이번 C0 에서 함께 고친다.** 그 문서가 `SOURCE` 를 "스캐너 로컬 / SfM 임의"
  라고 정의하고 있었기 때문에, 이 계약만 바꾸면 source of truth 가 둘이 된다. §3 의 프레임 그림과
  §14 결정 이력을 지금 갱신하고, `Frame` enum 에 값이 들어가는 것은 Phase 3A 로 남긴다 — 결정의
  기록이 먼저이고 구현이 뒤따른다.

### AD-2 — TLS registration/evaluation leakage (BLOCKING, 이번 C0 의 핵심 결정)

image-only SfM 은 metric scale 이 없다. TLS 로 Sim3 를 fit 하면 scale 과 정렬을 얻지만, **같은 TLS
점으로 정렬을 맞추고 그 점에 정확도를 재면 평가가 오염된다.**

Phase 3 는 **두 모델을 모두 허용하되, 어느 쪽인지 artifact 가 말하게 한다.** 이 구분은 새 어휘가
아니라 이미 있는 `Scale.basis` (`core/manifest.py:133`) 로 표현한다.

| scale basis | 정합 근거 | TLS holdout | 허용되는 주장 |
|---|---|---|---|
| **`known_target`**, TLS ICP 없음 (선호) | 측량 타깃·기준선 등 **TLS 가 아닌 독립 metric 증거**뿐 | 전체를 평가에 쓸 수 있다 | `geometry_accuracy`, `volume_accuracy` 가능 |
| **`known_target`** + 선언된 TLS 부분집합으로 ICP | 타깃 + **그 TLS 부분집합** | 그 부분집합과 **겹치지 않는** 구간만 | support ∩ holdout = ∅ 일 때만 claim |
| **`sim3_to_tls`** (허용, 조건부) | **선언된 registration support 집합**의 TLS 기하 | support 와 **겹치지 않는** 구간만 | support ∩ holdout = ∅ 일 때만 claim, 아니면 거부 |
| 그 외 (타깃 없음, **또는 전체 TLS 클라우드 ICP**) | 평가 기준 클라우드 자체 | — | **diagnostic 전용**, 영구히 claim 불가 |

**`basis` 는 scale 의 출처일 뿐 pose 의 출처가 아니다.** 독립 타깃으로 scale 을 얻었더라도 그 뒤
전체 TLS 클라우드에 ICP 를 돌리면 **최종 `T_tls_from_sfm` 은 평가 기준 클라우드를 보고 최적화된
것**이다. 그 상태로 같은 클라우드에서 accuracy 를 재면 leakage 다. 그래서 support 는 basis 가 아니라
**최종 변환을 움직인 기하 전체**로 정의한다: ① 초기 Sim3 의 대응과 ② ICP 의 target 의 **합집합**
(§7 `support_ranges_m`). 전체 TLS 로 ICP 를 돌렸다면 support 는 사실상 TLS 전체이므로 disjoint 한
holdout 이 남지 않고, `known_target` 이어도 결과는 diagnostic 이다. ICP 를 claim 경로에서 쓰려면
target 을 **선언된 부분집합**으로 제한해야 한다.

강제 조건:

1. **정합에 무엇을 썼는지 남는다.** `RegistrationRecord` (§7) 가 초기 Sim3 의 support 와 ICP 의
   support 를 **각각** 기록한다: 종류(타깃 CSV / TLS 부분집합), 파일 digest, **chainage 구간**,
   대응 개수, inlier, residual. 둘의 합집합이 claim gate 가 보는 support 다.
2. **겹침을 판정할 수 있다.** support 구간(①∪②)과 `split.geometry_holdout.chainage_ranges_m` 의
   교집합을 계산할 수 있어야 한다. 계산할 수 없으면 (support 구간 미기록, 또는 ICP 를 돌렸는데
   `icp_support` 가 비어 있음) 그것 자체가 거부 사유다.
3. **겹치면 fail closed.** 겹친 상태로 `geometry_accuracy` / `volume_accuracy` 를 내지 않는다.
   숫자를 죽이는 것이 아니라 **claim 을 죽인다** — 같은 숫자를 `geometry_diagnostic` 으로는 낼 수 있다.
4. **diagnostic 정렬과 claim 정렬을 구분한다.** 전체 클라우드 ICP 는 "그림이 맞는지 보는" 용도로
   유효하고, 그 결과로 만든 숫자는 **basis 와 무관하게** 영구히 diagnostic 이다.
5. **비교는 같은 평가 영역에서 한다.** TLS-assisted GS 와 image-only GS 를 비교할 때 holdout,
   section 파라미터, 적분 구간이 같아야 한다 — Phase 1C/2 의 `eval/volume/paired.py` 규칙(같은 grid,
   공통 구간만)을 그대로 쓴다.

이 결정은 Phase 3B 에서 코드가 된다. 새 알고리즘은 없다: `diagnose` 가 이미 내는 숫자에 **support
기록과 겹침 판정**을 붙이고, `eval/protocol.py:75` 의 video 규칙을 확장하는 일이다.

#### AD-2.1 — 지금 코드에 있는 leakage 경로 (Phase 3B 가 닫아야 할 목록)

오늘 **활성** 인 누수는 없다. image-only dataset 빌더가 아예 없기 때문이다. 아래는 전부 **잠재**이고,
Phase 3B 가 그 빌더를 쓰는 순간 실제가 된다. 감사에서 확인한 것만 적는다.

1. **`source="tls"` 하드코딩이 video 게이트를 통째로 우회한다.** `dataset/materialize.py:755`·`:768`.
   Phase 3 가 새 빌더 대신 `from-e57` 를 재사용하면 manifest 가 `source="tls"` 라고 말하게 되고,
   그러면 `core/manifest.py:320` 과 `eval/protocol.py:74-83` 의 **video 정합 요구가 둘 다 적용되지
   않는다**. image-only 재구성이 정합 기록 없이 metric claim 을 받는다. 가장 먼저 닫아야 할 하나.
2. **holdout 선언 요구가 `sfm_sparse` 에서 건너뛰어진다.** `eval/protocol.py:109` 는
   `if missing and init.source == "tls"` 다. image-only dataset 은 `init.source="sfm_sparse"` 이므로
   holdout 구간을 init 제외 목록에 선언하지 않아도 거부되지 않는다.
3. **정합 기준이 학습 초기화 클라우드를 가리키고 있다.** `configs/dataset/video.yaml:28` 과
   `video360.yaml:36` 의 `register.reference_tls` 는 동반 TLS dataset 의 **`init_points.ply`** —
   즉 그 TLS 모델을 학습 초기화한 바로 그 점들이다. 아직 아무도 읽지 않지만, 쓰인 대로 배선하면
   평가용이 아니라 학습 초기화 기하에 정합하게 된다. 정합 기준은 **평가측 TLS 산출물**이어야 한다.
4. **init 파일 경로가 자유롭고 dataset hash 밖으로 나갈 수 있다.** `manifest.initialization.file` 은
   dataset 디렉토리에 붙는 임의 상대경로이고 `core/manifest.py:363` 은 존재 여부만 본다. 반면
   `train/runner/base.py:35` 의 dataset hash 는 리터럴 `init_points.ply` 만 glob 한다. `../raw/tls_full.ply`
   를 가리키면 학습 초기화로 읽히면서 dataset hash 에는 잡히지 않는다.
5. **init 클라우드의 출처를 확인하는 코드가 없다.** `sfm_sparse` 라고 선언하고 TLS PLY 를 놓아도
   모든 소비자가 믿는다 (`golden_gate.py:218-233` 은 frame 과 bbox 만, `train/staging.py:113` 은
   frame 이 `UNKNOWN` 이어도 통과).
6. **`sparse/0/points3D.txt` 는 오늘 TLS 기하의 두 번째 사본이다** (`materialize.py:675-686`), 그리고
   manifest 에는 points3D 의 출처를 적는 칸이 없다. image-only 경로에서는 여기가 SfM 구조여야 한다.
7. **staging 은 언제나 init 파일로 points3D 를 대체한다** (`train/staging.py:111`, `use_init_points` 는
   CLI 에서 끌 수 없다). 반대로 끄면 `init_source="points3D.txt"` 로 기록되는데, 이 저장소가 만들 수
   있는 모든 dataset 에서 그 파일도 TLS 다 — 즉 "TLS 로 초기화하지 않았다" 고 기록된 run 이 TLS 로
   초기화된다.
8. **평가 기준 클라우드의 identity 가 어디에도 안 남는다.** `eval/geometry/evaluate.py` 는 `--tls-ply`
   를 자유 경로로 받아 frame 과 chainage 만 보고, `GeometryReport` 에 기준 파일의 해시가 없다. 그래서
   같은 `raw/tls_full.ply` 가 정합 대상이자 평가 기준이어도 **어느 쪽 artifact 에도 흔적이 없다.**
9. **정합 support 를 적을 칸이 없다.** `diagnose` 는 `n_source` 만 남기고 (`diagnostics.py:52-62`),
   `Registration` 은 `extra="forbid"` 라 손으로 붙여 넣어도 support 를 표현할 수 없다.
10. **SfM 출력 디렉토리가 dataset 의 `sparse/0` 과 형태가 같다.** `sfm/base.py:59` 가 만드는
    `work_dir/sparse/0/*.txt` 는 `core/manifest.py` 가 LOCAL_METRIC 이라고 선언하는 레이아웃과
    구별되지 않는다. **임의 scale 임을 말하는 표지가 없다** — AD-1 이 필요한 이유다.
11. **360 crop 의 convention 이 기본값으로 E57 을 사칭한다.** `equirect.py:75` 의 `conv` 기본값은
    `PanoConvention()` 이고 그 `source` 는 문자열 `"E57Embedded"` 다. 보정 없이 만든 video crop 이
    E57 에서 온 것처럼 manifest 에 기록되고, `fix_intrinsics: true` 와 고정 rig 외부파라미터까지
    더해지면 **아무도 측정하지 않은 기하 구속**이 SfM 에 주어진다. golden gate 는 spherical 경로에서
    orientation 채점을 건너뛰므로 잡히지 않는다.
12. **`SfMOptions.camera_params`** (`sfm/base.py:21` → `colmap_incremental.py:25`) 는 CLI 에 없지만
    옵션 하나만 열면 TLS 로 측정한 intrinsics 를 "독립" 재구성에 못 박을 수 있고, 그 사실이 기록되지
    않는다.
13. **`ingest/e57/pose_to_colmap.py:21 stations_to_colmap`** 은 TLS station pose 와 `RingCropSpec` 으로
    완전히 포즈가 잡힌 COLMAP 모델을 만든다. "360 rig 에 초기 포즈를 주자" 는 쉬운 지름길이고, 그
    지름길은 독립 재구성이 아니다.
14. **`materialize.py:388-391 sanity_checks`** 는 카메라 중심이 TLS bounding box 밖이면 거부한다 —
    SfM 모델을 이 빌더에 통과시키면 **TLS 범위가 어떤 카메라를 받아들일지 결정**하게 된다.
15. **mask 는 학습까지 간다.** `train/runner/base.py:38` 의 dataset hash 가 `masks/**/*` 를 포함하고
    `train/staging.py:102-106` 이 staged tree 로 복사한다. 오늘 mask 는 TLS 입력이 없다 — 그 상태를
    유지해야 한다.

이 목록은 §13 의 fail-closed 표로 이어진다.

### AD-3 — SfM 증거는 artifact 여야 한다

`SfMResult` (`sfm/base.py:29`) 는 프로세스가 끝나면 사라진다. 남는 것은 COLMAP 산출물과 `sfm.log`
뿐이고, **어느 이미지 집합을, 어느 backend 의 어느 버전이, 어떤 옵션으로 재구성했는지** 알 수 없다.
그 상태로는 "이 dataset 은 image-only 로 만들어졌다" 를 나중에 증명할 수 없다.

Phase 3A 는 `SfmRecord` 를 만든다. 새 아키텍처가 아니라 `SurfaceRecord` / `DepthManifest` 와 **같은
모양** 이다: 디스크에 있고, 자기 identity 를 갖고, 소비자가 다시 읽어 검증한다.

### AD-4 — 전처리 증거도 artifact 여야 한다

현재 `minegs ingest video select --out` 은 결정 리스트를 dump 하지만 원본 video digest 도, 추출
설정도, record id 도 없다. Phase 3A 는 `FrameSetRecord` 로 **추출 → 선별 → (360) crop → mask** 를
하나의 계보로 묶는다.

### AD-5 — dataset 은 하나, 문만 하나 더

두 번째 schema 를 만들지 않는다. `materialize.py` 가 하드코딩한 `source="tls"` /
`initialization.source="tls"` 를 **경로에 따라 결정되도록** 열고, image-only 경로에서는

* `source`: `video` 또는 `video360`
* `initialization.source`: `sfm_sparse`
* `init_points.ply`: **SfM sparse 점**을 측정된 Sim3 로 TLS_GLOBAL → LOCAL_METRIC 으로 옮긴 것

이어야 한다. 그리고 이것은 **선언이 아니라 검증 가능해야 한다**: dataset 은 자신의 init 클라우드가
어느 `SfmRecord` 에서 나왔는지 기록하고, 빌더는 그 결속을 다시 확인한다. TLS 는 이 경로에서
registration/evaluation 증거이지 training initialization 이 아니다.

### AD-6 — claim 경계

image-only dataset 은 기본적으로 `geometry_diagnostic` 이다. `geometry_accuracy` /
`volume_accuracy` 로 올라가려면 **전부** 충족해야 한다.

1. `RegistrationRecord` 가 있고 품질 게이트를 통과한다 (§7).
2. AD-2 의 leakage 규칙을 통과한다.
3. `scale.basis` 가 무엇이든 **support ∩ holdout = ∅** 이다. `known_target` 은 *scale* 의 출처를
   말하지 pose 의 출처를 말하지 않으므로, 타깃으로 scale 을 잡았더라도 TLS 로 pose 를 refine 했다면
   그 TLS 가 support 다 (AD-2). TLS support 가 전혀 없으면 support 는 빈 집합이고 overlap 도 없다.
4. Phase 1A/1B/1C 사슬을 그대로 통과한다 — 검증된 `minegs_render` depth 에서 나온 verified surface,
   재현되는 sections, gap-safe volume. **Gaussian center 를 surface 로 쓰지 않는다.**
5. 학습이 holdout 이미지·구간을 보지 않았다 (기존 `split` / `init.excluded_chainage_ranges_m` 규칙).

하나라도 불충분하면 숫자는 나오되 claim 은 나오지 않는다.

---

## 4. Image / 360 input 계약

일반 영상과 360 영상은 **다른 입력**이다. 하나의 전처리인 척하지 않는다.

### 4.1 일반 영상 / 이미지 집합

```
video (또는 이미지 디렉토리)
 → frame 추출 (fps, scale, start/duration)     … 설정과 원본 digest 가 기록된다
 → blur / 중복 선별                             … 결정이 프레임별로 기록된다 (삭제하지 않는다)
 → (선택) mask                                  … 생산 경로가 기록되고 COLMAP 에 전달된다
 → SfM 입력 이미지 집합
```

* 카메라는 하나이고 intrinsics 는 SfM 이 추정한다 (`fix_intrinsics: false` 가 기본).
* 프레임 선별은 **파일을 지우지 않는다**. 어떤 프레임이 왜 빠졌는지가 증거다.

### 4.2 360 영상

```
equirectangular video / 이미지
 → frame 추출
 → blur / 중복 선별 (파노라마 수준)
 → 결정론적 ring crop (N yaw × M pitch, 합성 K)
 → crop ↔ parent frame ↔ (yaw, pitch, K) 결속이 artifact 에 남는다
 → nadir mask (삼각대·작업자)
 → COLMAP rig (같은 parent frame 의 crop 들이 한 rig)
 → SfM (fix_intrinsics: true)
```

* crop 의 intrinsics 는 합성이므로 정확히 알려져 있다 (`RingCropSpec.K()`).
* **crop 의 orientation 은 파일 이름이나 인덱스가 아니라 명시적 기록이 증거다.** 현재
  `rig_config_json` 은 `image_prefix = "p0y03/"` 같은 디렉토리 규약으로 회전을 결속한다
  (`rig.py:46`). rig_config 안의 회전은 명시적이지만 **그 디렉토리에 실제로 그 yaw 로 렌더된 crop 이
  들어 있는지는 아무도 확인하지 않는다.** Phase 3A 는 crop 마다 `(parent_frame_id, yaw_deg,
  pitch_deg, K, digest)` 를 기록하고, rig 생성은 그 기록에서 파생시킨다. 이름은 편의이고 기록이
  증거다.
* 한 parent frame 의 crop 들은 광학 중심을 공유한다 — rig 의 `cam_from_rig` 는 회전만, translation 은 0.

---

## 5. 전처리 artifact — `FrameSetRecord`

SfM 에 들어가는 이미지 집합 하나를 설명하는 디스크 artifact. 기존
`ProvenanceRecord` / `sha256_tree` 를 재사용한다.

| 필드 | 내용 |
|---|---|
| `frameset_id` | 이 집합의 identity |
| `kind` | `video` \| `video360` \| `image_set` |
| `source` | 원본 파일/디렉토리 이름과 **digest** (video 는 파일 digest) |
| `extraction` | fps, scale, start/duration, ffmpeg argv, ffmpeg 버전 |
| `selection` | 프레임별 `(name, blur, hash, keep, reason)` + 임계값 |
| `crops` | 360 일 때: `RingCropSpec` + crop 별 `(parent_frame, yaw, pitch, K, digest)` |
| `masks` | mask 생산 방식(nadir/box·파라미터)과 파일 digest, 어느 이미지에 대응하는지 |
| `images` | SfM 에 실제로 들어가는 이미지 목록과 digest, 집합 digest |
| `frame` | `SFM_INTERNAL` 이전 단계이므로 frame 주장 없음 |

규칙: 선별에서 빠진 프레임도 **기록에 남는다**. mask 는 이미지 이름과 **대응이 확인된 것만** 유효하다
(COLMAP 은 이름이 어긋난 mask 를 조용히 무시한다).

---

## 6. SfM artifact 와 frame semantics — `SfmRecord`

| 필드 | 내용 |
|---|---|
| `sfm_id` | identity |
| `frameset_id` + `images_sha256` | 무엇을 재구성했는지 (재확인 가능) |
| `backend` / `backend_version` | `colmap_global` \| `colmap_incremental`, 실제 `colmap --version` 출력 |
| `commands` | 실행된 argv 목록 (구성된 것이 아니라 실행된 것) |
| `options` | matcher, camera model, fix_intrinsics, rig_config digest, masks 사용 여부 |
| `model` | `sparse/0` 의 digest, 등록 이미지 수, 3D 점 수, 카메라 모델/파라미터 |
| `components` | COLMAP 이 만든 sparse model 개수. **1 이 아니면 선택은 명시적이어야 한다** |
| `frame` | `SFM_INTERNAL` |
| `metric_state` | `arbitrary_scale` — 이 재구성은 metric 이 아니다 |
| `substitution` | SfM 이 대체 구현으로 실행되었는가 (`real_sfm_execution: false`) |

규칙:

* **`sparse/0` 하드코딩을 끝낸다.** component 가 여럿이면 어느 것을 쓰는지 명시적으로 기록하고,
  기본값은 "여러 개면 거부" 다. 조용히 0 번을 집지 않는다.
* `SfmRecord` 없이 만들어진 sparse model 은 Phase 3 의 어떤 metric 경로에도 들어갈 수 없다.
* `metric_state: arbitrary_scale` 은 장식이 아니다. 이 값이 `registered_metric` 으로 바뀌는 유일한
  방법은 §7 의 registration 이다.

---

## 7. Metric registration 계약 — `RegistrationRecord`

```
SFM_INTERNAL sparse / 포인트
      │  ① 초기 Sim(3)  : 타깃 대응 (known_target) 또는 선언된 TLS support (sim3_to_tls)
      │  ② SE(3) ICP    : scale 을 바꾸지 않는 정밀화 (선택)
      │  ③ diagnostics  : scale, rmse, median, p90, inlier_ratio, n
      ▼
TLS_GLOBAL
```

| 필드 | 내용 |
|---|---|
| `registration_id` | identity |
| `sfm_id` | 무엇을 정합했는지 |
| `basis` | `known_target` \| `sim3_to_tls` |
| `initial_support` | ① 에 쓴 것: 타깃 CSV digest **또는** TLS 파일 digest + 사용한 부분집합의 정의 |
| `initial_support_ranges_m` | ① 의 support 가 덮는 chainage 구간 |
| `icp_support` | ② 에 쓴 target 기하: TLS 파일 digest + 부분집합의 정의 (ICP 를 돌렸다면 필수) |
| `icp_support_ranges_m` | ② 의 support 가 덮는 chainage 구간 |
| `support_ranges_m` | 위 둘의 **합집합**. claim gate 가 holdout 과 겹침을 판정하는 값 |
| `T_tls_from_sfm` | 측정된 Sim3 (scale 포함) |
| `icp` | 사용 여부, `converged`, iterations, max_dist |
| `diagnostics` | 기존 `RegistrationDiagnostics` 그대로 |
| `quality_gate` | 통과 여부와 어떤 기준으로 판정했는지 |

규칙:

* **선언 금지.** `SFM_INTERNAL → TLS_GLOBAL` 은 언제나 측정이다. `SourceFrameConfig` 의
  `explicit_identity` 는 이 경로에서 거부된다.
* **support 는 최종 변환을 움직인 모든 기하다.** ① 의 대응만이 아니라 ② 의 ICP target 까지
  포함한다 — ICP 는 target 클라우드를 보고 transform 을 계속 최적화하므로, ICP 에 들어간 TLS 는
  정합 근거이지 무관한 참조가 아니다. `basis` 가 무엇이든 `support_ranges_m` 은 둘의 합집합이고,
  claim gate 는 그 합집합으로 판정한다 (§AD-2).
* **ICP 는 scale 을 바꾸지 않는다** (`rigid_icp.py:58`, `with_scale=False`). 이것은 유지되는 불변식이고
  테스트로 고정한다.
* **미수렴 ICP 는 claim 경로에서 거부**된다. 현재는 `converged=False` 여도 결과가 반환된다.
* **RANSAC 전체 fallback 금지.** `align_correspondences` 는 inlier 가 부족하면 전체 점으로 fit 한다
  (`initial_alignment.py:60`). claim 경로에서는 그 fallback 이 일어났다는 사실이 기록되고 거부된다.
* **`rmse_m` 단독 신뢰 금지.** `diagnose` 의 rmse 는 inlier 에 대해서만 계산된다
  (`diagnostics.py:52`). 게이트는 `inlier_ratio` 와 함께 판정한다.
* **임계값은 이번 C0 에서 정하지 않는다.** 숫자(최소 inlier_ratio, 최대 rmse, 최소 대응 수)는
  **pilot / commissioning 데이터 또는 pre-G2 관측**으로 정하고 **freeze 한 뒤**, claim-bearing G2
  에서는 바꾸지 않는다. 즉 threshold 를 정하는 데이터와 claim 을 만드는 데이터는 같지 않다 —
  결과를 보고 기준을 맞추면 그 기준은 아무것도 거르지 못한다. 그 전까지는 **게이트의 형태만**
  계약이고, 값이 없으면 claim 은 나오지 않는다 (fail closed). 임의 숫자를 지금 발명하지 않는다.

### 7.1 진단이 잡는 것과 잡지 못하는 것

잡는다: 큰 잔차, 낮은 inlier 비율, 뒤집힌 scale, 대응 부족.
**잡지 못한다**: 갱도처럼 축 방향으로 near-degenerate 한 형상에서 축을 따라 미끄러진 정합(국소적으로
좋고 전역적으로 틀린 해), support 가 한 구간에 몰려 있을 때의 외삽 오차, mirrored 해(Umeyama 의 det
보정이 막지만 대응 자체가 잘못된 경우는 아니다). 그래서 `support_ranges_m` 이 진단의 일부다 — 어디를
근거로 맞췄는지 모르면 어디까지 믿을 수 있는지도 말할 수 없다.

---

## 8. 기존 dataset 계약으로의 변환

`minegs dataset from-sfm` (가칭) 이 하는 일:

```
SfmRecord + RegistrationRecord + FrameSetRecord
 → sparse/0      : SfM 모델을 LOCAL_METRIC 으로 옮겨 기록 (카메라·이미지·점)
 → images/       : FrameSet 의 이미지 (hardlink/copy, 기존 규칙)
 → masks/        : 있으면 그대로
 → init_points.ply : **SfM sparse 점**에서 생성 (voxel/max_points 는 기존 설정 재사용)
 → centerline.csv  : 설계 중심선 또는 추출 (기존 규칙)
 → manifest.json   : source=video|video360, initialization.source=sfm_sparse,
                     scale.basis=known_target|sim3_to_tls, registration=<diagnostics>,
                     T_tls_from_local (기존 규칙: SE3, translation-only baseline)
```

규칙:

* dataset schema 를 포크하지 않는다. 기존 `dataset validate` 와 `eval protocol` 은 그대로 돈다.
* **Golden Gate 는 그대로 돌지 않는다.** `dataset/golden_gate.py:118` 은 `tls_station` capture group 이
  없으면 거부한다 — 그 게이트는 "각 station 의 자기 스캔을 그 station 의 이미지에 투영해 한 공간임을
  보인다" 는 TLS 전용 증거다. image-only dataset 에는 station 스캔이 없으므로 이 게이트는 **적용
  불가이며, 통과했다고 말해서도 안 된다**. 대신 Phase 3B 가 같은 질문("카메라와 기하가 한 물리 공간에
  있는가")에 답하는 image-only 판을 만든다: (i) SfM 자체의 reprojection 증거, (ii) **등록된** TLS
  reference 를 SfM 카메라에 투영한 overlay — 이것은 TLS 를 *증거*로 쓰는 것이지 initialization 으로
  쓰는 것이 아니다 — 그리고 (iii) `real_data_validation_status` 는 여전히
  `pending_human_inspection` 이다. 자동 점수가 사람의 확인을 대신하지 않는다.
* **`init_points.ply` 는 TLS 에서 오지 않는다.** 그리고 그것은 선언이 아니라 확인이다: manifest 는
  init 클라우드가 파생된 `sfm_id` 를 기록하고, 빌더와 검증기는 그 결속을 다시 확인한다.
* `train/staging.py:111` 은 `init_points.ply` 로 `points3D` 를 대체하고 track 을 비운다. image-only
  경로에서는 SfM track 을 보존하는 편이 나을 수 있다 (Phase 4 `depth_loss` 와도 얽힌다). **이번
  C0 에서는 결정하지 않는다** — §16 의 열린 질문이다.
* capture group 어휘도 이미 있다: `core/manifest.py:30 GroupType` 은 `tls_station` 외에
  `trajectory_segment` / `camera_rig` / `mobile_mapping_segment` 를 갖는다. 일반 영상은
  `trajectory_segment`, 360 ring 은 `camera_rig` 다 — 새 타입을 만들지 않는다.
* `T_tls_from_local` 은 여전히 SE(3) 다. scale 은 registration 에서 이미 끝났다.

---

## 9. Provenance chain

```
video/이미지 (digest)
  → FrameSetRecord      (추출 설정, 선별 결정, crop 결속, mask)
  → SfmRecord           (backend+버전+명령, 이미지 집합, 모델 digest, SFM_INTERNAL/arbitrary_scale)
  → RegistrationRecord  (basis, support+구간, 측정된 Sim3, ICP, diagnostics, gate)
  → dataset             (dataset_id/hash, source=video*, init=sfm_sparse+sfm_id, registration)
  → run → depth → surface → sections → volume     (Phase 0D/1A/1B/1C 그대로)
  → Phase 3 비교 report
```

각 단계는 **앞 단계의 identity 를 다시 읽어 확인**한다. Phase 2 의 규칙을 그대로 따른다: 주장은
읽는 것이 아니라 다시 도출하는 것이고, fingerprint 는 원장에서 복사하지 않는다.

---

## 10. Claim 경계 요약

| 상황 | 숫자 | claim |
|---|---|---|
| SfM 만 있고 registration 없음 | 없음 (metric 아님) | 없음 |
| registration 있으나 품질 게이트 미달 | diagnostic | 없음 |
| `sim3_to_tls`, support ∩ holdout ≠ ∅ | diagnostic | **없음** |
| `sim3_to_tls`, support ∩ holdout = ∅, 게이트 통과 | claim 가능 | `geometry_accuracy`, `volume_accuracy` |
| `known_target`, TLS support 없음, AD-2 leakage gate + 품질 게이트 통과 | claim 가능 | `geometry_accuracy`, `volume_accuracy` |
| `known_target`, 선언된 TLS 부분집합 ICP, support ∩ holdout = ∅, 두 게이트 통과 | claim 가능 | `geometry_accuracy`, `volume_accuracy` |
| `known_target` + 전체 TLS ICP | diagnostic | **없음** (basis 는 pose 의 출처를 말하지 않는다) |
| 위 모두 + 검증된 surface 사슬 없음 | diagnostic | 없음 |
| 합성 데이터 / 대체 SfM | 구조 검증 | **어떤 과학적 claim 도 없음** |

---

## 11. Structural test 전략

CI 에는 COLMAP 도 ffmpeg 도 GPU 도 없다. Phase 2 가 trainer/renderer 를 대체하고 **그 사실을
기록했던** 방식을 그대로 쓴다.

**Phase 3 가 새로 추가하는 substitution seam 은 SfM 하나다.** 그러나 3C 의 전 구간 게이트는
image-only dataset → 학습 → depth → surface → geometry/section/volume 까지 도는 것이므로, **Phase 2
의 trainer·renderer 대체를 그대로 함께 쓴다.** ffmpeg 가 없으면 프레임 추출도 대체된다. 즉 3C 에서
동시에 대체되는 것은 하나가 아니라 셋 또는 넷이고, 문서가 "SfM 하나뿐" 이라고 말하면 그 게이트가
무엇을 증명하는지 잘못 말하는 것이 된다.

| seam | 언제 | 기록 |
|---|---|---|
| SfM (신규, Phase 3) | CI 에 COLMAP 없음 | `real_sfm_execution: false` — `SfmRecord` 와 report |
| trainer (Phase 2) | CI 에 GPU·gsplat 없음 | `real_gpu_execution: false` |
| renderer (Phase 2) | CI 에 CUDA 없음 | `real_renderer_execution: false` |
| frame extraction | CI 에 ffmpeg 없음 → 파일 복사 | 추출 설정과 대체 사실을 `FrameSetRecord` 에 |

규칙:

* 모든 seam 은 **Python 에서만** 주입된다. CLI 에는 어떤 seam 도 여는 플래그를 만들지 않는다.
* 모든 대체 사실은 artifact 와 report 에 남고, 하나라도 대체되었으면 그 실행은 **어떤 과학적
  claim 도 만들지 않는다**. 구조 검증이지 G2 가 아니다.
* 그 외는 실제 코드로 돈다: 실제 crop 리샘플, 실제 선별 지표, 실제 rig 구성, 실제 COLMAP 모델 IO,
  실제 Umeyama/ICP/diagnostics, 실제 dataset 빌더·validator·protocol judge, 실제 Phase 1A/1B/1C 사슬.
* 합성 입력은 기존 `core/synthetic.py` 의 터널과 렌더 경로에서 만든다 — 파노라마를 만들고, 거기서
  ring crop 을 잘라 "360 영상 프레임" 으로 쓰면 crop→rig→SfM 입력 경로 전체가 실제 코드로 검증된다.

최소 구조 테스트 (Phase 3C 에서 확정):

| # | 무엇을 고정하는가 |
|---|---|
| S1 | `SFM_INTERNAL` 클라우드는 metric 소비자에 들어가지 못한다 |
| S2 | registration 없는 image-only dataset 은 metric claim 을 받지 못한다 |
| S3 | support ∩ holdout ≠ ∅ 이면 claim 이 거부된다 (숫자는 diagnostic 으로 남는다) |
| S3b | `known_target` 이어도 전체 TLS ICP 를 거치면 claim 이 거부된다 (support 는 ①∪②) |
| S4 | `init_points.ply` 가 SfM 에서 오지 않으면 image-only 빌드가 거부된다 |
| S5 | crop 의 orientation 기록과 실제 crop 이 어긋나면 거부된다 |
| S6 | ICP 가 scale 을 바꾸지 않는다 |
| S7 | 미수렴 ICP / RANSAC 전체 fallback 은 claim 경로에서 거부된다 |
| S8 | sparse component 가 여럿이면 명시적 선택 없이는 거부된다 |
| S9 | mask 이름이 이미지와 어긋나면 조용히 무시되지 않고 거부된다 |
| S10 | TLS-assisted 와 image-only 비교가 같은 grid·공통 구간에서만 이뤄진다 |
| S11 | 대체된 seam 이 하나라도 있으면 report 가 그것을 전부 드러내고 claim 이 나오지 않는다 |

---

## 12. Real G2 정의

Phase 3 G2 는 **실측 데이터로만** 성립한다.

필요한 것: 실제 갱도의 영상/360 영상, 실제 COLMAP 실행, 실제 GPU 학습과 렌더, 실제 TLS reference,
그리고 사람이 눈으로 본 결과.

G2 판정 항목:

1. image-only 재구성이 TLS reference 에 등록되고, registration 품질이 기록된다.
2. 등록된 image-only GS 의 `geometry_accuracy` / `volume_accuracy` 가 leakage 없이 산출된다.
3. **같은 평가 조건**(같은 holdout, 같은 section 파라미터, 공통 적분 구간)에서 TLS-assisted GS 와
   image-only GS 를 비교한 수치가 하나의 report 에 남는다.
4. `human_visual_review_status` 는 사람이 본 뒤에만 바뀐다. minegs 는 절대 `pass` 를 쓰지 않는다.

그 전까지의 표현은 고정이다:

> Phase 3 image/360 independent reconstruction path is <구현 상태>.
> Real-data scientific validation remains NOT VALIDATED.
> Phase 3 G2 remains PENDING.

합성 SfM 결과를 G2 라고 부르지 않는다.

---

## 13. Fail-closed 목록

| 상황 | 동작 |
|---|---|
| ffmpeg / COLMAP 없음 | `MissingDependencyError` (기존 동작 유지) |
| SfM 이 모델을 못 냄 / 등록 이미지가 기준 미만 | 실패 |
| sparse component 가 여럿 | 명시적 선택이 없으면 거부 |
| rig_config 를 지정했는데 파일이 없음/깨짐 | 거부 |
| mask 디렉토리를 줬는데 이미지 이름과 대응하지 않음 | 거부 (조용한 무시 금지) |
| crop 기록과 실제 crop digest 불일치 | 거부 |
| `SFM_INTERNAL` 클라우드가 metric 소비자에 도달 | 거부 |
| `SourceFrameConfig` 로 SfM 출력을 TLS_GLOBAL 로 선언 | 거부 |
| registration 없음 / 품질 지표 없음 | metric claim 거부 (기존 `protocol.py:79`) |
| registration 게이트 미달 | claim 거부, diagnostic 유지 |
| support 구간 미기록 | claim 거부 (겹침을 판정할 수 없으므로) |
| ICP 를 돌렸는데 `icp_support` 미기록 | claim 거부 |
| 전체 TLS 클라우드로 ICP (basis 무관) | claim 거부, diagnostic 유지 |
| support ∩ holdout ≠ ∅ | claim 거부 |
| RANSAC 전체 fallback 발생 | claim 경로에서 거부 |
| ICP 미수렴 | claim 경로에서 거부 |
| `init_points` 가 `sfm_id` 와 결속되지 않음 | image-only 빌드 거부 |
| image-only dataset 에 TLS Golden Gate 를 통과했다고 기록 | 거부 (해당 게이트는 적용 불가) |
| image-only dataset 이 `source="tls"` 로 기록됨 | 거부 (video 게이트 우회, AD-2.1 §1) |
| `init.source="sfm_sparse"` 인데 holdout 구간이 init 제외 목록에 없음 | 거부 (`protocol.py:109` 의 구멍) |
| 정합 기준이 `init_points.ply` 를 가리킴 | 거부 (평가측 산출물이어야 한다) |
| `initialization.file` 이 dataset hash 밖을 가리킴 | 거부 |
| `sfm_sparse` 선언과 실제 init 클라우드의 출처 불일치 | 거부 |
| 평가 기준 클라우드의 identity 미기록 | claim 거부 (겹침을 판정할 수 없다) |
| 360 crop 의 pano convention 이 보정 없이 기본값 | 거부 (측정하지 않은 기하 구속) |
| SfM 카메라 intrinsics 가 TLS 측정값으로 고정됨 | 기록 없으면 거부 |
| TLS station pose 가 image-only 재구성의 초기 포즈로 들어감 | 거부 |
| `scale.basis` 없음 | metric claim 거부 (기존 동작) |
| 대체 SfM 으로 만든 결과 | 구조 검증 전용, claim 없음 |

---

## 14. PR 분할

### Phase 3A — Image/360 input + SfM evidence boundary

* `minegs ingest video` 정리: 360 crop 생성 경로 신설(기존 `crop_equirect` 재사용), mask 생산 경로
  연결(`write_mask` 를 실제로 부르는 길), 선별 결정의 identity.
* `FrameSetRecord`, `SfmRecord` 도입. `sparse/0` 하드코딩과 다중 component 처리.
* 눈에 보이는 fail-open 정리: `--mapper` 값 검증, dead field `SfMOptions.mapper`, 비재귀 glob,
  `--output_path` 디렉토리 생성, vocab tree 없는 `loop_detection`, crop 레이아웃과
  `single_camera_per_folder` 의 모순.
* `SFM_INTERNAL` frame 도입과 fail-closed 경계.
* **metric 주장 없음.** 이 PR 이 끝나도 image-only dataset 은 만들 수 없다.

### Phase 3B — Metric registration + dataset materialization

* **`minegs eval register` 의 `SE3 @ Sim3` 오류를 먼저 고친다.** 이 명령은 지금 한 번도 성공한 적이
  없으므로, 3B 의 첫 커밋은 버그 수정과 그 버그를 고정하는 테스트다.
* `diagnose` 에 실제 테스트를 붙인다 — 상수 stub 이 통과하지 못하도록.
* 측정된 Sim3 를 **실제로 적용**하는 지점을 하나로 정한다: dataset materialization 에서 한 번
  (`SFM_INTERNAL` → `TLS_GLOBAL` → `LOCAL_METRIC`). 평가 시점에 변환하지 않는다.
* `RegistrationRecord`: support·구간·gate·측정된 Sim3 를 artifact 로.
* `align_correspondences` fallback / 미수렴 ICP / rmse 단독 판정의 fail-closed 처리.
* leakage 규칙(AD-2)과 `protocol.py` 확장.
* `minegs dataset from-sfm`: 기존 dataset 계약으로 변환, `init_points` 는 SfM 에서.
* image-only 판 Golden Gate (TLS station overlay 가 아니라 SfM reprojection + 등록된 TLS overlay).

### Phase 3C — Independent reconstruction structural gate + G2 비교

* image-only dataset → 학습 → depth → surface → geometry/section/volume 구조 게이트.
  신규 대체는 SfM 하나지만, 이 구간은 Phase 2 의 trainer·renderer 대체 위에서 돈다 (§11 표).
* TLS-assisted vs image-only 를 공통 구간에서 비교하는 report.
* real-data G2 runbook.

**왜 이 분할인가.** 3A 와 3B 의 경계는 "metric 주장이 시작되는 지점" 이다. 3A 가 끝나도 아직
`SFM_INTERNAL` 이고, 그래서 3A 를 잘못 만들어도 틀린 과학적 숫자가 나올 수 없다. 3B 는 그 경계를
넘는 유일한 PR 이고 leakage 결정 전부가 여기에 모인다. 3C 는 실행 증거와 비교이며, Phase 2 가
증명했듯이 구조 게이트는 앞 단계가 고정된 뒤에 붙여야 한다.

reality audit 결과 3A 를 더 쪼갤 이유는 보이지 않는다 — 360 crop·mask·SfM record 는 모두 `FrameSet`
→ `SfmRecord` 라는 한 계보에 속하고, 따로 내면 중간 상태의 artifact 를 두 번 설계하게 된다. 다만
3B 는 실제 구현 중에 (i) registration artifact + fail-closed 와 (ii) dataset 변환으로 갈라질 수 있다.
그때는 이유를 적고 분리한다.

---

## 15. Definition of Done

### C0 (이 문서)

1. `docs/PHASE3_CONTRACT.md` 가 존재하고 §1–§15 를 담는다.
2. reality audit 이 실제 호출 경로에 근거한다 (file:line).
3. TLS registration/evaluation leakage 결정이 명시되어 있다 (AD-2).
4. SfM frame 결정이 명시되어 있다 (AD-1).
5. ROADMAP 의 Phase 3 서술이 audit 결과와 어긋나지 않는다.
5b. ARCHITECTURE 가 AD-1 과 어긋나지 않는다 — frame 그림, §7 정합, §14 결정 이력.
6. 구현은 시작하지 않는다. 이 문서 + stale docstring 정정 + ROADMAP/ARCHITECTURE 문서 정정이 전부다.
7. PR 을 열되 **merge 하지 않는다**. 독립 검토 후 Phase 3A 지시를 받는다.

### Phase 3 전체 (참고)

* 3A/3B/3C 가 각각 구조적으로 검증되고 CI 가 통과한다.
* image-only dataset 이 기존 dataset 계약을 **포크 없이** 만족한다.
* leakage 규칙이 코드로 강제되고 negative test 가 있다.
* real E57/영상 G2 는 실제 실행 전까지 **PENDING / NOT VALIDATED** 다.

---

## 16. 열린 질문 (C0 에서 닫지 않는다)

1. **image-only staging 의 track 처리.** `train/staging.py:111` 은 `init_points.ply` 로 `points3D` 를
   대체하고 image track 을 비운다. TLS 초기화에서는 옳지만, image-only 에서는 SfM 이 만든 track 이
   곧 그 재구성의 관측 증거다. 보존이 나은지, 보존이 Phase 4 `depth_loss` 설계와 어떻게 맞물리는지는
   실제 학습 결과를 한 번 본 뒤 결정한다.
2. **registration 품질 게이트의 숫자.** 최소 `inlier_ratio`, 최대 `rmse_m`, 최소 대응 수. 임의 숫자를
   지금 발명하지 않는다 — Phase 3 G2 에서 실측을 근거로 고정한다. 그 전까지 값이 없으면 claim 은
   나오지 않는다.
3. **`sim3_to_tls` support 의 최소 조건.** 구간이 분리되어 있기만 하면 되는가, 아니면 support 가
   holdout 을 기하적으로 둘러싸야 하는가(외삽 금지). 갱도의 축방향 near-degeneracy 를 감안하면
   후자가 안전하지만, 실측 배치를 보기 전에는 과도한 제약일 수 있다.
4. **일반 영상의 scale 증거.** 360 이 아닌 단일 카메라 영상에서 `known_target` 을 실제로 확보할 수
   있는지(현장에 측량 타깃이 있는지)는 운영 조건이다. 없다면 그 경로는 `sim3_to_tls` 로만 가능하고,
   holdout 분리가 유일한 방어선이 된다.
5. **COLMAP 버전 호환.** `global_mapper` / `rig_configurator` / `--FeatureExtraction.use_gpu` 등은
   문자열로만 검증되어 있다. 실제 COLMAP 과 처음 만나는 순간은 Phase 3A 이고, 거기서 플래그가 틀리면
   계약이 아니라 명령을 고친다.

---

## 17. 구현 기록 — `phase-3-image360`

§1–§16 은 C0 freeze 그대로 둔다. 이 절은 그 계약에 맞춰 실제로 구현된 것과, **구현 중에 계약이
틀렸거나 비어 있던 곳**을 기록한다. 계약이 판정 기준이므로, 어긋난 곳은 여기서 명시한다.

### 17.1 분할 변경

§14 는 3A/3B/3C 세 PR 로 나눠 두었다. Product Owner 지시로 **하나의 branch·하나의 PR** 로
구현했다. 내부 체크포인트 C1–C4 는 §14 의 경계를 그대로 따르고(C1 = 3A, C2/C3 = 3B, C4 = 3C),
각 체크포인트는 별도 commit 으로 남아 있어 순서대로 읽을 수 있다. 분할의 근거였던 "metric 주장이
시작되는 지점" 은 commit 경계로 보존된다.

### 17.2 만들어진 것

| 계약 | 구현 |
|---|---|
| §5 `FrameSetRecord` | `minegs/ingest/video/models.py`, 빌더 `.../build.py` |
| §6 `SfmRecord` | `minegs/ingest/video/sfm/models.py`, 실행 `.../sfm/run.py` |
| §7 `RegistrationRecord` | `minegs/eval/register/models.py`, 측정 `.../register/run.py` |
| §8 dataset 변환 | `minegs/dataset/from_sfm.py` |
| §7.1 image-only Golden Gate | `minegs/dataset/golden_gate_sfm.py` |
| §14(3C) 구조 게이트 | `minegs/e2e/phase3.py` + `tests/test_phase3_gate.py` |
| §14(3C) 비교 | `minegs/eval/compare/paths.py` |

CLI: `minegs ingest video frames|frameset|rig|sfm`, `minegs eval register`,
`minegs dataset from-sfm`, `minegs dataset golden-gate`(source 에 따라 분기),
`minegs eval compare-paths`. `minegs ingest video select` 는 없어졌다 — 선별 결과가 디렉토리
레이아웃이 되었으므로 독립 명령으로 남을 이유가 없다.

Phase 3 workflow 는 Phase 2 의 `Stage` 열거와 ledger 를 **그대로** 쓴다. `INGEST` 가 프레임
집합·재구성·registration 이 되고 `DATASET` 이 image-only 빌더가 될 뿐, `TRAIN`–`REPORT` 는 Phase 2
handler 그대로다. 학습·depth·surface·geometry·sections·volume 의 구현은 하나뿐이고, 두 경로가
따로 흐를 수 없다.

### 17.3 대체된 seam (§11 표의 실제 값)

| seam | 이 저장소에서 | 기록되는 곳 |
|---|---|---|
| frame extraction | `copy_extractor` — ffmpeg 없음 | `ExtractionRecord.real_execution: false` |
| SfM | 테스트 stand-in — COLMAP 없음 | `SfmRecord.real_sfm_execution: false` |
| trainer | Phase 2 stand-in — GPU 없음 | `real_gpu_execution: false` |
| renderer | Phase 2 stand-in — CUDA 없음 | `real_renderer_execution: false` |

네 개 모두 Python 에서만 주입된다. CLI 에 여는 플래그는 없다. 넷 다 report 의 maturity note 로
올라오고, `eval compare-paths` 의 `real_execution` 은 **선언된 플래그가 전부 있고 전부 참일 때만**
참이다 — 아무도 보고하지 않은 단계는 "대체되지 않았다" 가 아니라 "모른다" 이고, 모르면 거짓이다.

**REAL COLMAP EXECUTION: NOT PERFORMED.** 이 저장소에서 COLMAP 은 한 번도 실행되지 않았다.
`colmap_incremental` / `colmap_global` 의 argv 는 문자열 수준에서만 검증되어 있다(§16.5).
"SfM validated" 라고 쓸 수 있는 근거는 없다.

### 17.4 §11 의 S1–S11 이 어디에 있는가

| S | 테스트 |
|---|---|
| S1 | T1, T2 |
| S2 | T24 |
| S3 | T18 |
| S3b | T16 |
| S4 | T21 |
| S5 | T9 |
| S6 | `tests/test_eval.py` 의 registration scale 회복 + T-gate `test_registration_recovers_the_scale_nothing_told_it` |
| S7 | T15, T17, T19 |
| S8 | `run_sfm` 의 다중 component 거부 (`tests/test_cli.py`) |
| S9 | T10 |
| S10 | T29, T30 |
| S11 | `test_the_report_says_what_was_substituted_and_claims_nothing`, `test_the_comparison_refuses_to_be_read_as_a_measurement`, T34, T34b |

### 17.5 계약이 비어 있던 곳 — 구현 중 발견

1. **holdout 이 선언만 되고 제외되지 않았다.** §8 은 `init.excluded_chainage_ranges_m` 에 holdout 을
   적게 했지만, image-only 경로에서 init cloud 는 **재구성 그 자체**다. 구간의 점을 남겨두면
   holdout 이 학습에서 빼놓은 프레임으로 삼각측량된 구조를 모델에 그대로 넘겨주면서, manifest 는
   빼놓았다고 말한다. `init_points.ply` 와 `sparse/0` 를 chainage 로 걸러 넣었다. 둘 다 거른 이유는
   `stage_dataset(use_init_points=False)` 가 `sparse/0` 로 초기화할 수 있기 때문이다 — 두 경로 중
   하나만 지키는 제외는 제외가 아니다.
2. **360 crop 의 stem 이 겹쳤다.** `p0y00/v_07.png` 와 `p0y01/v_07.png` 는 Phase 1B 의 depth 이름
   규약(§stem 하나에 map 하나)에서 같은 `v_07.npy` 를 요구한다. 한쪽 렌더가 다른 쪽을 덮어쓰고 두
   view 가 같은 map 을 back-project 했을 것이다. crop 이름이 view 를 품도록 바꿨다. Phase 1B 계약은
   건드리지 않았다 — 고칠 곳은 이름을 만드는 쪽이다.
3. **frame set 이 밀항자를 잡지 못했다.** `check_frameset` 은 record 가 **적은** 이미지의 digest 만
   검사했고, SfM 은 그 목록이 아니라 `images/` 디렉토리를 glob 한다. 나중에 복사해 넣은 파일은 모든
   digest 가 맞은 채로 재구성에 들어갔다. 이제 디렉토리를 열거해 record 에 없는 이미지를 거부한다.
   §5 가 "선별이 레이아웃의 성질" 이라고 말한 것을 실제로 성질로 만든 것은 이 검사다.
4. **image-only gate 의 가시성 기준이 틀렸다.** 카메라별 하한은 긴 갱도의 올바른 재구성을 떨어뜨린다
   — 갱도에서 모든 view 는 자기 주변만 보고, 갱구의 view 는 바깥의 어둠을 본다. 잡아야 할 실패는
   "**모든** 카메라가 아무것도 못 본다" 이므로, 중앙값과 blind 비율로 판정하고 분포는 어느 쪽이든
   report 에 남긴다.

### 17.5b 독립 검토(round 1) 가 찾은 claim boundary 구멍 4건

§17.5 의 4건은 구현 중에 찾은 것이고, 아래 4건은 **첫 독립 검토**가 찾았다. 전부 "기능이 안
돌아간다" 가 아니라 **claim/provenance 를 우회할 수 있는 경로**였다. 공통 원인은 하나다 —
*결론을 다시 도출하지 않고 읽었다*.

1. **registration 의 claim 을 재도출하지 않았다.** `decide_claim()` 은 생성 시 한 번 실행되고,
   `claim_allowed` / `claim_refusals` / `support_ranges_m` 는 그 결론이다. `check_registration()`
   은 그 결론을 되읽기만 했고, 품질 게이트의 판정은 `quality_gate.json` 이라는 **record 밖 파일**
   에 있어서 재도출할 재료조차 없었다. → `QualityGate` 를 `RegistrationRecord` 안으로 옮기고,
   `check_registration()` 이 `evaluate_gate()` 와 `decide_claim()` 을 **다시 실행**해 기록된
   결론과 대조한다. 어긋나면 거부.
   덧붙여 **manifest 의 사본**이 record 와 같은지도 검사하지 않았다 — protocol judge 는 record 가
   아니라 manifest 를 읽는다. record digest 가 맞아도 manifest 쪽 `claim_allowed` 만 뒤집으면
   측정이 거부한 claim 이 허용되었다. → `CLAIM_BEARING_FIELDS` 전부를 record 와 대조한다.
2. **"숨은 TLS init 없음" 이 원본 SfM 에 묶여 있지 않았다.** `_check_init_is_sfm_geometry` 는
   `init_points.ply` 를 **같은 dataset 안의** `sparse/0` 와 비교했다. 둘 다 같은 TLS geometry 로
   바꾸면 서로 완벽히 일치하므로 통과했다. 한 주장의 사본 두 개는 그 주장의 증거가 아니다.
   → 선택된 SFM_INTERNAL model 을 `provenance/phase3/sfm_model/` 에 **번들**하고(digest 로 고정),
   거기서 측정된 Sim(3) → local origin → holdout 제외 순서로 **다시 도출**해 dataset 의 두 cloud
   와 **카메라 중심**까지 대조한다. 포즈까지 보는 이유는 점만 보면 view 를 갈아끼울 수 있기 때
   문이고, 이것이 golden gate 를 대체하지는 않는다 — 원본 재구성 자체가 어긋난 경우는 여기서
   충실히 재현되고 **눈으로 보는 쪽**에서 걸린다.
3. **`images_excluded=True` 가 실제 train split 과 연결되지 않았다.** `train_images()` 가 읽기
   시점에 걸러 주기는 하지만 chainage 를 **아는** 그룹만 거르고, 못 재는 그룹은 **남긴다**. 즉
   manifest 는 외삽 시험이라고 기록하면서 holdout 안의 그룹이 학습에 들어갈 수 있었다. CLI 에는
   플래그조차 없어 기본값 `True` 가 자동 적용되었다. → 기본값을 **`False`(복원 시험)** 로 내리고
   CLI 에 노출했으며, `True` 일 때는 빌드 시점에 holdout 과 겹치는 capture group 을
   `train_groups` 에서 **빼고**, chainage 를 못 재는 그룹이 있으면 **거부**한다. 확인할 수 없는
   제외는 제외가 아니다.
4. **`real_execution` 이 누락과 덮어쓰기를 통과시켰다.** 두 경로의 flag 를 하나의 dict 로 합쳐
   판정했기 때문에 (i) 필수 key 존재 여부를 보지 않았고 — flag 하나만 true 여도 참이 되었다 —
   (ii) image 쪽 값이 같은 이름의 TLS 쪽 값을 **덮어썼다**. 실제 G2 에서 TLS trainer 대체 사실이
   사라질 수 있었다. → 경로별 필수 목록(`REQUIRED_EXECUTION`)을 두고 **따로** 판정한다. 없는
   key 는 "모른다" 이고, 모르면 거짓이다.

부수적으로, inlier 가 하나도 없을 때 diagnostics 가 NaN 이 되고 NaN 은 JSON 왕복을 통과하지
못해 record 가 **쓰이고 다시 읽히지 않는** 상태가 되었다. 두 단계 뒤에 "null 은 float 이 아니다"
라는 스키마 오류로 드러나던 것을, 발생 지점에서 그 이유로 거부한다.

T31–T36 이 이 여섯을 고정한다. 여섯 guard 모두 mutation check 로 load-bearing 임을 확인했다 —
각각 무력화하면 담당 테스트가 실제로 깨진다.

### 17.5c 독립 검토(round 2)

1. **capture group 의 chainage 가 한 점이었다.** traverse 경로의 그룹은 연속 프레임의 묶음이라
   갱도의 **구간**을 덮는데, `chainage_m` 하나(멤버 평균)만 기록했다. 그래서 *중앙은 holdout 밖,
   한쪽 끝은 holdout 안* 인 그룹 — straddling group — 이 학습에 남았다. `CaptureGroup` 에는
   `chainage_range_m` 필드가 이미 있었고 채우지 않았을 뿐이다. 이제 멤버들의 chainage 최소·최대를
   기록하고, 제외 판정은 **span ∩ holdout** 으로 한다. 판정 함수는 manifest 의 `train_images()`
   가 쓰는 것과 **같은** `spans_overlap()` 이다 — 두 개를 두면 경계에 걸친 그룹에서 언젠가
   서로 다른 답을 낸다. 360 그룹은 crop 들이 광학 중심을 공유하므로 span 이 한 점이고 비용이 없다.
   T37 이 고정한다.
2. **품질 게이트의 threshold 집합이 검증되지 않았다.** `.get` 으로 읽고 없으면 건너뛰었기 때문에
   `{}` 가 **전부 통과**시켰다 — 대응 3점에 잔차 9.9 m 인 정합이 `passed=True` 로 claim 을 얻었다.
   부분 집합도, 오타 난 key 하나도 마찬가지였다. claim 을 지는 게이트는
   `max_rmse_m`·`min_inlier_ratio`·`min_correspondences` **셋 다** 필요하다 — 각각은 혼자서는
   눈이 멀었다: `rmse_m` 은 inlier 위에서 계산되므로 2% 만 맞춘 fit 도 작은 잔차를 보고하고,
   inlier 비율은 점의 개수를 말하지 않으며, 대응 몇 개로 둘 다 만족시킬 수 있다. 누락·미지의
   key 는 이유와 함께 거부되고, `None`(아직 정하지 않음) 과 `{}`(판정했다고 기록되었으나 아무것도
   보지 않음) 를 구분한다. T38 이 고정한다.

T37·T38 의 guard 4개도 mutation check 로 load-bearing 임을 확인했다.

### 17.5d 실제 COLMAP 과의 첫 접촉 (2026-09-22)

§16.5 가 열어 둔 질문 — "실제 COLMAP 과 처음 만나는 순간 플래그가 틀리면" — 에 대한 부분적
답이다. acceptance 컨테이너에 COLMAP 을 설치해 CLI 를 처음으로 실물 바이너리에 물렸다.
**설치된 것은 3.9.1 이고 이 프로젝트는 ≥ 4.0 을 구동하므로, 실제 재구성은 수행되지 않았다.**
그럼에도 두 건이 드러났고 둘 다 실물 이미지도 사람의 눈도 필요하지 않았다.

1. **실패한 version probe 가 version 으로 기록되고 있었다.** `_cli_version()` 은 `check=False`
   로 실행하고 `stdout or stderr` 를 그대로 기록했다. COLMAP 3.9.1 은 `--version` 을 받지
   않으므로, 그 바이너리를 거친 모든 provenance record 가 재구성을 만든 엔진의 버전 자리에
   ``E... colmap.cc:158] Command `--version` not recognize`` 를 담았다. 조용히. 이제 종료
   코드가 0 이 아니면 아무것도 확정되지 않은 것이고, 확정되지 않은 것은 `None` 이다 — 나중에
   버전으로 읽힐 문장이 아니라. `-h`(COLMAP 이 배너를 찍는 곳) 는 `--version` 이 실패한 뒤에만
   시도한다. 이 필드는 Phase 0–3 의 모든 `ProvenanceRecord.tool_versions` 와
   `SfmRecord.backend_version` 에 들어간다.
2. **버전 게이트가 없었다 — 존재 여부만 확인했다.** 현재 Ubuntu LTS 에서 `apt install colmap`
   이 주는 것은 3.9.1 이고, 그것은 같은 인터페이스의 옛 버전이 아니다. `global_mapper` 도
   `rig_configurator` 도 없고 `FeatureExtraction.*` 를 `SiftExtraction.*` 로 부른다. 그래서
   첫 명령이 ``unrecognised option '--FeatureExtraction.use_gpu'`` 와 로그 경로만 남기고
   죽었다 — 잘못된 COLMAP 이 아니라 이 프로젝트의 버그처럼 읽히는 실패였다. 이제 실행 전에
   거부하고 이유를 말한다. **읽을 수 없는 버전은 거부하지 않는다**: 파싱에 실패했다고 멀쩡한
   4.x 를 막는 것은 잘못된 것을 막는 것과 다르다.

여전히 수행되지 않은 것: 실제 갱도 영상, 실제 COLMAP 재구성, 사람의 육안 확인. 절차는
[PHASE3_ACCEPTANCE.md](PHASE3_ACCEPTANCE.md).

### 17.6 성숙도

> Phase 3 image/360 independent reconstruction path is implemented and structurally tested.
> Real-data scientific validation remains NOT VALIDATED.
> Phase 3 G2 remains PENDING.

§12 의 real G2 조건은 하나도 충족되지 않았다. 합성 터널, 대체 SfM, 대체 학습, 대체 렌더러이고,
image-only gate 의 `real_data_validation_status` 는 `pending_human_inspection` 이다 — 사람이 아직
보지 않았고, 이 저장소의 어떤 코드도 그 값을 다른 것으로 쓰지 않는다(T28).
