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
| ffmpeg frame 추출 | **CLI-wired** (command builder 만 테스트됨) | `cli/ingest.py:428` → `ingest/video/frames.py:33 extract_frames` 가 실제 `subprocess.run`; 테스트는 `tests/test_train_ingest_viz.py:334` 의 argv 문자열 검사뿐 |
| ffprobe | **implemented not wired** | `frames.py:44 probe_command` — 호출자 없음 |
| blur / dHash 중복 선별 | **structurally tested** | `dedup_blur.py:20 blur_score`, `:27 dhash`; `cli/ingest.py:447` 이 `select_frames` 실행 후 결정을 JSON 으로 씀. 지표 자체는 `tests/test_train_ingest_viz.py:329` 가 실제로 계산 |
| 선별 결정의 identity | **없음** | `cli/ingest.py:458` 이 `dataclass.__dict__` 를 그대로 dump. 원본 video digest·fps·추출 설정 없음, record id 없음 |
| equirect → pinhole crop 리샘플 | **structurally tested (E57 경로에서 production-wired)** | `ingest/common/equirect.py:74 crop_equirect`; 실제 사용처는 `dataset/materialize.py:347` (E57 파노라마) |
| **video 360 ring crop 생성 경로** | **없음** | `crop_equirect` 를 부르는 video 경로가 없다. `minegs ingest video` 에 crop 명령이 없음 |
| ring crop 기하 (`RingCropSpec`) | **structurally tested** | `equirect.py:23`, `:60 R_scanner_from_cam`; `core/synthetic.py:143`, `dataset/materialize.py:347` 이 사용 |
| COLMAP rig config 작성 | **CLI-wired + structurally tested** | `cli/ingest.py:467` → `video/rig.py:53 write_rig_config`; `tests/…:313` 이 `cam_from_rig_rotation` 구조를 검사 |
| crop ↔ orientation 결속 | **약함 (계약 부재)** | `rig.py:46` 이 `image_prefix = f"{view.name}/"` — 즉 디렉토리 이름이 회전을 가리킨다. rig_config 안의 회전은 명시적이지만, `p0y03/` 에 실제로 yaw index 3 으로 렌더된 crop 이 들어 있는지는 **아무도 검증하지 않는다** |
| mask 생성 (nadir / box) | **implemented not wired** | `video/masks.py:14, 23, 33` 은 배열을 만들고 `:40 write_mask` 가 파일을 쓰지만 **`write_mask` 호출자가 저장소에 하나도 없다**. `nadir_mask_for_crop` 은 테스트에서만 불린다 (`tests/…:322`) |
| mask → COLMAP 전달 | **CLI-wired** | `sfm/colmap_incremental.py:29` 이 `--ImageReader.mask_path` 를 붙이고 `cli/ingest.py:491 --masks` 로 전달됨. 즉 **소비 경로는 있고 생산 경로가 없다** |
| COLMAP feature/matcher/mapper 명령 구성 | **structurally tested (문자열만)** | `sfm/colmap_incremental.py:8 _common`, `:48`; `sfm/colmap_global.py:15`; 테스트 `tests/…:338` 은 `"global_mapper" in cmd` 수준의 문자열 검사 |
| COLMAP 실행 | **CLI-wired, 테스트 없음** | `sfm/base.py:41 run` 이 `subprocess.run(..., check=True)` 로 실제 실행하고 `sfm.log` 에 명령을 남김. CI 에 COLMAP 이 없어 **한 번도 실행된 적 없음** |
| 다중 sparse model 처리 | **미정의 (fail-open)** | `base.py:58` 과 `colmap_global.py:43` 이 `sparse/0` 을 하드코딩한다. COLMAP 이 분리된 component 를 여러 개 낼 때 조용히 0 번을 집는다 |
| `SfMResult` | **휘발성 (증거 아님)** | `sfm/base.py:29` — `sparse_dir`, `n_registered`, `n_points`, `backend`, `log` 뿐인 in-memory dataclass. 디스크에 남는 것은 COLMAP 산출물과 `sfm.log` 뿐이고, backend 버전·옵션·입력 이미지 집합·모델 digest·frame 의미는 **아무 데도 기록되지 않는다** |
| GLUEMAP | **placeholder only** | `sfm/gluemap.py:14` → `NotYetImplementedError` |
| COLMAP model IO | **structurally tested** | `ingest/common/colmap_io.py`; `tests/test_pointcloud_colmap.py` |
| Umeyama Sim3 | **structurally tested** | `eval/register/sim3.py:11`; `tests/test_eval.py:87` 이 scale 1.02 복원을 검사 |
| RANSAC 대응 정합 | **structurally tested, fail-open 잔존** | `register/initial_alignment.py:35`; inlier 가 3 개 미만이면 `:60` 에서 **전체 점으로 되돌아가 fit** 한다 (조용한 degrade) |
| SE(3) ICP | **structurally tested** | `register/rigid_icp.py:24`; `umeyama(..., with_scale=False)` (`:58`) 이므로 **scale 을 바꾸지 않는다**. 다만 `converged=False` 여도 결과를 반환한다 (호출자가 판정해야 함) |
| registration diagnostics | **structurally tested** | `register/diagnostics.py:38 diagnose`; `rmse_m` 는 **inlier 에 대해서만** 계산된다 (`:52`) — inlier_ratio 를 함께 보지 않으면 의미가 없다 |
| registration → manifest | **수작업** | `cli/eval_cmd.py:152` 의 도움말이 그대로 말한다: *"registration JSON (paste into manifest.registration)"*. SfM 모델·source cloud·TLS 파일과의 결속이 없다 |
| target 없는 registration | **fail-open** | `cli/eval_cmd.py:174` — 타깃이 없으면 노란 경고를 찍고 **`Sim3.identity()` 로 진행**한다. 즉 "이미 metric 이라고 가정" |
| registration support 기록 | **없음** | `diagnose` 는 `n_source` 만 남긴다. **어느 TLS 기하를 정합 근거로 썼는지 남지 않는다** |
| video source 의 claim 차단 | **structurally tested** | `eval/protocol.py:75-83` — `source in ("video","video360")` 인데 `registration` 이 없거나 품질 지표가 0 이면 metric claim 거부 |
| image-only dataset 빌더 | **없음** | 유일한 실빌더는 `cli/dataset.py:258 from-e57`. `dataset/materialize.py:755` 이 `initialization.source="tls"` 를, `:768` 이 `source="tls"` 를 **하드코딩**한다 |
| SfM frame 의 표현 | **없음** | `core/frames.py:23 Frame` 은 `SOURCE / TLS_GLOBAL / LOCAL_METRIC / BACKEND_INTERNAL` 뿐. ARCHITECTURE §3 은 SOURCE 를 "스캐너 로컬 / SfM 임의" 로 설명하지만, SOURCE → TLS_GLOBAL 의 문 (`dataset/build_config.py:30 SourceFrameConfig`) 은 `se3()` (`:60`) 를 돌려준다 — **구조적으로 scale 을 담을 수 없다** |
| manifest 의 registration/scale 어휘 | **이미 존재** | `core/manifest.py:138 Registration`, `:32 DatasetSource = "tls"\|"video"\|"video360"`, `ScaleBasis = "tls_pose"\|"sim3_to_tls"\|"known_target"`, `InitSource = "tls"\|"sfm_sparse"\|"random"` |

### 2.1 재사용할 수 있는 것

* **frame 선별 지표**: `blur_score` / `dhash` / `hamming` (`dedup_blur.py`) — 실제 로직, 테스트 있음.
* **ring crop 기하 전체**: `RingCropSpec` / `CropView.R_scanner_from_cam` / `crop_equirect`
  (`equirect.py`) — E57 파노라마 경로에서 이미 production 으로 쓰인다. 360 영상은 **같은 코드를
  다른 입력에 연결**하는 문제이지 새 구현이 아니다.
* **COLMAP rig**: `rig_from_ring` / `rig_config_json` (`rig.py`).
* **COLMAP 명령 구성과 실행 골격**: `sfm/base.py`, `colmap_incremental.py`, `colmap_global.py`.
  mask 전달(`--ImageReader.mask_path`)과 rig 구성(`rig_configurator`)이 이미 들어 있다.
* **정합 수학 전부**: `umeyama`, `align_correspondences`, `icp_point_to_point`, `diagnose` —
  실제 로직이고 `tests/test_eval.py:87,97` 이 합성 데이터로 검증한다. Phase 3 는 여기에 **게이트와
  provenance 를 씌우는 것**이지 다시 구현하는 것이 아니다.
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
  는 **문자열로만** 검증되어 있다. 실제 COLMAP 과의 호환성은 Phase 3 에서 처음 확인된다.
* video → 360 crop 경로 자체가 없다.
* image-only dataset 빌더가 없다.
* registration 결과를 manifest 로 옮기는 자동 경로가 없다 (복사/붙여넣기).
* `sparse/0` 하드코딩, RANSAC 전체 fallback, target 없는 identity Sim3, 미수렴 ICP 반환 — 모두
  **조용한 degrade** 이고 Phase 3 에서 fail-closed 로 바꿔야 한다.

### 2.3 ROADMAP 정정

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
* ARCHITECTURE §3 의 프레임 그림은 이 값이 코드로 들어오는 Phase 3A 에서 함께 고친다. 이번 C0 는
  계약만 고정하고 불변식 문서를 미리 바꾸지 않는다.

### AD-2 — TLS registration/evaluation leakage (BLOCKING, 이번 C0 의 핵심 결정)

image-only SfM 은 metric scale 이 없다. TLS 로 Sim3 를 fit 하면 scale 과 정렬을 얻지만, **같은 TLS
점으로 정렬을 맞추고 그 점에 정확도를 재면 평가가 오염된다.**

Phase 3 는 **두 모델을 모두 허용하되, 어느 쪽인지 artifact 가 말하게 한다.** 이 구분은 새 어휘가
아니라 이미 있는 `Scale.basis` (`core/manifest.py:133`) 로 표현한다.

| scale basis | 정합 근거 | TLS holdout | 허용되는 주장 |
|---|---|---|---|
| **`known_target`** (선호) | 측량 타깃·기준선 등 **TLS 가 아닌 독립 metric 증거** | 전체를 평가에 쓸 수 있다 | `geometry_accuracy`, `volume_accuracy` 가능 |
| **`sim3_to_tls`** (허용, 조건부) | **선언된 registration support 집합**의 TLS 기하 | support 와 **겹치지 않는** 구간만 | support ∩ holdout = ∅ 일 때만 claim, 아니면 거부 |
| 그 외 (타깃 없음, 전체 클라우드 ICP) | 평가 기준 클라우드 자체 | — | **diagnostic 전용**, 영구히 claim 불가 |

강제 조건:

1. **정합에 무엇을 썼는지 남는다.** `RegistrationRecord` (§7) 가 support 의 종류(타깃 CSV / TLS
   부분집합), 파일 digest, **chainage 구간**, 대응 개수, inlier, residual 을 기록한다.
2. **겹침을 판정할 수 있다.** support 구간과 `split.geometry_holdout.chainage_ranges_m` 의 교집합을
   계산할 수 있어야 한다. 계산할 수 없으면 (support 구간 미기록) 그것 자체가 거부 사유다.
3. **겹치면 fail closed.** 겹친 상태로 `geometry_accuracy` / `volume_accuracy` 를 내지 않는다.
   숫자를 죽이는 것이 아니라 **claim 을 죽인다** — 같은 숫자를 `geometry_diagnostic` 으로는 낼 수 있다.
4. **diagnostic 정렬과 claim 정렬을 구분한다.** 전체 클라우드 ICP 는 "그림이 맞는지 보는" 용도로
   유효하고, 그 결과로 만든 숫자는 영구히 diagnostic 이다.
5. **비교는 같은 평가 영역에서 한다.** TLS-assisted GS 와 image-only GS 를 비교할 때 holdout,
   section 파라미터, 적분 구간이 같아야 한다 — Phase 1C/2 의 `eval/volume/paired.py` 규칙(같은 grid,
   공통 구간만)을 그대로 쓴다.

이 결정은 Phase 3B 에서 코드가 된다. 새 알고리즘은 없다: `diagnose` 가 이미 내는 숫자에 **support
기록과 겹침 판정**을 붙이고, `eval/protocol.py:75` 의 video 규칙을 확장하는 일이다.

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
3. `scale.basis` 가 `known_target` 이거나, `sim3_to_tls` 이면서 support ∩ holdout = ∅ 이다.
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
| `support` | 타깃 CSV digest **또는** TLS 파일 digest + 사용한 부분집합의 정의 |
| `support_ranges_m` | support 가 덮는 chainage 구간 (겹침 판정의 근거) |
| `T_tls_from_sfm` | 측정된 Sim3 (scale 포함) |
| `icp` | 사용 여부, `converged`, iterations, max_dist |
| `diagnostics` | 기존 `RegistrationDiagnostics` 그대로 |
| `quality_gate` | 통과 여부와 어떤 기준으로 판정했는지 |

규칙:

* **선언 금지.** `SFM_INTERNAL → TLS_GLOBAL` 은 언제나 측정이다. `SourceFrameConfig` 의
  `explicit_identity` 는 이 경로에서 거부된다.
* **ICP 는 scale 을 바꾸지 않는다** (`rigid_icp.py:58`, `with_scale=False`). 이것은 유지되는 불변식이고
  테스트로 고정한다.
* **미수렴 ICP 는 claim 경로에서 거부**된다. 현재는 `converged=False` 여도 결과가 반환된다.
* **RANSAC 전체 fallback 금지.** `align_correspondences` 는 inlier 가 부족하면 전체 점으로 fit 한다
  (`initial_alignment.py:60`). claim 경로에서는 그 fallback 이 일어났다는 사실이 기록되고 거부된다.
* **`rmse_m` 단독 신뢰 금지.** `diagnose` 의 rmse 는 inlier 에 대해서만 계산된다
  (`diagnostics.py:52`). 게이트는 `inlier_ratio` 와 함께 판정한다.
* **임계값은 이번 C0 에서 정하지 않는다.** 숫자(최소 inlier_ratio, 최대 rmse, 최소 대응 수)는 실측
  데이터를 한 번 본 뒤 Phase 3 G2 결정으로 고정한다. 그 전까지는 **게이트의 형태만** 계약이고, 값이
  없으면 claim 은 나오지 않는다 (fail closed). 임의 숫자를 지금 발명하지 않는다.

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
| `known_target`, 게이트 통과 | claim 가능 | `geometry_accuracy`, `volume_accuracy` |
| 위 모두 + 검증된 surface 사슬 없음 | diagnostic | 없음 |
| 합성 데이터 / 대체 SfM | 구조 검증 | **어떤 과학적 claim 도 없음** |

---

## 11. Structural test 전략

CI 에는 COLMAP 도 ffmpeg 도 GPU 도 없다. Phase 2 가 trainer/renderer 를 대체하고 **그 사실을
기록했던** 방식을 그대로 쓴다.

* **대체되는 것은 SfM 하나뿐**이다. `sfm=` seam 으로 주입하고 `real_sfm_execution: false` 를
  `SfmRecord` 와 report 에 남긴다. CLI 에는 이 seam 을 여는 플래그를 만들지 않는다.
* 그 외는 실제 코드로 돈다: 실제 crop 리샘플, 실제 선별 지표, 실제 rig 구성, 실제 COLMAP 모델 IO,
  실제 Umeyama/ICP/diagnostics, 실제 dataset 빌더·validator·protocol judge·golden gate, 실제 Phase
  1A/1B/1C 사슬.
* 합성 입력은 기존 `core/synthetic.py` 의 터널과 렌더 경로에서 만든다 — 파노라마를 만들고, 거기서
  ring crop 을 잘라 "360 영상 프레임" 으로 쓰면 crop→rig→SfM 입력 경로 전체가 실제 코드로 검증된다.
* ffmpeg 가 없는 CI 에서는 프레임 추출을 파일 복사로 대체하되, **추출 설정과 대체 사실을 기록**한다.

최소 구조 테스트 (Phase 3C 에서 확정):

| # | 무엇을 고정하는가 |
|---|---|
| S1 | `SFM_INTERNAL` 클라우드는 metric 소비자에 들어가지 못한다 |
| S2 | registration 없는 image-only dataset 은 metric claim 을 받지 못한다 |
| S3 | support ∩ holdout ≠ ∅ 이면 claim 이 거부된다 (숫자는 diagnostic 으로 남는다) |
| S4 | `init_points.ply` 가 SfM 에서 오지 않으면 image-only 빌드가 거부된다 |
| S5 | crop 의 orientation 기록과 실제 crop 이 어긋나면 거부된다 |
| S6 | ICP 가 scale 을 바꾸지 않는다 |
| S7 | 미수렴 ICP / RANSAC 전체 fallback 은 claim 경로에서 거부된다 |
| S8 | sparse component 가 여럿이면 명시적 선택 없이는 거부된다 |
| S9 | mask 이름이 이미지와 어긋나면 조용히 무시되지 않고 거부된다 |
| S10 | TLS-assisted 와 image-only 비교가 같은 grid·공통 구간에서만 이뤄진다 |

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
| support ∩ holdout ≠ ∅ | claim 거부 |
| RANSAC 전체 fallback 발생 | claim 경로에서 거부 |
| ICP 미수렴 | claim 경로에서 거부 |
| `init_points` 가 `sfm_id` 와 결속되지 않음 | image-only 빌드 거부 |
| image-only dataset 에 TLS Golden Gate 를 통과했다고 기록 | 거부 (해당 게이트는 적용 불가) |
| `scale.basis` 없음 | metric claim 거부 (기존 동작) |
| 대체 SfM 으로 만든 결과 | 구조 검증 전용, claim 없음 |

---

## 14. PR 분할

### Phase 3A — Image/360 input + SfM evidence boundary

* `minegs ingest video` 정리: 360 crop 생성 경로 신설(기존 `crop_equirect` 재사용), mask 생산 경로
  연결(`write_mask` 를 실제로 부르는 길), 선별 결정의 identity.
* `FrameSetRecord`, `SfmRecord` 도입. `sparse/0` 하드코딩과 다중 component 처리.
* `SFM_INTERNAL` frame 도입과 fail-closed 경계.
* **metric 주장 없음.** 이 PR 이 끝나도 image-only dataset 은 만들 수 없다.

### Phase 3B — Metric registration + dataset materialization

* `RegistrationRecord`: support·구간·gate·측정된 Sim3 를 artifact 로.
* `align_correspondences` fallback / 미수렴 ICP / rmse 단독 판정의 fail-closed 처리.
* leakage 규칙(AD-2)과 `protocol.py` 확장.
* `minegs dataset from-sfm`: 기존 dataset 계약으로 변환, `init_points` 는 SfM 에서.
* image-only 판 Golden Gate (TLS station overlay 가 아니라 SfM reprojection + 등록된 TLS overlay).

### Phase 3C — Independent reconstruction structural gate + G2 비교

* image-only dataset → 학습 → depth → surface → geometry/section/volume 구조 게이트 (SfM 대체 seam).
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
6. 구현은 시작하지 않는다. 이 문서 + stale docstring 정정 + ROADMAP 사실관계 정정이 전부다.
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
