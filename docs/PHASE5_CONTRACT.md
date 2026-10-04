# Phase 5 — Long Tunnel / Chunking 계약 (freeze)

이 문서는 Phase 5 구현 **전에** 고정하는 계약이다. 구현은 이 문서를 판정 기준으로 삼고, 계약이 비어
있던 곳은 §12 구현 기록에 남긴다.

source of truth: 범위와 gate 는 [docs/ROADMAP.md](ROADMAP.md) §Phase 5, 불변식은
[docs/ARCHITECTURE.md](ARCHITECTURE.md). Phase 4 의 학습·depth 계약은
[docs/PHASE4_CONTRACT.md](PHASE4_CONTRACT.md).

## 0. 시작 기록 (2026-10-04)

| 항목 | 값 |
|---|---|
| base | `main @ 1d9167cf87ca808563eb36df6ac67111b4653b7b` (PR #16 merge) |
| branch | `phase-5-long-tunnel` |
| 테스트 수 (base) | 828 collected (torch 없이), torch 모듈 15 |

시작 시점의 성숙도 (이 문서가 바꾸지 않는다):

> Phase 4 implementation: COMPLETE · structural validation: PASS · real GPU training: **NOT PERFORMED**
> · real heavy-vs-baseline test: **NOT PERFORMED** · scientific validation: **NOT VALIDATED** ·
> Phase 4 G3: **PENDING** · Phase 3 manual acceptance: **DEFERRED** · Phase 3 G2: **PENDING**

## 1. 목적과 non-goals

**목적.** 하나의 긴 갱도 dataset 을 chainage 를 따라 여러 학습 chunk 로 나눠 처리하되, dataset
identity, `LOCAL_METRIC` 좌표계, 전역 train/test/holdout 계약, depth-supervision leakage 계약,
evidence provenance, geometry/volume 평가를 훼손하지 않는다. 새 재구성 방법이 아니라 기존 학습·평가
위의 얇은 orchestration/evidence 층이다.

**non-goals.** RunPod, distributed/multi-GPU, chunk 의 GPU 분할, multi-epoch, viewer/LOD/엔진 export,
2DGS/PGSR/새 backend, neural surface, TSDF/mesh 재설계, 최적 chunk 크기 자동 선택, VRAM scheduler,
자동 retry/resume, normalized world-space 학습, 새 scientific metric, G2/G3 실데이터 검증, Gaussian
모델 병합 (overlap 평균·dedup·opacity blending·cross-chunk fusion), chunk 간 ICP.

## 2. C0 reality audit (증거 표)

12 개 독립 읽기 (질문당 답 + 반박 재확인) 로 확인했다. 줄 번호는 base 기준.

| Q | 실제 코드 경로 | 발견 | 결정 |
|---|---|---|---|
| Q1 chunk_id 범위 | `core/chunking.py` (`Chunk`, `ChunkPlan`, `plan_chunks`, `assign_groups`); `Manifest.chunks` (`core/manifest.py:228`); CLI `dataset chunks [--write]` (`cli/dataset.py:132-161`), `train run --chunk` (`cli/train.py:104,154`); `RunConfig/RunRecord.chunk_id` (`train/runner/base.py:93,129`); `Runner.prepare` (`:239-249`); `stage_dataset(chunk_id=)` (`train/staging.py:190-198`); synthetic 이 `manifest.chunks` 를 씀 (`core/synthetic.py:233-238`). eval/e2e 에는 chunk 개념 없음 | (a) **결함**: chunk run 은 `chunk.T_tls_from_local` (chunk 중점 원점) 을 `run.json` 에 기록하지만 staging 은 pose·init 을 바꾸지 않으므로 출력은 dataset `LOCAL_METRIC` 에 있다 — 기록된 frame 이 거짓이다 (아무도 읽지 않아 지금은 실패하지 않음). (b) plan 을 manifest 에 쓰면 `manifest.json` 이 `DATASET_HASH_PATTERNS` 에 있으므로 dataset hash 가 바뀐다. (c) `chunk.groups` 는 test·holdout 그룹과 overlap 을 포함하고, staging 이 `train_images()` 와 교집합해서 분리를 지킨다. (d) `range_m` 은 overlap 을 포함한 범위이고 core 개념이 없다 | 기존 배관 (`chunk_id`, `stage_dataset(chunk_id=)`, `--chunk`) 을 확장한다. plan 은 manifest 밖의 별도 artifact (§4). manifest 의 `chunks` 는 legacy 로 읽기만 허용하고 학습에 쓰지 않는다. `chunk.T_tls_from_local` 은 쓰지 않는다 (frame 결함 수정) |
| Q2 group support | `CaptureGroup.span()` (`core/manifest.py:76-81`): `chainage_range_m` 또는 `(chainage_m, chainage_m)` 또는 `None`. TLS station = 점 span (`dataset/materialize.py:622,692`), SfM group = 구성원 카메라 중심의 chainage 범위 (`dataset/from_sfm.py:329-341`). 360 ring = parent panorama 당 한 그룹 (`from_sfm.py:164`), plain video = 연속 `group_size` 프레임 (`:166`). `train_images()` (`manifest.py:280-290`) 는 `images_excluded` 일 때 holdout 과 겹치는 그룹을 통째로 제외 | span 이 `None` 인 train 그룹은 `group_chainage()` 에서 빠져 어떤 chunk 에도 들어가지 않는다 — **조용히 학습에서 사라진다**. holdout 때문에 train 에서 빠진 SfM 그룹도 `capture_groups` 에는 남아 span 을 가진다 | 그룹은 atomic. 선택 = support 와 span 이 겹치는 **train 그룹 전체**, 이미지는 그 뒤 `train_images()` 와 교집합. span 이 없는 train 그룹이 있으면 plan 을 거부 (chunk 를 정할 수 없다). 모든 train 이미지가 최소 한 chunk 에 들어가는지 검증 |
| Q3 init | `stage_dataset` 은 init 전체를 읽고 `MAX_INIT_POINTS = 4_000_000` 으로 결정적 subsample (`staging.py:239-244`, `pointcloud.py:101-105`). `Support.locate` (`train/supervision/support.py:127-142`) 는 dataset centerline 을 `LOCAL_METRIC` 으로 옮겨 chainage 와 located 여부 (반경 ≤ 10 m, 양 끝 밖이 아님) 를 준다. init 은 빌드 시 이미 holdout 이 제거되어 있다 (`materialize.py:243-248`, `from_sfm.py:265-290`) | chunk 에 init 을 제한하는 코드는 없다. 재사용할 위치 결정 규칙은 `Support.locate` 다 | chunk init = `Support.locate` 로 located 이고 chainage 가 support 안인 init 점 → 그 다음 subsample. 위치 불명 점은 어떤 chunk 에도 넣지 않고 개수를 기록 (새 threshold 없음). holdout 은 다시 해석하지 않는다 (init 은 이미 holdout-free). SfM sparse-track init 경로 (`use_init_points=False`) 는 chunk 에서 거부 |
| Q4 depth | adapter 는 `trainset` 의 이미지 이름으로만 샘플을 찾는다 (`trainers/advanced_gs.py:299-317`) — staging 되지 않은 이미지의 샘플은 읽히지 않는다. 세 해시 pin (`staging.py:165-170`, `advanced_gs.py:139-144`, `runner/base.py:793-797`) 이 artifact 바이트 동일성을 요구한다 | 전역 검증 artifact 를 그대로 쓰면 된다. chunk 의 학습 이미지에 샘플이 하나도 없으면 학습이 끝난 뒤에야 FAILED 가 된다. `images_with_samples(names)` (`supervision/depth.py`) 는 호출자가 없다 | **새 child artifact 없음.** 전역 artifact 를 바이트 그대로 stage 하고 adapter 는 chunk 의 최적화 이미지 샘플만 쓴다. depth profile 의 chunk run 은 prepare 단계에서 chunk 이미지에 샘플이 있는지 확인해 없으면 학습 전에 거부. chunk 가 실제로 쓴 샘플 수를 기록 |
| Q5 evaluator | `SectionRecord` 는 출처 하나 (`eval/sections/models.py:146`), 병합/필터 없음. station grid = `np.arange(max(start, s_start), …, interval)` (`core/centerline.py:144-151`). `compare_to_reference(pred, ref, ranges)` (`eval/volume/paired.py:187`) 는 범위 안 station 만 쓰고 양쪽 모두 관측한 연속 구간만 적분. `require_same_grid` 는 축·dataset·frame·간격·두께·bin·station 목록을 비교. surface 렌더는 dataset 의 **모든** view 를 요구하고 빈 view 를 거부 (`eval/surface/render.py:519-523, 595-634`; `surface/models.py:303-315`) | section series 이어붙이기는 단일 출처 claim 경로를 통과할 수 없다. chunk 모델은 chunk 밖 view 에서 빈 depth 가 나와 surface 생성이 거부될 수 있다 | 각 chunk 의 section 은 **전체 축 grid** (start/end 없음) 로 자른다. 합성은 station 소유권 (§5.1) 으로 각 station 을 정확히 한 chunk 에서 가져오는 duck-typed stitched series 로 하고, 기존 `compare_to_reference` 로 평가한다 (volume 수학 재구현 없음). chunk run 의 depth 렌더 view 는 plan 이 정한 chunk view 집합 (support 와 겹치는 모든 그룹의 구성원) 이고, depth manifest 검증은 run 이 chunk run 이면 그 집합을 요구한다 |
| Q6 identity | `DATASET_HASH_PATTERNS` (`runner/base.py:33-44`): manifest, sparse txt, init, images, masks, `centerline.csv`, `provenance/**` | dataset 아래 새 디렉터리 (`chunks/`) 는 hash 밖이다 | plan 은 `<dataset>/chunks/<plan_id>/chunk_plan.json` 에 쓰고 dataset hash, centerline sha256 을 자기 안에 기록한다. dataset/centerline 이 바뀌면 거부 |
| Q7 frame | frame 은 SOURCE/SFM_INTERNAL/TLS_GLOBAL/LOCAL_METRIC/BACKEND_INTERNAL 뿐 (`core/frames.py:23-39`). float32 한계 5 000 m (0.6 mm) (`frames.py:62`), Phase 0C 는 station 을 원점에서 1 000 m 안으로 제한 (`dataset/frames.py:160-166`). gsplat 은 `T_local_from_internal = I` (`train/backends/gsplat.py:510`), `require_metric_outputs` 가 비항등을 거부 | 만들 수 있는 dataset 은 모두 dataset 단위 `LOCAL_METRIC` 하나에 float32 로 담긴다. chunk-local 원점이 필요하다는 증거 없음 | **CHUNK_LOCAL frame 없음.** 모든 chunk 는 dataset `LOCAL_METRIC`, `T_local_from_internal = I`. chunk run 의 `run.json` 은 dataset 의 `T_tls_from_local` 을 기록 |

## 3. Architecture decisions

* **AD-1 ChunkPlan 은 파생 실행 artifact 다.** dataset 이 아니다. `chunk_plan.json` 은 dataset hash
  밖에 있고 dataset id/hash, centerline sha256, 정책, chunk 목록을 기록한다. manifest 를 고치지 않는다.
  `dataset chunks --write` (manifest 에 plan 을 씀) 는 거부한다.
* **AD-2 core 와 support.** core = 그 chunk 가 최종 소유하는 구간, support = core ± overlap 을 축 범위로
  자른 학습 context. overlap 은 ownership 이 아니다.
* **AD-3 사용자가 정한다.** `core_length_m > 0`, `0 ≤ overlap_m < core_length_m`. 최적값이 아니라 실행
  매개변수다.
* **AD-4 하나의 dataset, 하나의 frame, 하나의 split.** chunk 마다 새 dataset id 를 만들지 않는다. chunk
  학습 identity = dataset identity + plan digest + chunk id + profile + staged hash + depth artifact +
  trainer identity.
* **AD-5 기존 경로 재사용.** 새 trainer·staging 구현·scheduler 없음. `train chunks` 는 plan 의 chunk
  순서대로 기존 `train run` 을 부르는 순차 루프이고, 첫 실패에서 멈춘다.
* **AD-6 Gaussian 은 병합하지 않는다.** canonical 표현은 chunk 별 모델의 색인 (`ChunkRunSet`).
* **AD-7 geometry 는 surface 에서.** GS 중심을 geometry 로 쓰지 않는다. chunk 별 검증된 surface → section
  → core-only 합성 → 기존 volume/비교.
* **AD-8 overlap 은 진단이다.** 인접 chunk 의 overlap section 일치도를 숫자로만 보고한다 (threshold·
  pass 필드 없음). chunk 간 일치는 reference 대비 정확도가 아니다.
* **AD-9 scene_scale 은 기록만 한다.** chunk 의 카메라 범위가 줄면 upstream `scene_scale` (depth 항 크기,
  MCMC 보정 `s`) 이 chunk 마다 다르다. 보정하지 않고 기록한다.

## 4. `ChunkPlanRecord` (schema 1.0, `chunk_plan.json`)

이름: manifest 의 legacy `ChunkPlan` 과 구별하려고 `ChunkPlanRecord`.

```
schema_version, plan_id (= "cplan_" + digest[:12]), plan_digest (sha256, 아래 내용의 canonical JSON),
dataset_id, dataset_hash, centerline_file, centerline_sha256,
axis_range_m: [s_start, s_end],
policy: {core_length_m, overlap_m, ownership_rule: "core_half_open_last_closed"},
chunks: [{chunk_id ("K000"…), ordinal, core_range_m, support_range_m,
          capture_groups (train, atomic), images (train ∩ groups),
          actual_image_support_m (선택된 그룹 span 의 합집합 외곽),
          view_groups, views (렌더용: support 와 span 이 겹치는 모든 그룹)}],
unplaced_groups (span 없는 비-train 그룹; 렌더에도 쓰지 않음),
provenance (minegs version, git, created_at — digest 밖)
```

`plan_digest` 는 provenance 를 뺀 내용의 해시다. 같은 dataset·centerline·정책이면 같은 plan 이다.

## 5. 규칙

### 5.1 core 와 ownership

* 경계 `b_k = s_start + k·L` (k = 0 … n−1), `b_n = s_end`, `n = ceil((s_end − s_start)/L)`. 마지막 core 는
  나머지 길이다. 나머지가 overlap 이하이면 직전 core 에 합친다 — 직전 chunk 의 support 가 이미 축 끝까지
  닿으므로 따로 학습할 것이 없고, 이미지 없는 sliver chunk 가 생기지 않는다 (C1 에서 정함).
* chunk i 의 core = `[b_i, b_{i+1})`, 마지막 chunk 만 `[b_{n-1}, s_end]`. chainage `s` 의 소유자는 정확히
  하나다: `owner(s) = searchsorted(b_1..b_{n-1}, s + ε, right)` (ε = 1e-9 m) — 모든 소유 판정은 이 함수 하나.
* support = `[max(s_start, b_i − overlap), min(s_end, b_{i+1} + overlap)]`.

### 5.2 verifier (§12 의 10 항목)

core 가 축 범위 안 · core 사이 gap 없음 · core 중복 없음 · 순서 단조 · support ⊇ core · support 가 정책과
일치 · dataset hash 일치 · centerline hash 일치 · chunk id 고유 · 같은 입력 → 같은 plan (verifier 가 plan
을 다시 만들어 digest 와 내용을 비교). 그룹 선택도 다시 만들어 비교한다.

### 5.3 capture group

* 선택 단위는 그룹. support 와 span 이 겹치면 (`spans_overlap`, 닫힌 구간) 그룹 **전체**.
* 학습 이미지 = 그 그룹의 구성원 ∩ `manifest.train_images()` ∩ `test_images()` 의 여집합. 전역 split 이
  먼저다. test 그룹, holdout 으로 빠진 그룹은 학습에 다시 들어오지 않는다.
* 그룹 span 이 support 밖으로 나가면 `actual_image_support_m` 이 nominal support 보다 넓어진다 —
  기록한다.
* span 이 없는 train 그룹 → plan 거부. 어느 chunk 에도 속하지 않는 train 이미지 → plan 거부. 학습
  이미지가 없는 chunk → plan 거부 (매개변수를 바꾸라는 이유와 함께).

### 5.4 init

`Support.locate` (dataset centerline, `LOCAL_METRIC`) 로 located 이고 chainage 가 support 안인 init 점만
stage 한 뒤 기존 `MAX_INIT_POINTS` subsample. 기록: 전체 / located / 선택 / 위치 불명 수.

### 5.5 depth supervision

전역 검증 artifact 를 바이트 그대로. 검증은 run 마다 전역 dataset 에 대해 (Phase 4 와 동일). chunk 의
staged 이미지 중 샘플이 있는 이미지가 없으면 학습 전에 거부. 기록: chunk 이미지 중 샘플 있는 이미지 수,
그 샘플 수.

### 5.6 frame

chunk run 의 `T_tls_from_local` = dataset 값, `frame_of_outputs = LOCAL_METRIC`,
`T_local_from_internal = I`. 합성은 변환을 추정하지 않는다 (ICP 없음).

## 6. chunk run evidence (`RunRecord` schema 1.3)

`chunk = {plan_id, plan_digest, plan_path, chunk_id, ordinal, core_range_m, support_range_m,
capture_groups, images, actual_image_support_m, init_points_total, init_points_located,
init_points_selected, init_points_unlocated, depth_images_with_samples, depth_samples_for_images}`.
`chunk_id` (기존 필드) 유지. 기존 dataset hash·profile·trainer·staged hash·runtime·supervision·실행
플래그 그대로. 비 chunk run 은 `chunk = null` 이고 명령·staging 의미가 바뀌지 않는다.

## 7. `ChunkRunSet` (schema 1.0, `chunk_run_set.json`)

```
chunk_set_id, plan_id, plan_digest, dataset_id, dataset_hash, profile,
chunks: [{chunk_id, ordinal, run_id, status, surface_id, sections_id, core_range_m, support_range_m,
          model (final model 경로·sha), trainer (scene_scale 등), real_gpu}],
complete, requested_core_length_m, completed_core_length_m, coverage_fraction,
missing_chunks, missing_intervals_m,
stitched: {station_count, owners: [[chunk_id, n_stations]], grid},
evaluation: {holdout (선언된 holdout 위 stitched vs reference), extent (전체 core, diagnostic),
             per_chunk (각 chunk core vs reference, diagnostic)},
seams: [{left, right, overlap_m: [lo, hi], n_paired_sections, median_abs_area_difference_m2,
         p95_abs_area_difference_m2, mean_signed_area_difference_m2}],
real_execution, g3_status: "PENDING", maturity_statement, notes, provenance
```

* 입력: plan, chunk 마다 (run 디렉터리, 그 run 의 surface 에서 전체 축 grid 로 자른 section), 스캔
  reference section (`raw_cloud`, 같은 grid).
* 거부: 다른 dataset · 다른 plan · 다른 profile · 같은 chunk 두 번 · plan 에 없는 chunk · run 의 chunk
  binding 이 plan 과 다름 · non-`LOCAL_METRIC` · `T_local_from_internal ≠ I` · section 이 그 run 의
  surface 가 아님 · grid 불일치 · reference 가 스캔 구름이 아니거나 예측과 같음.
* 누락 chunk 또는 FAILED run: 기본은 거부. `allow_incomplete` 를 명시하면 `complete = false`, 그 core 는
  `missing_intervals_m` 이 되고 0 으로 채우지 않는다.
* stitched series: 각 station 을 `owner(s)` chunk 의 section 에서 가져온다. 중복 chainage·단조성 위반·
  core gap 은 거부. volume 은 기존 `compare_to_reference` 의 적분 (overlap 이 두 번 적분될 수 없다:
  station 이 한 번씩만 있다).
* seam: 인접 chunk i, i+1 의 support 교집합에서 `compare_to_reference(sections_{i+1}, sections_i, [교집합])`
  의 section 통계. 이름이 말하듯 일치도이고 정확도가 아니다.
* real: 모든 chunk run 이 `real_gpu_evidence` 를 만족할 때만. 아니면 structural.

## 8. Fail-closed 목록 (Phase 5 추가)

| 요청 | 결과 |
|---|---|
| `core_length ≤ 0`, `overlap < 0`, `overlap ≥ core_length` | `ContractError` |
| plan 의 gap / 중복 소유 / 비단조 / support ⊉ core / id 중복 / 정책 불일치 | `ContractError` (verifier) |
| plan 생성 뒤 dataset 또는 centerline 변경 | `ContractError` |
| span 없는 train 그룹, chunk 에 속하지 않는 train 이미지, 학습 이미지 없는 chunk | `ContractError` |
| `--chunk` 없이 `--chunk-plan`, plan 없이 `--chunk`, plan 에 없는 chunk | `ContractError` |
| `dataset chunks --write` | `ContractError` (dataset identity 가 바뀐다) |
| chunk staging 에서 SfM sparse-track init | `ContractError` |
| depth profile chunk 인데 chunk 이미지에 샘플 없음 | `ContractError` (학습 전) |
| chunk run 의 depth 렌더가 chunk view 집합과 다름 | `ContractError` |
| ChunkRunSet 의 섞인 dataset/plan/profile, 중복·누락 chunk, FAILED run, 비 metric, grid 불일치, 중복 chainage, core gap | `ContractError` (누락·FAILED 는 `allow_incomplete` 로만 incomplete set) |
| chunk binding 이 다른 두 run 의 `compare-runs` | `ContractError` |

## 9. negative tests (지시서 §36–§40)

plan (core/overlap 값, gap, 중복 소유, dataset 변경, centerline 변경), capture group (360 ring·video
그룹 atomic, test 이미지·holdout 제외 이미지 부활 없음, 결정적 선택), staging (plan 에 없는 chunk, plan 과
dataset hash 불일치, 다른 chunk identity, 선택 이미지 ≠ staged, init 증거 불일치), composition (§8 의 목록),
single-run regression (chunk 없는 `train run` 의 argv·staging·record 불변).

## 10. 합성 구조 fixture

직선 약 300 m 축, core 100 m, overlap 20 m → chunk 3 개 (0–100/0–120, 100–200/80–220, 200–300/180–300).
일부 capture group 이 chunk 경계를 가로지른다. holdout 하나는 chunk core 안, overlap 근처. 성능 결과가
아니라 구조 검증이다.

## 11. claim 경계와 종료 성숙도

허용: long-tunnel chunk planning / chunk-aware staging·training / chunk ownership·composition:
IMPLEMENTED, structural validation: PASS. 금지: long tunnel scientifically validated, seam-free
reconstruction, optimal chunk size determined, geometry accuracy improved, large-scale mine validated,
G2/G3 passed.

```
Phase 5 implementation:             COMPLETE
Structural validation:              PASS
Long-tunnel chunk planning:         IMPLEMENTED
Chunk-aware training path:          IMPLEMENTED
Chunk ownership/composition:        IMPLEMENTED
Overlap seam diagnostics:           IMPLEMENTED
Long-tunnel evaluation path:        IMPLEMENTED
Real GPU chunked training:          NOT PERFORMED
Real long-tunnel mine dataset:      NOT VALIDATED
Optimal chunk size:                 NOT DETERMINED
Scientific long-tunnel validation:  NOT VALIDATED
Phase 5 G3:                         PENDING
Phase 4 G3:                         PENDING
Phase 3 G2:                         PENDING
```

## 12. 구현 기록

(C1–C4 에서 채운다.)
