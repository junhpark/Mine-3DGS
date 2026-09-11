# minegs 개발 로드맵

**이 문서가 implementation sequence · validation gate · Definition of Done 의 source of truth 다.**
architecture invariant 는 [ARCHITECTURE.md](ARCHITECTURE.md) 가 담당한다. 두 문서는 서로를 참조하되
같은 내용을 중복해서 길게 갖지 않는다.

## 0. 읽는 법

이 문서는 두 단어를 엄격히 구분한다.

| 용어 | 뜻 |
|---|---|
| **implemented** | 코드가 있고 synthetic/unit 레벨에서 동작한다. 실데이터·GPU 검증은 아직이다. |
| **validated** | 해당 Phase 의 Gate 를 실제 데이터(또는 실제 GPU)에서 통과했다. |

Phase 는 Gate 를 통과해야 완료다. 코드가 머지되었다는 사실만으로 Phase 를 완료로 선언하지 않는다.
검증되지 않은 경로는 흉내 내지 않고 **fail-closed** 한다 (ARCHITECTURE.md §1.6, invariant 6·11·12).

## 1. 현재 상태 요약

| Phase | 이름 | 상태 |
|---|---|---|
| 0A | Foundation & Contract Freeze | **implemented, G1 통과** — PR #1 이 closeout |
| 0B | Real E57 Ingest | implemented (인터페이스+로직), **not validated** — 실제 E57 필요 |
| 0C | Metric Dataset Golden Gate | implemented, **not validated** — Golden Gate 미수행 |
| 0D | Local GS Baseline | implemented (어댑터·스테이징·러너), **not validated** — GPU 학습 미수행 |
| 1 | Metric Surface & Evaluation | 부분 implemented (지표·단면·체적), surface 추출 미구현 |
| 2 | E57 End-to-End MVP | 미착수 |
| 3 | Image / 360 Independent Reconstruction | 부분 implemented (커맨드 빌더·rig·정합), 미검증 |
| 4 | Advanced GS / Heavy Profile | 미착수 — `depth_loss` 는 여기서 설계 |
| 5 | Long Tunnel & Chunking | 예약만 (manifest.chunks) |
| 6 | RunPod / Reproducible Compute | **미구현, fail-closed** |
| 7 | Multi-Epoch Change Detection | 미구현 — single manifest 는 change claim 불가 |
| 8 | Viewer / Export / Web | 부분 implemented (Viser·export), FastAPI 미착수 |

## 2. Validation gate 체계

모든 검증을 같은 무게로 돌리지 않는다.

| Gate | 언제 | 내용 | 비용 |
|---|---|---|---|
| **G0 Fast CI** | 매 push / PR update | ruff, format, unit, schema, contract test, CLI smoke | 수십 초~수 분 |
| **G1 Synthetic Integration** | 기능 PR 에서 필요 시 | 합성 갱도, 프레임 변환, manifest 검증, 단면·체적 해석값, 커맨드 생성, fail-closed 동작 | GPU·실 E57 불필요 |
| **G2 Real-Data Golden Gate** | Phase 종료 / 명시적 milestone | 실제 E57 ingest, 파노라마 재투영, GPU gsplat 실행, TLS 오버레이, holdout 형상 평가 | 매 commit 수행 금지 |
| **G3 Benchmark / Release** | 큰 milestone | 장거리 갱도, heavy profile, RunPod 재현성, multi-epoch benchmark, runtime/memory | 개발 중 반복 금지 |

G0/G1 은 GitHub Actions 가 돌린다. G2/G3 는 실데이터·GPU 가 있는 환경에서 사람이 돌리고 결과를 기록한다.

## 3. Phase 정의

### Phase 0A — Foundation & Contract Freeze

데이터 계약·좌표계 계약·실행 계약·provenance·CLI·evaluation claim gate·CI 구조를 고정한다.

**범위**: 패키지 구조, config/schema versioning, manifest v1, migration, 좌표 프레임 타입,
provenance, CLI, 합성 데이터셋, contract 강제, CI, 미지원 경로 fail-closed.

**Gate**: G1 synthetic contract gate.

**Definition of Done**
- config/manifest/schema 계약이 일관됨
- TLS_GLOBAL ↔ LOCAL_METRIC frame invariant 가 명확함
- 미지원 경로가 fail-closed (silent fallback 없음)
- provenance hash 가 training-relevant artifact 를 포함함
- synthetic contract test 통과
- Python 3.10/3.11/3.12 CI 통과
- Dockerfile lint 통과
- architecture / roadmap 문서가 이후 Phase 범위를 명확히 정의함

**주장하지 않는 것**: 실제 E57 pipeline 검증, 파노라마 정합, GPU 학습 검증, RunPod 실행 가능,
heavy profile 실행, 논문급 형상 검증, multi-epoch 검증.

### Phase 0B — Real E57 Ingest

실제 E57 파일 구조를 안전하게 해석한다.

**범위**: inventory, scan enumeration, station mapping, scan 추출/분리, pose 추출,
PanoSource 탐지, embedded/external/vendor 파노라마 매핑, 대용량은 PDAL.

**하지 않을 것**: 3DGS 학습, 형상 평가, cloud.

**Gate (G2)**: 실제 E57 소구간 ingest 성공. 산출물 — scan/station inventory, pose table,
파노라마 매핑 리포트, 추출 point cloud 샘플, provenance.

실제 E57 데이터가 없으면 Phase complete 로 선언하지 않는다.

### Phase 0C — Metric Dataset Golden Gate

E57 point cloud · 파노라마 · 카메라 pose · LOCAL_METRIC 좌표가 실제 공간에서 일치함을 검증한다.

**범위**: 파노라마 규약, equirect → ring crop, 합성 intrinsics, station pose → COLMAP,
데이터셋 계약 생성, `init_points.ply`, TLS_GLOBAL ↔ LOCAL_METRIC, Viser 정합, 재투영 오버레이.

**Gate (G2) — Golden Gate**: 실제 TLS 포인트를 파노라마에 재투영했을 때 벽면 edge, 파이프,
케이블, 갱도 경계, 식별 가능한 물체가 영상과 정렬된다. Viser 에서 TLS · camera frustum ·
initialization point 가 동일 공간에서 일치한다.

**이 Gate 가 실패하면 Phase 0D 로 이동하지 않는다.**

### Phase 0D — Local GS Baseline

실제 소구간을 로컬 GPU 에서 gsplat baseline 으로 끝까지 학습한다.

**범위**: pinned GPU docker image, gsplat v1.5.3 executable contract, LocalRunner,
light profile, staging, checkpoint/output, LOCAL_METRIC 출력 정규화 계약.

**원칙**: `normalize_world_space=false`, BACKEND_INTERNAL = LOCAL_METRIC.

**Gate (G2)**: 실제 갱도 소구간 학습 성공 — trainer 가 실행되고, 설정한 step 까지 도달하고,
PLY 가 생성되고, PLY 좌표 범위가 물리적으로 타당하고, 출력 프레임이 LOCAL_METRIC 이며,
TLS_GLOBAL 로 변환했을 때 입력 TLS 와 겹친다.

GPU smoke 를 통과하지 않으면 Phase 0D 완료라고 하지 않는다.

### Phase 1 — Metric Surface & Evaluation

3DGS representation 을 측량 가능한 surface representation 으로 변환한다.

**범위**: rendered depth, surface point 생성, mesh/TSDF, 양방향 형상 지표, geometry holdout,
중심선 단면, A(s), 체적 적분, 설계 대비, missing-data 처리.

**원칙**: 가우시안 중심을 TLS 와 직접 비교하지 않는다. 항상 GS → depth/mesh/surface → 평가.

**이 Phase 에서 갚을 기술 부채**: 현재 `integrate_sections` 는 invalid section 을 제거한 뒤
양쪽 valid section 사이를 그대로 사다리꼴 적분한다. 큰 결측 구간을 가로질러 적분하면 체적이
과대·과소 평가된다. 수정 방향 — 연속된 valid segment 별로만 적분, coverage fraction 기록,
missing interval 명시, coverage 가 부족하면 volume accuracy claim 자체를 거부.

**Gate**: G1 (합성 known geometry) + G2 (실제 holdout 형상 평가, 양방향 지표,
section/volume 리포트 일관성).

### Phase 2 — E57 End-to-End MVP

최초의 완전한 scientific workflow 를 실제 갱도에서 완성한다. **minegs v0.1 MVP milestone.**

E57 → 파노라마 → metric dataset → local gsplat → surface → TLS_GLOBAL → 형상 비교 →
단면 → 체적 → 리포트.

**Gate (G2)**: 실제 갱도 한 구간에서 end-to-end 리포트 생성 — 갱도 연장, 스테이션 수,
이미지 수, 학습 설정, 재구성 소요시간, 형상 정확도 median/P95, completeness, 단면 면적 오차,
체적 오차, valid coverage.

이 Phase 완료 전에는 cloud infrastructure 와 advanced backend 최적화를 우선하지 않는다.

### Phase 3 — Image / 360 Independent Reconstruction

TLS 초기 형상 없이 일반 영상·360 영상만으로 독립 재구성 경로를 만든다.

**범위**: ffmpeg 추출, 블러/중복 제거, 마스크, 360 ring crop, COLMAP rig,
global/incremental SfM, 희소 재구성, Sim3 초기 정합, SE3 ICP, 정합 진단,
metric dataset 계약으로 변환.

**Gate (G2)**: 독립 영상/360 재구성을 TLS reference 에 등록하고 정량 검증.
TLS-assisted GS 와 image-only GS 의 metric accuracy 비교가 가능해야 한다.

### Phase 4 — Advanced GS / Heavy Profile

Phase 2/3 에서 **실제로 관측된 failure mode** 를 근거로 품질을 개선한다.
기술을 먼저 추가하지 않는다.

**후보**: appearance embedding, 조명 보정, bilateral grid, depth supervision,
normal supervision, 2DGS, PGSR, surface-aware backend.

**여기서 해결할 계약 부채 — `depth_loss`**: upstream gsplat 의 depth supervision 은 COLMAP
image → point observation track 을 사용한다. 현재 staging 은 `init_points.ply` 를 `points3D`
로 쓰면서 그 track 을 비우므로 둘은 구조적으로 양립하지 않는다. 그래서 Phase 0A/0D 에서
`depth_loss` 요청은 `ContractError` 로 거부된다 (`minegs/train/backends/gsplat.py`,
`DEPTH_LOSS_REFUSAL`). Phase 4 에서 depth supervision 을 다시 설계할 때 다음을 명시해야 한다.

- depth 의 source (TLS 렌더 깊이 / SfM / 센서)
- image correspondence 를 어떻게 유지할 것인가 (tracked SfM geometry 병행 등)
- leakage 거동 — holdout 구간의 깊이가 학습에 들어가지 않는가
- metric frame — 깊이가 어느 프레임의 m 인가
- uncertainty — 깊이 신뢰도를 loss 에 어떻게 반영하는가

**같은 Phase 의 별도 항목 — `normalize_world_space=true`**: 현재 거부된다
(`NORMALIZE_REFUSAL`). 활성화하려면 upstream gsplat 정규화와의 **equivalence test** 가
선행되어야 한다. 재구현한 `similarity_from_cameras` + `align_principle_axes` 가 실제 파서와
동일한 변환을 만드는지 데이터셋별로 확인하고, 커맨드를 만드는 쪽과 변환을 계산하는 쪽이
같은 파일시스템을 보는지(docker path 문제)도 함께 해결한다.

**Gate (G3)**: 동일 dataset/protocol 에서 baseline 대비 정량적 개선.
"학습이 됐다" 만으로 Phase complete 로 하지 않는다.

### Phase 5 — Long Tunnel & Chunking

소구간에서 검증된 파이프라인을 장거리 갱도(100 m–1 km)로 확장한다.

**범위**: 중심선 chainage 청킹, overlap, 청크별 local frame, 청크 학습, 병합/전이,
메모리 제어, 재현성, seam 진단. XYZ 격자 청킹으로 바꾸지 않는다.

**Gate (G3)**: 모든 청크 완료, overlap 정렬, 심각한 seam 불연속 없음,
전역 TLS 재구성 복원 가능, 메모리/런타임 거동 문서화.

### Phase 6 — RunPod / Reproducible Compute

검증된 로컬 파이프라인을 cloud GPU 로 옮긴다. RunPod 는 연구 core 가 아니라 compute
infrastructure 이므로 로컬 scientific workflow 가 안정화된 뒤 구현한다.

**범위**: 동일 GPU 이미지 digest, dataset-only 업로드, network volume, pod-side staging,
resume, checkpoint, **exit-code 기반 status**, artifact 동기화 복귀, provenance, GPU 타입 기록.

**현재 상태**: `RunPodRunner.submit` 은 외부 호출 전에 `NotYetImplementedError` 를 던진다.
필요한 단계는 `minegs/train/runner/runpod.py` docstring 에 있다. 특히 이전 스케치에 없던
두 단계 — 원격 remote 의 내용을 pod `/data` 로 들여오는 단계, pod 산출물을 remote 로
내보내는 단계 — 를 반드시 포함한다. pod lifecycle state (EXITED/TERMINATED) 만으로 성공을
판정하지 않는다.

**Gate (G3)**: Local ↔ RunPod 재현성. 동일 code SHA · docker digest · dataset hash · profile
조건에서 결과 차이가 허용범위 내인지 확인.

### Phase 7 — Multi-Epoch Change Detection

동일 갱도 구간의 반복 계측에서 실제 변화량을 추정한다.

**범위**: epoch pair 계약, 동일 chainage 정렬, epoch 호환성 검사, surface 차분,
단면 차분, 체적 변화, 불확실성, 유의성 임계.

**pair-level protocol**: 이 Phase 에서 `judge_change(manifest_a, manifest_b)` (또는 동등한
pair evaluator) 를 도입한다. 검증 항목 — 서로 다른 epoch ID, 호환 가능한 좌표 프레임,
호환 가능한 metric scale basis, 공통 reference axis, 겹치는 평가 chainage,
누수 없는 initialization/evaluation 구간.

**현재 상태**: single manifest 의 `judge()` 는 `Protocol.CHANGE` 도 `Claim.CHANGE_VOLUME` 도
절대 생성하지 않는다. `diff_sections` 는 산술 결과이며 `geometry_diagnostic` 으로만 표시된다.

**Gate (G2/G3)**: 합성 known-change 데이터셋에서 ground truth 복원, 실제 반복 계측에서
변화 없는 구간의 안정성 확인, 변화 구간에서 물리적으로 타당한 ΔV.

### Phase 8 — Viewer / Export / Web

연구 결과를 분석·공유 가능한 인터페이스로 만든다.

**범위**: Viser 뷰어 개선, TLS/GS/mesh 토글, 단면 슬라이더, chainage 스크러버, 히트맵,
before/after 비교, `.spz` export, SuperSplat 호환, (선택) FastAPI.

FastAPI 는 실제 multi-user/remote 요구가 생길 때만 추가한다.

**Gate**: 연구자가 CLI 결과를 시각적으로 검토하고 export 할 수 있을 것.
Web application 자체를 core research gate 로 삼지 않는다.

## 4. PR 크기 및 개발 방식

Phase 하나 = 1~3 PR. 하나의 PR 에서 ingest · training · evaluation · cloud · UI 를
동시에 구현하지 않는다.

Phase 0B 예시:

| PR | 범위 |
|---|---|
| #2 | E57 inventory + scan/station contract |
| #3 | PanoSource + station/panorama 매핑 |
| #4 | Phase 0B real-E57 closeout (실데이터 검증 수정만) |

실데이터 검증에서 나온 수정은 별도 closeout PR 로 둔다.

## 5. Architecture freeze

다음 invariant 는 임의로 변경하지 않는다 (상세: ARCHITECTURE.md §1, §3, §5, §9).

1. dataset contract 는 하나.
2. raw source 는 training runtime 에 직접 의존하지 않는다.
3. TLS_GLOBAL ↔ LOCAL_METRIC 은 metric-preserving transform.
4. LOCAL_METRIC 은 1 unit = 1 m.
5. backend internal transform 은 반드시 reversible.
6. 검증되지 않은 scientific claim 은 fail-closed.
7. evaluation leakage 는 manifest contract 에서 차단.
8. 가우시안 중심을 geometry surface 로 간주하지 않는다.
9. raw → dataset → run → eval → export provenance 를 유지한다.
10. cloud execution 이 local scientific result 를 바꾸지 않는다.
11. 실제 기능과 문서/상태 표현은 항상 일치한다.
12. 실데이터 Gate 를 통과하지 않은 Phase 는 implemented 와 validated 를 구분한다.

architecture 변경이 필요하면 구현 중 암묵적으로 바꾸지 말고 ARCHITECTURE.md §14
(결정 이력) 에 별도 decision 으로 기록한다.

## 6. 현재 fail-closed 목록

계약상 거부되는 경로와 근거. 각 항목은 회귀 테스트를 가진다.

| 요청 | 결과 | 근거 | 해제 시점 |
|---|---|---|---|
| `reconstruction` run 에 형상 정확도 요청 | `ProtocolViolation` (exit 3) | §5 성능 주장 불가 | 해당 없음 (설계) |
| single manifest 의 change claim | `judge` 가 부여 안 함, `require` 는 `ProtocolViolation` | change = 2 epochs | Phase 7 pair protocol |
| `depth_loss: true` | `ContractError` (exit 2) | TLS staging 이 COLMAP track 제거 | Phase 4 |
| `normalize_world_space: true` | `ContractError` (exit 2) | upstream equivalence 미검증 | Phase 4 |
| `--profile heavy` 실행 | `ContractError` — depth_loss 때문에 | 위와 동일 | Phase 4 |
| `--runner runpod` | `NotYetImplementedError` (exit 4) | 미구현 | Phase 6 |
| `backend_args` 에 하이픈/언더스코어 두 철자 | `ContractError` | tyro 는 둘 다 받으므로 거부를 우회할 수 있다 | 해당 없음 (설계) |
| surface 추출(depth 렌더·TSDF) | `NotYetImplementedError` | 미구현 | Phase 1 |
| GLUEMAP SfM | `NotYetImplementedError` | 의존성 무거움, 보류 | Phase 3 |
| `pgsr` / `2dgs` / `splatfacto` backend | `NotYetImplementedError` | 미구현 | Phase 4 |
| `inria` backend | `ContractError` | non-commercial 라이선스 | 해당 없음 |
| gsplat trainer 미탐지 | `ContractError` | wheel 에 trainer 없음 | 해당 없음 (docker 로 해결) |
| `raw/` push | `ContractError` | 원본은 로컬에만 (§1.4) | 해당 없음 (설계) |
| `scale.basis` 없는 metric claim | claim 거부 | §4 | 해당 없음 (설계) |
