# Phase 2 — E57 End-to-End MVP 계약 (freeze)

이 문서는 Phase 2 구현 **전에** 고정한 계약이다. 구현이 이 문서를 바꾸는 것이 아니라, 이 문서가
구현의 판정 기준이다. 범위·완료 조건·표현(wording)을 여기서 정한다.

source of truth: 범위와 gate는 [docs/ROADMAP.md](ROADMAP.md), 불변식은
[docs/ARCHITECTURE.md](ARCHITECTURE.md). 이 문서는 그 둘 위에서 Phase 2만 다룬다.

## 1. 무엇을 만드는가

새 reconstruction 알고리즘이 아니다. Phase 0A–1C 가 이미 닫아 둔 계약과 증거 경계를 **실제 E57
워크플로 하나로 관통**시켜 minegs v0.1 End-to-End MVP 를 만든다.

```
E57
 → inventory / image mapping / extraction
 → camera convention evidence
 → metric dataset
 → dataset validation / protocol / holdout
 → local gsplat training
 → minegs-rendered metric depth
 → verified metric surface
 → TLS_GLOBAL geometry evaluation
 → predicted sections
 → TLS reference sections
 → section-area validation
 → gap-safe predicted/reference volume
 → volume validation
 → Phase 2 E2E report
```

## 2. Stage graph

| # | stage | 하는 일 | 기존 계약 |
|---|---|---|---|
| 1 | `ingest` | E57 inventory → image mapping → extraction (staging tree) | Phase 0B.1–0B.3 |
| 2 | `dataset` | camera calibration → `from-e57` → validate → protocol → golden gate | Phase 0C |
| 3 | `train` | local gsplat run → SUCCEEDED `RunRecord` | Phase 0D.2 |
| 4 | `depth` | `render-depth` → `DepthManifest` | Phase 1B |
| 5 | `surface` | `surface-depth` → `SurfaceRecord` (`minegs_render`) | Phase 1A/1B |
| 6 | `geometry` | holdout TLS 에 대한 양방향 형상 평가 | Phase 1A/1C |
| 7 | `sections_volume` | predicted/reference 단면 + gap-safe 체적 비교 | Phase 1C + 신규 paired validation |
| 8 | `report` | 위 artifact 들만 모아 Phase 2 report 생성 | 신규 |

stage 순서는 `minegs/e2e/models.py` 의 `STAGE_ORDER` 하나로만 선언한다. 문서와 코드가 갈라질
자리를 만들지 않는다.

## 3. Workflow state schema

`<workflow_dir>/workflow.json` = `WorkflowState` (schema 1.0).

```
workflow_id · source_sha256 · created_at · updated_at · config
stages: { <stage>: StageRecord }
provenance
```

`StageRecord`:

```
stage · status · started_at · completed_at · elapsed_seconds
input_fingerprint · inputs · outputs · command
git_commit · minegs_version · tool_versions · runtime_env · failure_reason
```

`status` ∈ `pending | running | succeeded | failed | reused`.

`reused` 는 `succeeded` 와 구별한다 — report 가 "이 워크플로가 실제로 돌린 것" 과 "이전 실행에서
가져온 것" 을 말할 수 있어야 하기 때문이다.

### 재사용 규칙 (silent reuse 금지)

`input_fingerprint` 는 그 stage 가 **실제로 의존하는 identity** 들의 sha256 이다. 재사용 전에
지금의 세계에서 다시 계산해 대조한다.

| stage | fingerprint 재료 |
|---|---|
| `ingest` | E57 sha256 + 추출 옵션 |
| `dataset` | staging digest + build config hash |
| `train` | dataset_id + dataset_hash + profile + backend |
| `depth` | run_id + dataset_hash + checkpoint sha256 |
| `surface` | depth manifest id + **depth map bytes** (`depth_digest`) + run_id + dataset_hash |
| `geometry` | **디스크에서 다시 읽어 검증한** surface_id + point_sha256 + depth_source + dataset_hash + TLS reference digest |
| `sections_volume` | 위와 같음 + section parameters |
| `report` | paired_validation.json digest + geometry report digest (+ `_upstream` 로 위 전부) |

`ingest` 와 `dataset` 은 고정 경로에 publish 하므로, 이미 있는 artifact 를 교체하는 것은
`--rebuild-from` 이 그 stage 를 포함할 때뿐이다. 평상시 실행은 덮어쓰지 않는다.

재료는 **세상에서 다시 읽은 것**이어야 하고, 원장에서 복사한 것이면 안 된다. 원장에 적힌
`point_sha256` 은 SURFACE stage 가 어떤 파일에 대해 한 *진술*이고, 그 진술이 아직 참이냐고
원장에게 물으면 답은 언제나 "그렇다" 이기 때문이다. 그래서 evaluation 단계는 surface artifact
를 `load_surface` + `check_surface` 로 다시 읽는다 — `minegs eval geometry` 가 적용하는 바로 그
gate 다. 마찬가지로 manifest 의 digest 는 어떤 depth map 이 선언되었는지만 말하고 그 안에 무엇이
있는지는 말하지 않으므로, SURFACE 는 map 자체를 `depth_digest` 로 덮는다.

fingerprint 가 다르면 **fail closed** 다. 예전 SUCCESS 를 재사용하지 않고, 무엇이 움직였는지
말한 뒤 명시적 rebuild 를 요구한다. E57 교체 · build config 변경 · dataset hash 변경 · run 교체 ·
surface 교체 중 하나라도 있으면 downstream evidence 는 stale 이다.

이것은 **training resume 이 아니다.** 이미 성공한 TRAIN stage 를 다시 돌리지 않고 그 run
artifact 에서 DEPTH 부터 이어 간다는 뜻이다. gsplat training 자체의 resume 은 여전히 미구현이고
fail closed 다 (Phase 0D.3).

## 3.1 CLI

`minegs e2e run | status | report` — 하나의 discoverable entry point.

| 커맨드 | 하는 일 |
|---|---|
| `run <config> --work-dir <dir> [--through <stage>] [--rebuild-from <stage>]` | stage 를 순서대로 실행한다. 비싼 쪽. |
| `status --work-dir <dir> [--json <file>]` | 읽기 전용. 어디까지 됐고, 지금 실행하면 무엇이 거부되는지. |
| `report --work-dir <dir> [--out <dir>]` | 이미 있는 artifact 에서 문서만 다시 만든다. **stage 를 하나도 실행하지 않는다.** |

workflow 실행과 report 생성은 분리한다. 제목 한 줄을 고치려고 E57 추출이나 training 을 다시
하지 않는다.

CLI 에는 trainer 나 renderer 를 대체하는 flag 가 **없다**. 그 seam 은 Python 에서 structural
gate 가 잡고, 잡았다는 사실을 stage 와 report 양쪽에 기록한다. `--fake-renderer` 같은 flag 는
GPU 가 한 번도 돌지 않은 채로 Phase 2 report 를 만들어 내는 길이고, 이 저장소의 계약이 막으려는
것이 정확히 그것이다.

## 4. Report schema

`phase2_report.json` = `Phase2Report` (schema 1.0) 가 **machine-readable source of truth** 다.
`phase2_report.md` 는 그것의 사람이 읽는 표현이며, 수치를 다시 계산하지 않는다.

블록: `source · dataset · training · reconstruction · geometry · sections · volume · runtime ·
maturity · stages`.

채울 수 없는 값은 **null + 이유** 다. 그럴듯한 값을 넣지 않는다.

report 는 원장(ledger)에서 읽지 않고 stage 의 **출력 파일**을 인용하는 유일한 지점이다
(`paired_validation.json`). 그래서 그 파일은 stage 가 기록한 digest 와 대조된다. digest 가
기록되지 않은 원장은 "맞다" 고 답할 수 없으므로 신뢰하지 않고 거부한다.

report 재생성(`minegs e2e report`)은 stage 를 실행하지 않지만, 완료된 stage 가 전부 여전히
current 인지 먼저 확인하고 아니면 거부한다. 실행이 거부하는 상태에서 문서만 새로 만들 수 있으면
그것이 stale evidence 를 세탁하는 가장 짧은 경로가 되기 때문이다.

`report` stage 의 문서는 자기 record 가 확정된 뒤 한 번 더 쓰인다. stage 안에서 쓴 문서는 자기
자신을 `running` 으로밖에 적을 수 없고, 최종 원장은 `succeeded` 라서 workflow 의 최종 진술이
workflow 와 어긋나기 때문이다. 결과를 예측해서 적는 대신, 쓰고 나서 무슨 일이 있었는지 적는다.

## 5. Paired TLS validation (신규)

ROADMAP 의 Phase 2 G2 는 형상 지표뿐 아니라 **단면 면적 오차 · 체적 오차 · valid coverage** 를
요구한다. predicted volume 만 출력하는 것으로는 충족되지 않는다.

그래서 reconstruction 과 held-out TLS reference 를 **같은 section grid** 에서 비교한다: 같은
reference centerline · 같은 chainage · 같은 `interval_m`/`thickness_m`/`angle_bins` · 같은
holdout range.

* 단면: station 단위로 pair 하고 signed/absolute/relative 오차, MAE, median, P95 를 낸다.
  relative error 는 reference 면적이 수치적으로 유효할 때만 계산한다 — 0 division 을 숨기지 않는다.
* 체적: predicted 와 reference 를 **common integration domain** 에서만 비교한다. 둘 중 하나라도
  결측인 구간은 가로질러 적분하지 않는다. Phase 1C 의 gap-safe helper 를 재사용하고 새 사다리꼴
  구현을 만들지 않는다.

서로 다른 coverage 에서 계산한 100 m³ 와 102 m³ 를 빼는 것은 오차가 아니다.

**새 scientific claim enum 을 만들지 않는다.** 이것은 Phase 2 G2 validation evidence 이지
`Claim` 의 새 값이 아니다.

## 6. Structural validation 과 real G2 의 차이

이 둘은 **다르다**. 섞어 부르지 않는다.

| | synthetic structural gate | real G2 |
|---|---|---|
| 입력 | 합성/tiny E57 | 실제 갱도 E57 |
| backend | 대체(substituted) | 실제 gsplat |
| renderer | 대체 | 실제 `GsplatDepthRenderer.render` |
| GPU | 없음 | 실제 CUDA |
| 검증하는 것 | control path 와 evidence path | 과학적 정확도 |
| CI | 돈다 | 돌지 않는다 |

합성 renderer 결과를 **절대 G2 라고 부르지 않는다.**

## 7. 완료 조건

### Phase 2 implementation (이 PR 의 merge 조건)

* 단일 E2E orchestration path 존재
* 기존 scientific boundary 전부 보존
* runtime stage checkpoint 동작
* paired TLS section validation 존재
* paired TLS volume validation 존재
* gap-safe common-domain 비교 존재
* machine-readable + human-readable report 존재
* synthetic E2E gate 통과
* 기존 regression 전부 통과, CI green
* scientific overclaim 없음

### Phase 2 scientific milestone (별도)

* real E57 G2 실행
* real gsplat training 실행
* real `GsplatDepthRenderer.render` 실행 — 이 함수는 지금까지 **한 번도 실행된 적이 없다**
* final report 생성
* 사람의 육안 검토 완료
* 지표 검토

## 8. Maturity wording

실제 G2 전까지 다음을 유지한다.

> Phase 2 E57 end-to-end workflow is implemented and structurally tested.
> Real-data scientific validation remains NOT VALIDATED.
> Phase 2 G2 remains PENDING.

실제 G2 이후에도 그 결과를 **평가한 데이터셋을 넘는 일반적 과학적 정확도 주장으로 바꾸지 않는다.**

`human_visual_review_status` 는 `pending | pass | fail` 이고, minegs 는 `pending` 외의 값을 쓰지
않는다. 사람이 다음을 본 뒤에만 바뀐다: camera/TLS overlay · 샘플 RGB/depth · surface vs TLS
overlay · holdout 위치 · section overlay · 명백한 frame flip/방향/스케일 이상.

## 9. Out of scope

TSDF · mesh 추출 · mesh 체적 · 새 GS backend · 2DGS · PGSR · depth supervision · pose refinement ·
antialiased renderer · image-only reconstruction · 360 independent SfM · long-tunnel chunk
training · RunPod · multi-GPU · training resume · change/`judge_change` · multi-epoch · viewer
재설계 · web UI · 임의 accuracy threshold · 임의 coverage threshold · 임의 최대 section interval.

실제 failure mode 가 Phase 2 결과에서 확인된 뒤 Phase 4/5 에서 다룬다.

## 10. Definition of Done

구현:

> Starting from an E57 survey, Mine-3DGS can reproducibly carry the evidence chain through
> metric dataset creation, training, rendered metric depth, verified surface reconstruction,
> held-out TLS geometry evaluation, section and volume validation, and a single auditable
> Phase 2 report, without bypassing any Phase 0–1 scientific contract.

과학적 milestone:

> The same workflow has been executed on a real underground E57 dataset with a real GPU and the
> real gsplat renderer, the held-out TLS evaluation report has been generated, and the required
> visual evidence has been reviewed by a human.
