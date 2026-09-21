# Phase 2 실제 E57 G2 실행 runbook

이 문서는 **실제 갱도 E57 한 개를 실제 GPU 가 있는 로컬 장비에서 end-to-end 로 돌리는 절차** 다.
계약은 [PHASE2_CONTRACT.md](PHASE2_CONTRACT.md), 범위·gate 는 [ROADMAP.md](ROADMAP.md) 가 정한다.
여기는 그 계약을 **어떻게 실행하는가** 만 다룬다.

> **이 문서의 절차를 CI 가 돌리지 않는다.** CI 가 돌리는 것은 합성 structural gate
> (`tests/test_e2e_gate.py`) 이고, 그것은 G2 가 아니다. 합성 renderer 결과를 G2 라고 부르지
> 않는다.

`minegs eval render-depth` 가 호출하는 `GsplatDepthRenderer.render` 는 **지금까지 한 번도 실제
CUDA 환경에서 실행된 적이 없다.** 이 runbook 의 DEPTH stage 가 그 최초 실행이다. 여기서 처음
드러나는 문제가 있을 것이라고 가정하고 시작한다.

---

## 1. 환경

| 항목 | 요구 | 확인 |
|---|---|---|
| OS | Linux 또는 Windows | — |
| GPU | CUDA GPU 1장 (baseline 은 GPU 1장에 고정된다) | `nvidia-smi` |
| Python | 3.10 / 3.11 / 3.12 | `python -V` |
| minegs | 이 저장소 | `pip install -e ".[e57,gs]"` |
| E57 reader | `pye57` | `python -c "import pye57"` |
| backend | `gsplat` + `torch` (CUDA 빌드) | `python -c "import torch, gsplat; print(torch.cuda.is_available())"` |

`torch.cuda.is_available()` 가 `False` 면 **여기서 멈춘다.** minegs 는 CPU 로 대신 돌리지 않고
`ContractError` 로 거부한다 (fail closed). GPU 가 없는 장비에서는 Phase 2 G2 를 할 수 없다.

버전은 기록된다 — 각 stage 의 `tool_versions`, TRAIN stage 의 `runtime_env` (`gpu_model`,
`cuda_version`) 가 `workflow.json` 에 남고, 알 수 없으면 **null + 이유** 로 남는다. 채워 넣지
않는다.

---

## 2. 입력 준비

필요한 것은 네 가지다.

1. **E57 원본** — 측량 원본 파일. 저장소에 **절대 커밋하지 않는다.** 로컬에만 둔다.
2. **build config** — dataset 빌드 계약 (Phase 0C). `minegs dataset build-config-example` 로 뼈대를 뜬다.
3. **centerline CSV** — 설계 중심선. build config 의 `centerline.file` 이 가리킨다.
4. **held-out TLS reference PLY** — **TLS_GLOBAL 프레임**, 평가 전용. 학습 입력이 아니다.

### TLS reference 의 좌표

reference 는 반드시 `TLS_GLOBAL` 로 선언되어 있어야 한다. 프레임이 다르면 geometry stage 가
거부한다 — 좌표계가 다른 두 점군의 거리는 두 좌표계 사이의 거리이지 두 표면 사이의 거리가 아니다.

UTM 같은 큰 오프셋을 가진 측량 좌표라면 **float64 로 써야 한다**:

```bash
# float32 는 |좌표| 가 5000 m 를 넘으면 거부된다 (양자화가 0.5 m 에 달한다)
```

### holdout

geometry holdout 은 build config 가 선언하고 `manifest.json` 과 `judge()` 가 source of truth 다.
E2E runner 는 holdout 을 **새로 만들거나 바꾸지 않는다.** 선언하지 않으면 `sections_volume`
stage 가 "held-out TLS 가 없다" 로 거부한다.

---

## 3. 디스크

`--work-dir` 하나에 다음이 전부 쌓인다.

| 산출물 | 대략 크기 |
|---|---|
| staging tree (스캔 PLY + 이미지) | **원본 E57 의 1–2 배** |
| dataset (`images/`, `sparse/0/`, `init_points.ply`) | 이미지 크기에 비례 |
| run (checkpoint) | profile 에 따라 수백 MB ~ 수 GB |
| depth maps (`<stem>.npy`, float32) | 뷰 수 × W × H × 4 B |
| surface / sections / report | 작다 |

17 GB E57 이면 **여유 60 GB 이상**을 잡는다. 줄이는 손잡이는 `voxel_m` (스캔 다운샘플),
`max_scan_points`, `scan_ids` (구간만), depth 의 `stride` 다. 줄인 값은 전부 fingerprint 에
들어가므로, 나중에 바꾸면 그 stage 부터 stale 이 된다 — 그것이 의도한 동작이다.

`--work-dir` 은 E57 원본과 **다른 디스크**에 두는 편이 안전하다.

---

## 4. workflow config

`minegs e2e run` 이 받는 YAML/JSON 하나다. 상대 경로는 **이 파일이 있는 디렉터리** 기준으로
해석된다.

```yaml
# workflow.yaml
schema_version: "1.0"

# --- 입력
source_e57: raw/ep1_survey.e57        # 생략하면 staging_dir 를 그대로 채택(adopt)한다
staging_dir: work/staging
build_config: build_config.yaml
dataset_dir: work/dataset
runs_dir: work/runs
tls_reference_ply: raw/tls_full.ply   # TLS_GLOBAL, 평가 전용

# --- ingest
voxel_m: 0.01
max_scan_points: null
scan_ids: []                          # []  = 전부
mapping: null                         # 파노라마 매핑 파일이 따로 있으면
vendor_manifest: null
images_dir: null                      # 이미지가 E57 밖에 있으면

# --- training
profile: light                        # light | ... (profile 이름)
backend: gsplat
runner: local
native: false

# --- depth / surface
min_alpha: null                       # null = 렌더러 기본값(0.5)
stride: 2
max_depth_m: null

# --- evaluation
max_dist_m: 1.0
interval_m: 1.0
thickness_m: 0.5
angle_bins: 180
```

Windows 예시 (경로 예시는 문서에만 둔다 — 소스 코드에 실제 경로를 하드코딩하지 않는다):

```yaml
source_e57: D:/survey/ep1/ep1_survey.e57
staging_dir: D:/minegs/work/staging
tls_reference_ply: D:/survey/ep1/tls_full.ply
```

config 를 바꾸면 **같은 `--work-dir` 로 다시 열리지 않는다.** 설정이 바뀐 재실행은 다른
workflow 이고, 기록된 fingerprint 는 더 이상 유효하지 않은 설정에 대한 진술이 되기 때문이다.
새 `--work-dir` 을 쓰거나, 영향을 받는 stage 를 명시적으로 다시 만든다.

---

## 5. 실행

```bash
minegs e2e run workflow.yaml --work-dir work/wf
```

이게 전부다. stage 순서대로 `ingest → dataset → train → depth → surface → geometry →
sections_volume → report` 를 돌리고, 이미 성공한 stage 는 입력 identity 가 정확히 같을 때만
재사용한다.

처음에는 **끊어서 돌리는 편이 낫다.** 각 stage 가 몇 시간짜리일 수 있다.

```bash
minegs e2e run workflow.yaml --work-dir work/wf --through dataset
minegs e2e status --work-dir work/wf          # 여기서 golden gate 결과를 먼저 본다

minegs e2e run workflow.yaml --work-dir work/wf --through train
minegs e2e run workflow.yaml --work-dir work/wf --through surface
minegs e2e run workflow.yaml --work-dir work/wf            # 나머지 전부
```

`--through dataset` 에서 멈춰 `status` 를 먼저 보는 것을 권한다. camera convention 이 측정되었고
golden gate 가 통과했는지가 그 뒤 전부의 전제다.

---

## 6. 산출물

```
work/wf/
  workflow.json                       # 원장(ledger): stage 별 status·시간·identity·명령
  artifacts/
    golden_gate/<id>/report.json      # Phase 0C 재투영 게이트
    runs/<run_id>/                    # RunRecord, checkpoint
    depth/<id>/depth_manifest.json    # Phase 1B — 어느 run 의 어느 checkpoint 인지
    surface/<id>/surface.json + .ply  # Phase 1A/1B — depth_source=minegs_render
    geometry/<id>/geometry.json       # 양방향 형상 지표
    sections_volume/<id>/
      sections_predicted.json
      sections_reference.json
      paired_validation.json          # 단면 면적 오차 · 체적 오차 · coverage
      volume_predicted.json
    report/<id>/
      phase2_report.json              # machine-readable source of truth
      phase2_report.md                # 그것의 사람이 읽는 표현
```

report 만 다시 만들고 싶으면:

```bash
minegs e2e report --work-dir work/wf --out report/ep1
```

**stage 를 하나도 실행하지 않는다.** 제목 한 줄을 고치려고 training 을 다시 하지 않는다.

다만 **공짜는 아니다.** report 재생성은 먼저 완료된 stage 들의 입력 identity 를 다시 읽어
대조하고, 하나라도 움직였으면 거부한다 (§7 의 `stale` 과 같은 판정이다). 큰 E57 이면 여기서
source digest 를 다시 계산하는 비용이 든다. 값싼 명령이 비싼 명령의 gate 를 우회하는 길이 되면
안 되기 때문이다 — 실행이 멈추는 입력 위에서 문서만 새로 찍어내는 것이 바로 그 우회다.

같은 이유로 `paired_validation.json` 을 손으로 고친 뒤 report 를 다시 만들 수 없다. 그 파일은
stage 가 기록한 digest 와 대조되고, 다르면 거부된다.

---

## 7. 완료된 stage 에서 이어 하기

```bash
minegs e2e status --work-dir work/wf
```

| 열 | 뜻 |
|---|---|
| `status` | `pending`/`running`/`succeeded`/`failed`/`reused` |
| `fingerprint` | 그 stage 가 실행될 때의 입력 identity 요약 |
| `stale` | **지금 다시 돌리면 거부될 stage** |

`stale` 이 비어 있으면 그대로 이어서 `run` 하면 된다. 이미 성공한 stage 는 `reused` 로 표시되고
다시 실행되지 않는다.

`stale` 이 있으면 **그 stage 의 입력이 움직였다는 뜻**이다. E57 이 교체되었거나, build config 가
바뀌었거나, dataset 이 다시 빌드되었거나, run 이 다시 학습되었거나, depth map 이나 surface 점군이
바뀌었다. 판정은 원장에 적힌 값을 다시 읽는 것이 아니라 **디스크의 실물을 다시 읽어** 내린다 —
depth 는 map 바이트까지, surface 는 published PLY 를 record 와 대조해서.
minegs 는 조용히 재사용하지도, 조용히 다시 돌리지도 않는다. 무엇이 바뀌었는지 말하고 멈춘다.

명시적으로 다시 만든다:

```bash
minegs e2e run workflow.yaml --work-dir work/wf --rebuild-from depth
```

그 stage 와 **그 뒤 전부**가 비워지고 다시 실행된다. 무엇이 다시 만들어졌는지가 원장에 남는다.

`--rebuild-from ingest` / `--rebuild-from dataset` 은 기존 staging tree 와 dataset 디렉터리를
**교체한다.** 이 교체는 `--rebuild-from` 이 그 stage 를 포함할 때만 일어난다 — 평상시 `run` 은
이미 있는 artifact 를 절대 덮어쓰지 않는다. 아무도 파괴를 요청하지 않은 artifact 는 증거이고,
파괴 의사를 말하게 하는 것이 그 파괴를 보이게 하는 유일한 방법이다. (하위 builder 의 foreign
file · ownership 보호는 그대로 동작한다. 디렉터리를 무조건 지우는 것이 아니라, 자기가 만든 것을
교체하라고 요청하는 것이다.)

> 이것은 training resume 이 아니다. 이미 성공한 TRAIN stage 를 다시 돌리지 않고 그 run artifact
> 에서 DEPTH 부터 이어 간다는 뜻이다. gsplat training 자체의 중단 후 재개는 여전히 미구현이고
> fail closed 다 (Phase 0D.3).

---

## 8. 실패를 들여다보기

stage 가 실패하면 원장에 `failed` 와 `failure_reason` 이 남고, 그 뒤 stage 는 손대지 않는다.

```bash
minegs e2e status --work-dir work/wf            # failure_reason 이 빨갛게 나온다
minegs e2e status --work-dir work/wf --json state.json
```

자주 나오는 것들:

| 증상 | 원인 | 할 일 |
|---|---|---|
| `no such E57 file` / `changed during ingest` | 파일 경로가 틀렸거나 추출 도중 파일이 바뀌었다 | 경로 확인, 복사 중이면 끝난 뒤 다시 |
| `dataset has no centerline` | build config 에 `centerline` 이 없다 | 중심선 CSV 를 선언한다 |
| `declares no geometry holdout` | holdout 미선언 | build config 의 `geometry_holdout.ranges_m` |
| golden gate 실패 | camera convention / 프레임 / 재투영 문제 | `artifacts/golden_gate/<id>/report.json` 을 먼저 본다 |
| `torch.cuda` 관련 거부 | GPU/backend 미설치 | 환경 §1 |
| `surface ... does not promote` | depth 가 이 run 에서 렌더된 것으로 재유도되지 않는다 | depth stage 를 다시 만든다 |
| `not the points this surface was built from` | surface PLY 가 record 와 다르다 | 손으로 고치지 말고 surface 를 다시 만든다 |
| `nothing to compare` | TLS reference 가 holdout 구간을 덮지 않는다 | reference 의 구간·원점을 확인한다 |
| `stage ... completed against different inputs` | stale | §7 `--rebuild-from` |
| `... exists; pass --overwrite to replace it` / `already holds extraction output` | staging/dataset 디렉터리가 이미 있는데 이번 실행이 교체 의사를 밝히지 않았다 | `--rebuild-from ingest` 또는 `--rebuild-from dataset` 으로 다시 실행한다. 평상시 실행은 절대 덮어쓰지 않는다 |
| `cannot report on this workflow` | 완료된 stage 의 입력이 움직였다 | §7 `--rebuild-from`, 또는 run 이 이미 쓴 report 를 그대로 읽는다 |
| `not the numbers this workflow produced` | report 가 인용하는 artifact 가 stage 기록과 다르다 | 파일을 고치지 말고 해당 stage 를 다시 돌린다 |
| `surface ... hashes to ..., but the record says ...` | published surface PLY 가 record 와 다르다 | surface 를 다시 만든다 (`--rebuild-from surface`) |

실패한 stage 는 고친 뒤 그냥 다시 `run` 하면 된다 (실패한 stage 는 재사용 대상이 아니다).

---

## 9. 증거 모으기

G2 기록으로 남길 것:

1. `work/wf/workflow.json` — 원장 전체 (stage 별 시간·identity·명령·도구 버전)
2. `phase2_report.json` + `phase2_report.md`
3. `artifacts/geometry/<id>/geometry.json`
4. `artifacts/sections_volume/<id>/paired_validation.json`
5. `artifacts/depth/<id>/depth_manifest.json` — `renderer` 가 실제 `GsplatDepthRenderer` 인지
6. `artifacts/surface/<id>/surface.json` — `depth_source: minegs_render` 인지
7. 육안 검토용 이미지 (§10)
8. `nvidia-smi` 출력, `pip freeze`

report 상단의 세 줄을 확인한다:

```
real GPU training executed: True
real depth renderer executed: True
```

둘 중 하나라도 `False` 면 그 실행은 **G2 가 아니다.**

**원본 E57, staging tree, 영상은 저장소에 올리지 않는다.** 저장소로 가는 것은 리포트와 JSON
증거뿐이고, pod 로 가는 것은 `dataset/` 뿐이다.

---

## 10. 육안 검토 체크리스트

G2 는 숫자만으로 통과시키지 않는다. 사람이 최소한 다음을 본다.

- [ ] **camera / TLS overlay** — 카메라 위치가 갱도 안에 있고 방향이 맞는가
- [ ] **샘플 RGB 렌더** — 학습 결과가 벽처럼 보이는가
- [ ] **샘플 depth 맵** — 유효 비율, 근/원거리 범위가 갱도 반경과 맞는가
- [ ] **surface vs TLS overlay** — 두께·오프셋·뒤집힘이 보이는가
- [ ] **holdout 위치** — 평가 구간이 실제로 학습에서 빠진 구간인가
- [ ] **section overlay** — 예측 단면과 TLS 단면이 같은 자리에 그려지는가
- [ ] **명백한 이상** — frame flip, 축 교환, 스케일(1 unit = 1 m) 오류

검토가 끝나면 `phase2_report.json` 의 `maturity.human_visual_review_status` 를 사람이
`pass` 또는 `fail` 로 바꾼다. **minegs 는 이 값에 `pending` 외의 것을 쓰지 않는다.**

`real_data_validation_status` 를 `validated` 로 바꾸려면 `human_visual_review_status` 가 `pass`
여야 한다 — 스키마가 그것을 강제한다. 숫자만으로 G2 가 통과되지 않는다.

---

## 11. 그 전까지의 표현

이 runbook 을 실제 갱도에서 끝까지 돌리기 전에는 다음을 유지한다.

> Phase 2 E57 end-to-end workflow is implemented and structurally tested.
> Real-data scientific validation remains NOT VALIDATED.
> Phase 2 G2 remains PENDING.
