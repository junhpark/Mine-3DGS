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
| 0B | Real E57 Ingest | **0B.1·0B.2·0B.3 implemented** (inventory·매핑·추출), **not validated** — 실제 E57 필요 |
| 0C | Metric Dataset Golden Gate | implemented, **not validated** — Golden Gate 미수행 |
| 0D | Local GS Baseline | **0D.1 implemented + structurally tested** (resume 계약·경로 변환·fail-closed). **0D.2 not run** — 실제 GPU 학습 미수행. 0D 전체는 **NOT COMPLETE** (§0D) |
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

실제 E57 파일 구조를 안전하게 해석한다. 세 개의 PR 로 나눈다.

| 하위 | 범위 | 상태 |
|---|---|---|
| **0B.1** | inventory, scan enumeration, scan/station 계약, pose 검사, CLI 리포트 | implemented (PR #2), **not validated** |
| **0B.2** | image asset 탐지, representation 분류, 증거 기반 station↔image 매핑 계약 | implemented (PR #4), **not validated** |
| **0B.3** | scan/image 추출, invalid-state mask, extraction manifest, staging 산출물 | implemented (PR #4), **not validated** |

**하지 않을 것**: 3DGS 학습, 형상 평가, cloud.

**0B.1 원칙 — inspection first, interpretation second.** E57 구조를 가정하지 않는다.
scan 하나가 station 하나라고, 모든 scan 에 pose·RGB·이름·GUID 가 있다고, 파노라마가 파일 안에
있다고, 좌표가 TLS_GLOBAL 이라고 가정하지 않는다. 없는 것은 `None` 으로 기록하고 이유를 남긴다.

* **식별자**: `scan_000`, `S000` 은 **해당 source 파일 안의 scan index** 로 결정된다. vendor
  name/GUID 에는 의존하지 않으므로 그것들이 없는 파일도 안정적으로 참조할 수 있지만, 파일 안의
  scan 순서에는 의존한다. 파일의 SHA-256 과 함께 쓰면 scan 을 모호함 없이 지목할 수 있다.
  vendor GUID 는 있으면 함께 보존하되 identity 로 삼지 않는다.
* **좌표**: E57 자체 좌표는 `SOURCE` 다. `T_source_from_scan` 처럼 방향을 이름에 적는다.
  TLS_GLOBAL 선언과 LOCAL_METRIC 변환은 dataset materialization(0C) 의 몫이다.
* **pose**: `absent`(선언 안 됨) · `unreadable`(선언됐으나 파싱 실패) · `invalid` · `identity` ·
  `valid` 다섯 상태를 구분한다. 선언됐으나 읽히지 않은 pose 를 "없음"으로 합치지 않는다.
  유효하지 않은 pose 에 `se3()` 를 호출하면 `E57PoseUnusableError` 로 거부된다 —
  추출 경로(`split`)도 같은 리더를 쓰므로 inventory 와 다른 답을 낼 수 없다.
  검증 항목은 finite·orthonormal·det ≈ +1·unit quaternion 이다.
  잘못된 pose 를 identity 로 바꾸지 않는다. 반올림된 회전(4~6자리)은 문제가 아니라 note 로 구분하고,
  변환용으로 orthonormalise 했다는 사실과 편차를 함께 기록한다.
* **비용**: header 만 읽는다. point count 는 `points.childCount()`, bounds 는 header 값이므로
  **메타데이터 파싱은 O(scan 수)** 이고 점 배열을 적재하지 않는다. 단 provenance SHA-256 은
  **O(파일 크기)** 다(`compute_hash=False` / `--no-hash` 로 생략 가능).
* **station**: `mapping_status = inferred_from_scan`. 확정은 0B.2 가 파노라마 근거로 한다.

**0B.2 원칙 — 매핑은 추론이 아니라 증거다.** "이 파노라마는 어느 station 것인가" 에 대한
답은 데이터가 말해 준 것만 쓴다. 틀린 매핑은 crash 하지도 warn 하지도 않는다 — 학습이 되고
수렴까지 하는 무의미한 재구성이 나올 뿐이다.

| 근거 | 출처 | status |
|---|---|---|
| `e57_associated_guid` | E57 자신의 `associatedData3DGuid` 가 정확히 한 scan GUID 와 연결 | `confirmed` |
| `vendor_manifest` | 캡처 소프트웨어가 생성한 index (`vendor`·`generated_by` 선언 필수) | `confirmed` |
| `explicit_mapping` | 사용자가 직접 작성한 CSV/JSON | `manual` |

* **금지되는 근거**: scan index == image index, 개수 동일, 파일 순서, lexical sort, fuzzy
  name matching, timestamp/EXIF 근접, 문서화되지 않은 vendor 관례, 검증된 카메라 형상 없는
  최근접 pose. 각각에 "매핑을 만들지 않는다" 를 확인하는 회귀 테스트가 있다. 진단용 **hint**
  로 출력하는 것은 가능하지만 `MappingRecord` 의 target 이 되지 않는다.
* **status**: `confirmed` · `manual` · `unmapped` · `ambiguous`(하나의 근거가 여러 target) ·
  `orphan`(근거는 있으나 target 없음) · `conflict`(근거끼리 모순). GUID 를 두 scan 이
  선언하면 first match 가 아니라 `ambiguous` 다. 사용자 매핑이 파일 자체 근거와 다르면
  덮어쓰기가 아니라 `conflict` 다 — **E57 이 선언한 GUID 가 이 파일에 없을 때(orphan)도
  마찬가지다.** target 이 없다는 것은 파일의 진술이 *해석 불가능*하다는 뜻이지 *진술이 없다*는
  뜻이 아니고, 그 진술을 가장 이해하지 못하는 순간에 다른 근거가 조용히 이기게 두는 것이
  이 모듈이 막으려는 바로 그 override 다. 명시적 override 정책이 필요해지면 그때는 선언된
  flag 여야지 target 이 없다는 부작용이어서는 안 된다.
* **provenance**: 결과를 바꾸는 입력은 전부 hash 한다 — E57, `--mapping`, `--vendor-manifest`,
  외부 이미지 각각. 같은 E57 을 서로 다른 CSV 로 매핑하면 서로 다른 결과이므로, E57 만 적힌
  provenance 는 둘을 구분하지도 재현하지도 못한다. 외부 이미지는 파일별 digest 와, 그것들로부터
  유도한 image-set digest 하나(추가 I/O 없음)를 함께 기록한다. `--no-hash` 는 전부에 일관되게
  적용되고 이유를 남긴다 — 일부만 반영된 digest 는 없는 것보다 나쁘다.
* **산출물**: `PanoMappingReport` (schema 1.0) — `E57Inventory` 에 필드를 더하지 않는다.
  inventory 는 "파일에 무엇이 있는가", 이쪽은 "무엇이 무엇과 짝인가" 이고, 두 번째 질문은
  새 매핑 파일로 다시 답할 수 있어야 한다.
* **외부 이미지**: 디렉토리만으로는 매핑하지 않는다. 파일명을 station ID 로 해석하지 않는다
  (`VendorExport` 의 `Station_03.jpg → S03` 추론도 이때 fail-closed 로 바뀌었다).

**0B.3 원칙 — 처음으로 point payload 를 읽는다. 산출물은 staging 이지 dataset 이 아니다.**

```
<work_dir>/inventory.json  pano_mapping.json  extraction_manifest.json
           scans/scan_000.ply + scan_000.pose.json
           images/image_000.jpg
```

`dataset/` 계약으로 오인될 구조를 만들지 않는다. TLS_GLOBAL 선언, LOCAL_METRIC origin,
train/test split, `init_points.ply`, dataset manifest 는 전부 0C 다.

* **invalid-state mask**: E57 scan 은 고정 길이 record array 다. `cartesianInvalidState` 로
  점을 지운 뒤 색을 `rgb[:len(xyz)]` 로 자르면 **첫 invalid 점 이후의 모든 색이 한 칸씩
  밀린다** — 뷰어에서는 멀쩡해 보이고 결과만 틀린다. boolean mask 를 한 번 만들어 xyz · RGB ·
  intensity · row/column · 나머지 유지 컬럼 **전부에 동일하게** 적용한다. 길이가 다른 컬럼은
  하드 실패다. state 1(방향만 유효)과 2 는 모두 제거한다.
* **색 범위**: scan 의 `colorLimits` 를 읽는다. 0..255 는 그대로, 다른 범위는 한 번 변환하고
  **변환 사실을 manifest 에 기록**한다. 선언이 없으면 8-bit 라고 가정하지 않고 raw 로 보존한다.
* **pose**: registered 추출(기본)은 SOURCE 프레임에 점을 놓고 usable pose 를 요구한다.
  pose 없는 scan 은 `--raw` 를 알려주며 거부한다. `--raw` 는 SCANNER 프레임 클라우드를
  `unregistered` 로 표시해 쓰고, 없는 pose 는 `null` 로 남긴다 — **identity 로 채우지 않는다**.
  `unreadable`/`invalid` pose 는 양쪽 경로 모두 거부한다. 점은 `read_scan_raw` 로 읽는다:
  `read_scan(transform=True)` 는 pye57 의 `rotation_matrix`/`translation` 으로 변환을 만들고,
  그 둘은 pose 없는 scan 에서 identity·0 을 돌려주는 silent fallback 이다.
* **이미지**: spherical·cylindrical embedded image 만 내보낸다. 바이트가 선언한 코덱과
  맞는지 확인하고, 아니면 이름만 바꿔 저장하지 않고 거부한다. pinhole·visual_reference·
  unknown 은 `skipped_images` 에 이유와 함께 기록한다 — 다른 projection 으로 재해석하지 않는다.
  perspective crop·cubemap·ring crop·COLMAP camera·undistortion 은 여기서 하지 않는다.
* **transactional staging**: run 전체를 `.<name>.minegs-partial` 임시 트리에 쓰고, 전부
  성공했을 때만 최종 `work_dir` 로 rename 해서 publish 한다. header preflight 는 잘린 payload 를
  볼 수 없으므로 실패는 12/40 번째 scan 에서 날 수 있고, 그때 scan 12개와 manifest 없는
  디렉토리가 남으면 PLY 개수만 세는 쪽에는 완성된 run 으로 보인다. 이전 트리는 지우지 않고
  옆으로 옮겼다가 마지막 rename 이 실패하면 되돌린다. 어디서 실패하든 `work_dir` 는 그대로다.
  `--overwrite` 도 "좋은 run 을 먼저 지우고 시도한다" 가 아니다. publish 도중(옆으로 옮긴 뒤
  rename 전) 죽으면 다음 실행이 그 백업을 복구한다 — stale 로 보고 지우면 crash 가 곧 데이터
  손실이 된다. 이 보장은 `extract` 의 것이고, deprecated `split` 은 여전히 flat 레이아웃에
  하나씩 쓴다.
* **디렉토리 소유권**: `inventory.json`·`pano_mapping.json`·`extraction_manifest.json`·`scans/`·
  `images/` 와 그 안의 `scan_NNN.*`/`image_NNN.*` 만 이 추출기의 산출물이다. 그 외 파일이
  하나라도 있으면 `--overwrite` 여부와 무관하게 거부한다 — 오타 난 경로가 데이터를 지울 수
  없어야 하고, 옵션의 뜻은 "내 지난 추출을 교체하라" 이지 "이 디렉토리를 비워라" 가 아니다.
  대상 검사는 hash 를 계산하기 전에 먼저 한다.
* **대용량**: pye57 에는 chunked reader 가 없어 scan 을 통째로 읽는다. 선언된 점 개수로 추정
  peak memory 를 note 로 보고하고, `--max-scan-points` 로 fail-closed 할 수 있다. production
  규모 타일링은 PDAL 경로(§22 경계)의 몫이고, PDAL 은 인터페이스 + fail-closed 의존성 검사로
  남는다 (bare pip CI 에 억지로 설치하지 않는다; 실제 PDAL 런타임 검증은 G2/manual).

**Gate (G2)**: 실제 E57 소구간 ingest 성공. 산출물 — scan/station inventory, pose table,
파노라마 매핑 리포트, 추출 point cloud 샘플, provenance.

실데이터로 확인할 것:

```
minegs ingest e57 inventory REAL.e57 --json inventory.json
minegs ingest e57 pano-map  REAL.e57 --json pano_mapping.json
minegs ingest e57 extract   REAL.e57 output/
```

사람이 확인: scan 수 · pose status · scan 배치 · image representation · station/image 매핑 ·
추출된 point cloud · 파노라마 추출 · SOURCE-frame 일관성.

실제 E57 데이터가 없으면 Phase complete 로 선언하지 않는다. 0B.1~0B.3 은 합성·fake·test-time
생성 E57 로만 검증되었고, 사용자의 실제 파일에서 위 세 명령을 돌려 확인하기 전까지
**Phase 0B 는 implementation complete 이지 validated 가 아니다.**

### Phase 0C — Metric Dataset Golden Gate

E57 point cloud · 파노라마 · 카메라 pose · LOCAL_METRIC 좌표가 실제 공간에서 일치함을 검증한다.

**범위**: 파노라마 규약, equirect → ring crop, 합성 intrinsics, station pose → COLMAP,
데이터셋 계약 생성, `init_points.ply`, TLS_GLOBAL ↔ LOCAL_METRIC, Viser 정합, 재투영 오버레이.

**Gate (G2) — Golden Gate**: 실제 TLS 포인트를 파노라마에 재투영했을 때 벽면 edge, 파이프,
케이블, 갱도 경계, 식별 가능한 물체가 영상과 정렬된다. Viser 에서 TLS · camera frustum ·
initialization point 가 동일 공간에서 일치한다.

**이 Gate 가 실패하면 Phase 0D 로 이동하지 않는다.**

### Phase 0D — Local GS Baseline

실제 소구간을 로컬 GPU 에서 gsplat baseline 으로 끝까지 학습한다. 두 단계로 나눈다.

| 하위 | 범위 |
|---|---|
| **0D.1** | LocalRunner, Docker staging, checkpoint path mapping, `--resume-from`, fail-closed missing checkpoint — **implemented + structurally tested** |
| **0D.2** | 실제 GPU baseline, checkpoint 생성, 중단, resume, 이어붙임 검증 — **미수행** |

**Phase 0D entry blocker — Docker `--resume`.** PR #1 검증에서 확인된 문제이며 0D 시작 시
**가장 먼저** 해결한다. 0B.2–0B.3 PR 에서는 고치지 않는다.

> Phase 0D entry blocker: Docker LocalRunner `--resume` must resolve checkpoints using the
> host staging path and translate only the executed command path into the container
> namespace. A requested but missing checkpoint must fail closed; silent restart from
> iteration 0 is forbidden.

현재 Docker 학습 경로는 container 안의 checkpoint 경로를 host 에서 검사하기 때문에 체크포인트를
찾지 못하고, `--resume` 을 요청해도 조용히 iteration 0 부터 다시 시작할 수 있다. 요청한
체크포인트가 없으면 fail-closed 여야 한다 — 재시작은 "조금 느린 resume" 이 아니라 다른 실험이다.

**Phase 0D.1 contract (구현 완료).** Checkpoint discovery is host-side, Docker execution uses
an explicit translated container path, and a requested resume without a valid checkpoint fails
closed. Actual interrupted-GPU-run continuation remains a Phase 0D.2 G2 requirement.

구체적으로:

* Resume 대상은 명시적이다. `--resume` boolean 은 `--resume-from runs/<run_id>` 로 대체했다.
  run_id 는 제출마다 새로 발급되므로 boolean 만으로는 "어느 run 을 이어받는가" 를 말할 수 없고,
  대상이 모호한 resume 은 조용히 fresh run 이 될 수 있다.
* Resume 은 parent 를 수정하지 않고 **child run** 을 만든다. parent 의 checkpoint 디렉토리만
  `:ro` 로 mount 하므로 실패한 child 가 이어받은 run 을 훼손할 수 없다.
* Checkpoint 탐색은 host 에서만 한다 (`minegs/train/runner/resume.py`). 실행 argv 의 경로는
  namespace 변환을 거친 값이다 — docker 는 `/data/resume/<name>`, `--native` 는 host 경로.
  Backend 는 host filesystem 을 보지 않고 `--ckpt` 만 붙인다.
* Latest checkpoint 는 파일명에서 **parse 한 iteration** 으로 고른다. lexical sort 는
  `ckpt_9.pt` 를 `ckpt_10.pt` 뒤에 놓는다. 읽을 수 없는 이름은 건너뛰지 않고 거부한다.
* Compatibility preflight: backend, dataset hash, chunk, 출력 프레임, training-critical profile
  키, 그리고 **staged dataset hash** 가 모두 일치해야 한다. `max_steps` 만 증가를 허용한다.
* `run.json` 의 `resume` 블록이 parent run id/dir, host checkpoint, 실행 경로, sha256,
  iteration 을 기록한다. checkpoint 는 `provenance.source_assets` 에도 들어간다 (§9).
* 모든 preflight 실패는 `subprocess.Popen` **이전**에 일어나고, run directory 가 만들어지기
  전에 일어난다.

**열린 결정 — gsplat v1.5.3 은 training 을 resume 할 수 없다.** 0D.1 구현 중 upstream
`examples/simple_trainer.py` (sha256 `79319e1c…62c05`) 를 직접 확인한 결과:

* `Config.ckpt` 의 docstring 은 *"Path to the .pt files. If provide, it will skip training and
  run evaluation only."* 이다.
* `main()` 은 `if cfg.ckpt is not None:` 이면 `eval`/`render_traj` 만 하고 끝나고, 아니면
  `train()` 을 부른다. 둘은 **상호 배타적**이다.
* `train()` 은 `init_step = 0` 을 무조건 설정하고 checkpoint 를 전혀 읽지 않는다.
* 저장되는 `.pt` 에는 `step` 과 `splats` (+ pose/appearance 모듈) 만 있다. optimizer moment 도,
  densification strategy 상태도 없다.

즉 upstream flag 조합만으로 학습을 이어붙일 방법이 없다. 학습 argv 에 `--ckpt` 를 넣으면
**학습이 아니라 parent 가중치에 대한 evaluation pass** 가 실행되며 아무 오류도 나지 않는다.
그래서 `GsplatBackend` 는 `resume=False` 를 선언하고 `--resume-from` 을 upstream 근거와 함께
거부한다. resume 인프라(위 항목 전부)는 backend 와 무관하게 구현·테스트되어 있고,
resume 가능한 trainer entry point 가 생기면 capability 선언과 flag emission 만 바뀐다.

**PO/architect 결정 필요**: (a) 현 상태 유지 — `--resume-from` 은 gsplat 에 대해 항상
fail-closed, 0D.2 는 중단 없는 단일 실행으로 진행. (b) minegs 소유의 resume 가능한 trainer
entry point 를 별도 phase 로 추가 (checkpoint 로드 + `init_step` 복원 + optimizer/strategy 상태
저장까지 필요 — upstream checkpoint 포맷 확장이 따라온다). 이 결정 전까지 0D.2 의 "중단 후
이어붙임" 항목은 수행할 수 없다.

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
| #2 | E57 inventory + scan/station contract (0B.1) |
| #4 | panorama 매핑 계약 (0B.2) + scan/image 추출 (0B.3) |
| 다음 | Phase 0B real-E57 closeout (실데이터 검증 수정만) |

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
| 매핑 없는 `E57Embedded` 파노라마 | `ContractError` | station↔panorama 추론은 증거가 필요하다 | 해당 없음 (설계) |
| index/개수/파일명/유사도 기반 station↔image 매핑 | 매핑을 만들지 않음 — `unmapped` | 틀린 매핑은 학습·수렴까지 되고 결과만 무의미하다 | 해당 없음 (설계) |
| 하나의 GUID 를 두 scan 이 선언 | `ambiguous` (first match 아님) | 근거가 target 을 특정하지 못한다 | 해당 없음 (설계) |
| 사용자 매핑이 E57 자체 근거와 모순 | `conflict` (덮어쓰기 아님) | 어느 쪽이 틀렸는지 알 수 없다 | 해당 없음 (설계) |
| `vendor`/`generated_by` 없는 vendor manifest | `ContractError` | 손으로 쓴 파일이 confirmed 로 보고되면 안 된다 | 해당 없음 (설계) |
| 파일명 기반 `VendorExport` station 추론 | `ContractError` | 문서화되지 않은 관례는 조용히 깨진다 | 해당 없음 (설계) |
| pose 없는 scan 의 registered 추출 | `E57PoseUnusableError` | SOURCE 배치는 pose 를 요구한다 (`--raw` 로 unregistered 추출) | 해당 없음 (설계) |
| `unreadable`/`invalid` pose 의 추출 | `E57PoseUnusableError` | 깨진 pose 를 identity 로 대체하지 않는다 | 해당 없음 (설계) |
| 길이가 다른 point/attribute 컬럼 | `ContractError` | 한쪽을 잘라 맞추면 속성이 엉뚱한 점에 붙는다 | 해당 없음 (설계) |
| 선언한 코덱과 다른 image blob | `ContractError` / `skipped` | 내용으로 포맷을 추측하지 않는다 | 해당 없음 (설계) |
| 미지원 image representation | `skipped_images` 에 이유 기록 | 다른 projection 으로 재해석하지 않는다 | Phase 0C |
| 이미 추출 산출물이 있는 디렉토리 (`inventory.json` 하나라도) | `ContractError` (`--overwrite` 로 교체) | 두 실행이 섞이면 구분할 수 없다 | 해당 없음 (설계) |
| 추출기 산출물 아닌 파일이 있는 디렉토리 | `ContractError` (`--overwrite` 여도) | 오타 난 경로가 데이터를 지울 수 없어야 한다 | 해당 없음 (설계) |
| 대상 경로가 디렉토리가 아님 | `ContractError` | staging 은 자기 디렉토리를 요구한다 | 해당 없음 (설계) |
| payload 도중 실패한 추출 (`extract`) | 아무것도 publish 하지 않음 (임시 트리 삭제) | 부분 결과가 완성된 run 처럼 보인다 | 해당 없음 (설계) |
| publish 도중 중단된 추출 | 다음 실행이 백업을 복구하고 note 로 보고 | crash 를 조용한 데이터 손실로 바꾸지 않는다 | 해당 없음 (설계) |
| symlink 인 대상 디렉토리 | `ContractError` | rename 은 링크를 갈아치우지 대상을 바꾸지 않는다 | 해당 없음 (설계) |
| 같은 파일을 `--mapping` 과 `--vendor-manifest` 둘 다로 | `ContractError` | 자기 자신과 일치해 confirmed 가 되어버린다 | 해당 없음 (설계) |
| 실패한 `--overwrite` run | 이전 추출 그대로 보존 | 새 run 의 실패가 좋은 run 을 파괴하면 안 된다 | 해당 없음 (설계) |
| 남아 있는 임시 트리 안의 외부 파일 | `ContractError` | 이 도구가 쓰지 않은 것은 지우지 않는다 | 해당 없음 (설계) |
| E57 의 orphan association 을 사용자 매핑이 덮어씀 | `conflict` (scan_id 없음) | target 이 없다고 진술이 없는 것은 아니다 | 명시적 override 정책 (미설계) |
| 결과를 바꾸는 입력이 provenance 에 없음 | E57·매핑 파일·vendor manifest·외부 이미지 전부 hash | 재현할 수 없는 결과는 근거가 아니다 | 해당 없음 (설계) |
| `--resume-from` 의 없는/모호한 checkpoint | `ContractError` (Popen 이전) | 재시작은 느린 resume 이 아니라 다른 실험이다 | 해당 없음 (설계) |
| lexical 로 고른 최신 checkpoint | parse 한 iteration 으로 선택 | `ckpt_9.pt` 가 `ckpt_10.pt` 뒤에 온다 | 해당 없음 (설계) |
| 읽을 수 없는 이름의 `ckpt_*` | `ContractError` | 건너뛰면 더 옛날 것을 resume 하고 성공을 보고한다 | 해당 없음 (설계) |
| multi-rank (distributed) checkpoint | `ContractError` | 한 rank 만 이어받으면 모델의 일부만 복원된다 | Phase 6+ |
| parent run 밖으로 나가는 checkpoint symlink | `ContractError` | provenance 없는 입력 | 해당 없음 (설계) |
| parent 와 다른 dataset/staged/backend/chunk/profile | `ContractError` (무엇이 다른지 명시) | 같은 실험의 연속이 아니다 | 해당 없음 (설계) |
| resume 요청인데 argv 에 `--ckpt` 없음 | `ContractError` (Popen 이전) | parent 를 주장하는 run id 로 fresh run 이 돈다 | 해당 없음 (설계) |
| gsplat 에 `--resume-from` | `ContractError` (upstream 근거 인용) | v1.5.3 은 학습을 이어붙일 수 없다 — §Phase 0D | 열린 결정 (§Phase 0D) |
| `backend_args` 로 들어온 `ckpt` | `ContractError` | 학습이 아니라 evaluation pass 가 조용히 실행된다 | 해당 없음 (설계) |
| 읽을 수 없는/scan 없는 E57 | `E57*` (`ContractError`, exit 2) | 무엇이 문제인지 문장으로 보고 | 해당 없음 (설계) |
| GLUEMAP SfM | `NotYetImplementedError` | 의존성 무거움, 보류 | Phase 3 |
| `pgsr` / `2dgs` / `splatfacto` backend | `NotYetImplementedError` | 미구현 | Phase 4 |
| `inria` backend | `ContractError` | non-commercial 라이선스 | 해당 없음 |
| gsplat trainer 미탐지 | `ContractError` | wheel 에 trainer 없음 | 해당 없음 (docker 로 해결) |
| `raw/` push | `ContractError` | 원본은 로컬에만 (§1.4) | 해당 없음 (설계) |
| `scale.basis` 없는 metric claim | claim 거부 | §4 | 해당 없음 (설계) |
