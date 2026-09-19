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
| 0C | Metric Dataset Golden Gate | **implementation complete, G1 structurally tested** (합성 staging → dataset → 재투영 Golden Gate) — **G2: DEFERRED / NOT VALIDATED** (PO 결정, §0C) |
| 0D | Local GS Baseline | **0D.1 resume safety contract** 및 **0D.2 local GPU baseline execution contract** implemented + structurally tested. **실제 GPU baseline 미실행**. 0D 전체는 **NOT COMPLETE** (§0D) |
| 1 | Metric Surface & Evaluation | **1A metric surface artifact + depth fusion** 및 **1B metric depth rendering: implemented + structurally tested** (§1). 검증된 depth manifest 가 있을 때만 `minegs_render` → `geometry_accuracy`; 외부 depth 는 diagnostic 전용. **실제 GPU rendering 미실행** (CI 에 CUDA·gsplat 없음), TSDF/mesh: NOT IMPLEMENTED, 실측 과학적 검증: NOT VALIDATED |
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

E57 point cloud · 카메라 pose · 이미지 · LOCAL_METRIC 좌표가 실제 공간에서 일치함을 증명한다.
데이터셋을 만드는 것이 목적이 아니라 **frame/camera convention 이 맞다는 것을 증명**하는 것이 목적이다.

```
E57 SOURCE  --explicit SE(3)-->  TLS_GLOBAL  --deterministic origin-->  LOCAL_METRIC
                                                                           |
                                          COLMAP cameras + images + leak-free TLS init
```

**상태**: `Phase 0C implementation complete · G1 structurally tested · G2: DEFERRED / NOT VALIDATED`.
G2 는 Product Owner 결정으로 개발 blocking gate 에서 일시적으로 제외했다. **PASS 로 간주하지 않는다** —
실제 E57 로 프레임·카메라 convention 을 확인하기 전까지 이 저장소는 실데이터 결과를 주장하지 않는다.
실제 E57 은 아직 이 경로로 실행하지 않았다. G2 는 사람이 오버레이와 Viser 를 보는 검사를 포함하며,
자동 점수 하나로 PASS 하지 않는다.

**입력**: Phase 0B.3 staging tree (`inventory.json` · `pano_mapping.json` · `extraction_manifest.json` ·
`scans/` · `images/`). 원본 E57 을 다시 해석하지 않는다. `--raw`(SCANNER 프레임) 트리와 `--no-hash`
트리는 거부한다 — 전자는 배치할 수 없고, 후자는 provenance 를 세울 수 없다.

**구현** (`minegs/dataset/`, `minegs dataset from-e57 STAGING OUT --config build.yaml`):

* **SOURCE 는 TLS_GLOBAL 이 아니다.** `source_frame.mode` 가 `explicit_identity` 또는
  `explicit_transform` 이어야 하고, 둘 다 사용자의 선언이다. 선언 없는 identity 는 없다. 반사(det −1),
  scale, 비강체 행렬은 거부한다. `TLS_GLOBAL` 의 GLOBAL 은 측지 CRS 가 아니라 "여러 scan/camera 를
  하나의 metric survey 좌표계로 표현하는 평가 기준 프레임" 이다 (ARCHITECTURE §3).
* **LOCAL_METRIC 은 translation-only.** `R = I`, scale 1. 원점은 `station_centroid_rounded`
  (station 위치 centroid 를 0.1 m 로 round) 또는 `explicit`. 같은 입력·설정이면 같은 변환이다.
* **Pinhole camera path 가 primary** (실제 Matterport Pro3 E57 = 121 scan × 6 pinhole face, 4096²).
  intrinsics 는 E57 `pinholeRepresentation` 이 선언한 `focalLength / pixelWidth / pixelHeight /
  principalPointX/Y` 에서만 온다 (`fx = focalLength / pixelWidth`). 하나라도 없으면 거부.
  FOV·focal·principal point 를 추측하지 않는다. 파일명(`Skybox 3`)과 index 는 orientation 근거가 아니다.
* **축 규약은 측정한다.** `minegs dataset calibrate-camera STAGING` 이 24 개 axis-aligned proper
  rotation 을 station 자신의 scan 을 station 자신의 image 에 투영해 RGB residual 로 채점하고,
  spatially separated **3 station 이상**(principal axis 기준 early/middle/late, index hard-code 아님)
  에서 같은 후보가 margin 이상으로 이기면 `camera_convention.json` 에 `selected` 로 기록한다.
  station 이 3 개 미만이면 `insufficient`, 불일치면 `inconsistent`, margin 부족이면 `ambiguous` 이고
  builder 는 모두 거부한다. scan 에 RGB 가 없으면 calibration 자체를 거부한다 — 색 없이 threshold 를
  넘길 수 있는 scoring 은 baseline 에 없으므로, 그 경우는 `R_e57cam_from_cam` 을 명시하는 사용자
  책임 경로만 남는다. artifact 는 `source_sha256` 으로 staging tree 에 묶이며, 다른 survey 에서 측정한
  artifact 는 유효해도 거부된다. 규약은 `R_e57cam_from_cam` (E57 image frame ← COLMAP camera frame)
  으로 저장되며, PR #3 의 Matterport 증거 `diag(1,−1,−1)` = `cam(+X,−Y,−Z)` 는 default 가 아니라
  명시적 설정 또는 calibration 결과로만 들어온다.
* **COLMAP pose 는 방향이 이름에 있다**: `T_local_from_cam = T_local_from_tls @ T_tls_from_source @
  T_source_from_e57cam @ R_e57cam_from_cam`.
* **Capture group** = resolved mapping 이 확정한 `image → scan → station` 하나당 `tls_station` 하나.
  `unmapped / ambiguous / orphan / conflict` 이미지가 하나라도 있으면 드롭하지 않고 build 를 거부한다.
* **Split** 은 요청될 때만 (`test_every` 또는 명시 목록). 아무것도 없으면 reconstruction-only.
* **Geometry holdout** 은 centerline 이 있을 때만, 그리고 모든 구간이 centerline 범위 **안에 완전히**
  들어갈 때만 (`s_start ≤ lo < hi ≤ s_end`, 1e-6 m 허용) — 끝을 넘는 구간은 survey 에 없는 chainage 에
  대한 형상 주장을 열어 준다. holdout 구간 point 는 **실제 좌표를 centerline 에 투영해** init 에서 제거하고, `points3D.txt` 는 **같은** leak-free 집합의 subsample 이다. 쓰고 난
  PLY 를 다시 읽어 holdout 안에 point 가 없음을 확인한 뒤에야 publish 한다.
* **Spherical path** 는 기존 `RingCropSpec`/`PanoConvention` 재사용. cylindrical 은 spherical 로
  처리하지 않고 거부한다.
* **Provenance 는 실제로 소비한 바이트를 기술한다.** 0C 가 읽는 scan/image 는 소비 시점에 다시 해싱해
  extractor 가 기록한 digest 와 비교하고(불일치 = 거부, 파일당 한 번만 해싱), 그 검증된 digest 가
  `source_assets` 에 들어간다. source E57 · 0B artifact 3개 · mapping input · camera convention ·
  centerline · config 파일도 전부 포함되며, 결과를 바꾸는 설정 전체가 `config_hash` 에 들어간다.
  `pano_mapping.json` 과 `extraction_manifest.json` 안의 mapping report 가 서로 모순이면 합치지
  않고 거부한다. 해석된 설정은 `dataset/build_config.json` 에 남는다.
* **Transactional**: `.<name>.minegs-partial` 에서 build → `Manifest.load_dataset` →
  `consistency_issues` → `judge` → §27 수치 검사 → 그 뒤에만 rename. `--overwrite` 는 이 도구가 쓴
  dataset(`manifest.json` + `build_config.json`, 그 외 파일 없음) 만 교체한다 — 오타 난 경로가 남의
  디렉토리를 지울 수 없어야 한다 (0B 와 같은 규칙).

**Golden Gate** (`minegs dataset golden-gate DATASET --staging STAGING --out DIR`): sampled station 의 scan 을
각 image 에 재투영한 depth/TLS-RGB overlay, 24 규약 재채점과 margin, 좌표 범위, roundtrip, holdout 누수
재검사, `report.json`, Viser 용 LOCAL_METRIC TLS sample. staging tree 는 같은 E57 이라는 것으로는
부족하고 — artifact·scan·image 가 dataset provenance 에 기록된 digest 와 **정확히** 일치해야 한다
(재추출·재매핑한 tree 는 다른 입력이다). `structural_result` 는 수치가 말하는 것이고 실패면 report 와
overlay 를 먼저 쓴 뒤 non-zero 로 종료한다. `real_data_validation_status` 는 항상
`pending_human_inspection` 이다.

**Gate (G2) — Golden Gate** (실데이터): 실제 0B production path (`inventory → pano-map → extract`) 로 만든
staging 에서 dataset 을 만들고, early/middle/late 3 station 이상에서 (Matterport 라면 face 여러 장) 벽면
edge · 갱도 경계 · 파이프/케이블 · 표지/물체 · 천장/바닥 방향 · scanner 위치 · 좌우 일관성이 overlay 와
정렬되고, Viser 에서 TLS ≈ init cloud, frustum 이 실제 station 위에, 시선이 갱도 영상과 맞는다.
PR #3 의 one-off exporter 출력은 G2 증거가 아니다.

**이 Gate 가 실패하면 Phase 0D.2 로 이동하지 않는다.** 잘못된 convention 으로 GPU 를 돌리는 것은 시간 낭비다.

### Phase 0D — Local GS Baseline

실제 소구간을 로컬 GPU 에서 gsplat baseline 으로 끝까지 학습한다.

| 하위 | 범위 | 상태 |
|---|---|---|
| **0D.1** | resume safety contract — silent restart 경로 제거, gsplat resume 능력 독립 확인, fail-closed 거부 | **implemented + structurally tested** |
| **0D.2** | 실제 GPU baseline (중단 없는 단일 학습) | **execution contract implemented + structurally tested — 실제 GPU run 미실행** |
| **0D.3** | MineGS 소유 resumable trainer & checkpoint contract | **보류 — 필요할 때만** |

**Phase 0D entry blocker — Docker `--resume`.** PR #1 검증에서 확인된 문제이며 0D 시작 시
**가장 먼저** 해결한다. 0B.2–0B.3 PR 에서는 고치지 않는다.

> Phase 0D entry blocker: Docker LocalRunner `--resume` must resolve checkpoints using the
> host staging path and translate only the executed command path into the container
> namespace. A requested but missing checkpoint must fail closed; silent restart from
> iteration 0 is forbidden.

당시 Docker 학습 경로는 container 안의 checkpoint 경로를 host 에서 검사했기 때문에 체크포인트를
찾지 못하고, `--resume` 을 요청해도 조용히 iteration 0 부터 다시 시작할 수 있었다. 재시작은
"조금 느린 resume" 이 아니라 다른 실험이다.

#### Phase 0D.1 — resume safety contract (완료)

Phase 0D.1 resume safety contract: **implemented + structurally tested.**
gsplat v1.5.3 training resume is unsupported and fails closed.

0D.1 이 실제로 보장하는 것은 다음 여섯 가지다.

* 이전의 silent restart 경로를 제거했다. `if resume: glob(out_dir/"ckpts")` 는 사라졌다.
* gsplat 의 training resume 능력을 upstream 소스로 **독립 확인**했다 (아래).
* `GsplatBackend` 는 `resume=False` 를 선언한다.
* resume 요청은 fail closed — `--resume-from` 은 `Runner.prepare` 에서 거부되고, 이는
  trainer 실행 이전이자 **child run directory 생성 이전**이다.
* raw `--ckpt` 가 학습 run 을 eval-only 로 바꿀 수 없다. `backend_args` 를 통한 경로도 포함해
  조립된 argv 를 스캔해서 거부하며, **철자에 의존하지 않는다** — key 와 argv token 을 같은
  규칙으로 정규화하므로 `--ckpt`, `-ckpt`, `ckpt `, `CKPT` 가 모두 같은 거부에 걸린다. 한 가지
  철자만 잡는 거부는 거부가 아니다.
* fresh command 에는 checkpoint 인자가 없다. run directory 안에 우연히 checkpoint 가 있어도
  자동 resume 하지 않는다 — resume 은 요청되는 것이지 추론되는 것이 아니다.

**주장하지 않는 것**: host/container resume 실행은 production capability 로 검증되지 않았다.
generic LocalRunner resume 은 구현되어 있지 않다. resume 가능한 backend 가 capability flag 만
켜면 되는 상태도 아니다 — checkpoint contract 자체가 아직 없다 (0D.3).

`--resume-from` 은 미래 인터페이스를 명확히 하기 위해 CLI 에 남아 있으나, 오늘은 항상 실패한다.

**gsplat v1.5.3 은 training 을 resume 할 수 없다** — upstream `examples/simple_trainer.py`
(sha256 `79319e1cd7404e4d1ba0c425634235c39e6054f0643b904a01feea6179462c05`, pinned image 가
clone 하는 것과 동일) 확인 결과:

* `Config.ckpt` docstring: *"Path to the .pt files. If provide, it will skip training and run
  evaluation only."*
* `main()` 은 `if cfg.ckpt is not None:` 이면 `eval`/`render_traj` 만 하고 끝나고, 아니면
  `train()` 을 부른다. 둘은 **상호 배타적**이다.
* `train()` 은 `init_step = 0` 을 무조건 설정하고 checkpoint 를 전혀 읽지 않는다.
* 저장되는 `.pt` 에는 `step` 과 `splats` (+ pose/appearance 모듈) 만 있다. optimizer moment 도,
  densification strategy 상태도 없다.

학습 argv 에 `--ckpt` 를 넣으면 학습이 아니라 parent 가중치에 대한 **evaluation pass** 가
실행되며 아무 오류도 나지 않는다. 그래서 거부한다.

**PO/architect 결정 (채택: (a)).** gsplat v1.5.3 용 training resume 은 구현하지 않는다.
0D.2 는 중단 없는 단일 학습으로 첫 실제 GPU baseline 을 검증한다. 진짜 중단-이어붙임은
Mine-3DGS 가 완전한 training state 를 복원할 수 있는 trainer/checkpoint contract 를 소유할
때까지 보류한다 (0D.3).

#### Phase 0D.2 — 실제 GPU baseline (미수행)

**진입 조건**: Phase 0C G2 는 DEFERRED 이므로 0D.2 **개발**은 진행한다. 다만 프레임·카메라
convention 이 실제 데이터에서 확인되지 않았으므로, 실제 신풍갱 데이터로 돌린 결과는 scientific
validation 으로 주장하지 않는다 — 합성 데이터 위의 execution contract 검증이다.

**현재 상태**: execution contract 는 구현·구조 검증 완료. `LocalRunner` 는 trainer 의 exit code
만으로 성공을 기록하지 않고, checkpoint·final PLY·step 진행·finite 좌표·frame invariant 를 모두
확인한 뒤에만 `SUCCEEDED` 를 publish 한다. **실제 GPU run 은 아직 수행하지 않았다.**

**범위**: pinned GPU docker image, gsplat v1.5.3 executable contract, LocalRunner,
light profile, staging, checkpoint/output, LOCAL_METRIC 출력 정규화 계약.

**원칙**: `normalize_world_space=false`, BACKEND_INTERNAL = LOCAL_METRIC, **GPU 정확히 1개**.

gsplat v1.5.3 은 `torch.cuda.device_count()` 만 보고 distributed 로 전환하며 (`gsplat/distributed.py`
`cli()`), 그 모드에서 PLY 는 rank 로 구분되지 않는다 (`ply/point_cloud_{step}.ply`, rank guard 없음).
따라서 baseline 은 docker `--gpus device=<n>` 과 `CUDA_VISIBLE_DEVICES` 양쪽으로 한 장에 고정한다.
multi-GPU 는 Phase 0D.2 범위가 아니다.

**detach 불가**: run 은 끝날 때 검증되고, 중간에 떠난 run 을 나중에 finalize 하는 경로가 없다.
그래서 `minegs train run` 은 항상 끝까지 기다린다 — detach 는 finalize-on-inspection 이 생긴 뒤의 일이다.

**Gate** — 중단 없는 단일 학습으로 다음 10 항목을 모두 확인한다. 1·3·4·5·6·7·8·9 는
`LocalRunner` 가 run 마다 자동으로 확인하고 `run.json` 에 기록한다 (아래 fail-closed 표 참조);
하나라도 확인되지 않으면 그 run 은 `SUCCEEDED` 가 되지 않는다. 2 와 10 은 사람이 한다.

1. pinned GPU image 를 빌드/사용한다 (`image@sha256:...`).
2. 작은 실제 또는 합성 갱도 구간으로 학습 job 을 실행한다.
3. CUDA / runtime 을 확인한다.
4. checkpoint 생성을 확인한다.
5. 예상한 training iteration 진행을 확인한다.
6. 생성된 PLY 를 확인한다.
7. BACKEND_INTERNAL → LOCAL_METRIC 계약을 확인한다.
8. `run.json` / provenance 를 확인한다.
9. runtime 과 GPU memory 를 기록한다.
10. 시각적·과학적 sanity check 를 수행한다.

기존의 "중단 → resume → 이어붙임" gate 는 0D.3 으로 연기한다.

GPU smoke 를 통과하지 않으면 Phase 0D 완료라고 하지 않는다.

#### Phase 0D.3 — MineGS-owned resumable trainer & checkpoint contract (보류)

진짜 resume 이 필요해진 경우에만 착수한다. 구현 **이전에** checkpoint 가 최소한 다음을 담도록
정의해야 한다.

* model / splats
* optimizer states
* LR scheduler states
* densification strategy state
* current training step
* pose / appearance / bilateral optimization state (활성화된 경우)
* 관련 RNG state
* checkpoint schema version
* trainer / backend version

Cross-version checkpoint migration 은 명시적으로 설계되기 전까지 지원하지 않으며 fail closed
한다. 이 계약이 정의되기 전에 checkpoint 이름 규칙(`ckpt_<iteration>.pt`,
`ckpt_<iteration>_rank<n>.pt`)이나 `--ckpt` 같은 gsplat 고유 가정 위에 generic resume API 를
고정하지 않는다.

### Phase 1 — Metric Surface & Evaluation

3DGS representation 을 측량 가능한 surface representation 으로 변환한다.

**범위**: rendered depth, surface point 생성, mesh/TSDF, 양방향 형상 지표, geometry holdout,
중심선 단면, A(s), 체적 적분, 설계 대비, missing-data 처리.

**원칙**: 가우시안 중심을 TLS 와 직접 비교하지 않는다. 항상 GS → depth/mesh/surface → 평가.

#### Phase 1A — metric surface artifact & depth fusion (완료)

Phase 1A metric surface artifact + depth fusion: **implemented + structurally tested.**

경계 하나만 닫는다: surface 는 **유도된 artifact** 이고, claim 을 담는 geometry 평가는 그 artifact
를 요구한다.

* `minegs eval surface-depth <depth_dir> <dataset_dir> --run-dir <run_dir> [--out] [--stride]
  [--max-depth]` — depth map(`<image stem>.npy`, 카메라 z 미터) 을 dataset 의 `sparse/0` pose 로
  역투영해 `runs/<run_id>/surface/depth_v001/{surface.json, surface_points.ply}` 를 publish 한다.
  Gaussian PLY(`point_cloud/`) 와 디렉터리를 섞지 않는다.
* `SurfaceRecord` (`minegs/eval/surface/models.py`, schema 1.0) — `surface_id · dataset_id ·
  dataset_hash · run_id · method · depth_source · frame · unit · point_file · point_sha256 ·
  point_count · depth_map_count · depth_sha256 · parameters · provenance`.
  `frame == LOCAL_METRIC`, `unit == m` 은 모델 수준 invariant 이고, `method` 는
  `depth_backprojection` 하나다. **surface.json 은 정확도를 주장하지 않는다.**
* **record 는 surface 가 아니다.** `check_surface` 는 `point_file` 을 다시 읽어
  `point_sha256`·point count·frame·유한성을 확인한다. 이 검증이 없으면 경계가 "surface 인가"
  가 아니라 "surface.json 이라는 파일이 옆에 있는가" 를 묻게 되고, Gaussian PLY 에 손으로 쓴
  record 를 붙이면 claim 경로로 그대로 들어온다.
* **surface 라는 것과 accuracy claim 을 할 수 있다는 것은 다르다.** `depth_source` 가 그 경계다.
  Phase 1A 가 받을 수 있는 depth 는 전부 `external_unverified`: 디렉터리의 `.npy` 는 기록된 run
  에서 나왔다는 증거를 갖고 있지 않고, 같은 depth 를 이 dataset 의 아무 성공한 run 에 붙여도
  같은 artifact 가 나온다. 따라서 **Phase 1A 에서는 `geometry_accuracy` 에 도달할 수 없고**
  `--diagnostic` 로만 수치를 낸다. `minegs_render` 는 Phase 1B renderer 가 run id·dataset hash·
  depth 별 digest 를 함께 낼 때 쓰는 값이며, gate 는 지금 써 둔다 — 나중에 검사를 추가해서가
  아니라 증거를 만들어서 여는 구조다.
* **Fail closed**: run 이 `succeeded` 가 아니거나 `dataset_id`/`dataset_hash` 가 다르거나
  `frame_of_outputs != LOCAL_METRIC` 이면 거부. 카메라 view 수 ≠ depth map 수 이면 거부
  (조용히 빠진 view = completeness·단면·체적의 계통 구멍). camera view 에 해당하지 않는
  depth map 이 섞여 있어도 거부한다. depth 해상도가 camera intrinsics 해상도와 다르면 거부한다 —
  full-resolution `fx,fy,cx,cy` 로 half-resolution grid 를 역투영하면 모든 ray 가 조용히 휘고,
  surface·record·평가가 전부 성공한 채로 기하만 틀린다. publish 는
  `.<name>.minegs-partial` → rename 으로 원자적이다.
* **Geometry gate**

  | 입력 | `--diagnostic` 없이 | `--diagnostic` |
  |---|---|---|
  | 원시 PLY | `ContractError` (exit 2) — "Gaussian centres are not surfaces" | 경고 + `geometry_diagnostic` |
  | surface artifact, `external_unverified` | `ContractError` (exit 2) — depth 가 run 과 묶여 있지 않음 | 경고 + `geometry_diagnostic` |
  | surface artifact, `minegs_render` | `geometry_accuracy` | `geometry_accuracy` |

* **구조적 검증만**: `tests/test_surface.py` T1–T10 (평면 역투영 · 무효 depth 필터 · 누락 실패 ·
  artifact publication · dataset/run mismatch · claim 경로의 원시 PLY 거부 · diagnostic 유지 ·
  실제 builder artifact 의 evaluator 도달과 claim demotion · record ≠ surface (digest 검증) ·
  depth 해상도 불일치). GPU 도, 렌더된 depth 도, 실측 데이터도 쓰지 않는다.
* **알려진 한계**: record 는 생산자의 선언이다. `point_sha256` 은 "이 record 는 저 바이트에 대한
  것" 을 보장하지만, 손으로 `depth_source: minegs_render` 를 적고 digest 까지 맞춘 파일을 막지는
  못한다. 이 경계는 사고를 막기 위한 것이고, 서명이 아니다.

#### Phase 1B — trained run → rendered metric depth (구현, 실제 GPU 미실행)

Phase 1B metric depth rendering: **implemented and structurally tested. Real GPU rendering and
scientific validation remain pending.**

`minegs eval render-depth <run_dir> <dataset_dir> [--out] [--min-alpha]` →
`render_depths(run_dir, dataset_dir, out_dir)` 가 `runs/<run_id>/depth/` 에
`<image stem>.npy` 와 `depth_manifest.json` 을 원자적으로 publish 한다.

* **책임 분리가 신뢰의 근거다.** renderer adapter(`DepthRenderer`)는 rasterizer 와 그 요구사항만
  갖고 배열을 낸다. 계약 검증(run 상태·dataset identity·metric frame·checkpoint identity·view별
  해상도·coverage·digest·publication)은 전부 orchestrator(`render_depths`)에 있다. 그래서 GPU 없는
  머신에서 adapter 를 대체해도 검사는 하나도 우회되지 않는다.
* **`GsplatDepthRenderer`**: `render_mode="ED"` — alpha 정규화된 expected ray termination depth,
  카메라 +z 방향으로 `backproject_depth` 가 먹는 것과 같은 양. metric 인 이유는 baseline 이
  `normalize_world_space` 를 거부해 BACKEND_INTERNAL = LOCAL_METRIC 이기 때문이고, 이는 가정이
  아니라 run 마다 `T_local_from_internal` 이 항등인지로 확인한다.
* **NaN 정책**: ray 가 `--min-alpha`(기본 0.5)만큼 불투명도를 쌓지 못하면 거리값이 없으므로 NaN.
  0 은 렌즈에 표면을 만들고 far plane 은 없는 벽을 만든다.
* **`DepthManifest`** (`models.py`, schema 1.0): `manifest_id · run_id · dataset_id ·
  dataset_hash · backend · checkpoint{file, sha256, step} · renderer{name, version, settings} ·
  frame · unit · depths[{image_id, camera_id, image_name, file, width, height, sha256,
  valid_ratio, min_m, max_m}] · provenance`.
* **승격 (§1A 연결)**: `build_depth_surface` 는 depth 디렉터리의 manifest 를 **검증한다** —
  `verify_depth_manifest` 가 run id, dataset id/hash, checkpoint identity(file·step, 파일이
  남아 있으면 digest), view 집합, camera 해상도, 파일별 digest 를 모두 대조해야
  `depth_source = minegs_render` 다. 하나라도 어긋나면 강등이 아니라 거부한다 (어긋났다는 것은
  무언가 움직였다는 뜻이고, 그때 조용히 계속하는 것이 가장 나쁘다). manifest 가 없으면
  `external_unverified` — PR #9 에서 남겨 둔 문자열 우회는 production path 에서 닫혔다.
* **검증한 bytes 와 소비한 bytes 는 같아야 한다**: fuser 는 manifest 의 `file` 을 읽지 않고
  naming contract(`<image stem>.npy`)로 파일을 찾으므로, verifier 는 `entry.file` 이 그 이름과
  같은지를 먼저 강제한다. 아니면 한 파일의 digest 를 검사하고 다른 파일을 역투영하게 된다.
* **renderer 가 재현하지 않는 run 은 거부**: `RENDER_CRITICAL_OPTIONS` = `pose_opt`(학습 중
  카메라 pose 갱신 → dataset pose 는 모델이 맞춰진 pose 가 아님), `antialiased`(opacity 누적
  방식이 달라 expected depth 가 달라지고 equivalence 미측정). 증인은 넷 — 실행된 argv,
  `backend_out/cfg.yml`, profile 의 requests, 그리고 `backend_args` — 이고 **하나라도** 걸리면
  거부한다. 추가로 checkpoint 에 `pose_adjust` 가 있으면 `check_checkpoint_blob` 이 그것만으로
  거부한다. `normalize_world_space` 를 재구현 대신 거부한 것과 같은 논리다.
* **`backend_args` 는 allowlist 로 읽는다** (`RENDER_NEUTRAL_BACKEND_ARGS`). `build_command` 가
  `backend_args` 를 trainer 에 그대로 전달하므로 나쁜 이름 목록으로는 `camera_model`·`with_ut`·
  `far_plane` 처럼 boolean 도 아니고 이름도 모르는 옵션을 잡을 수 없다. argv 는 docker 플래그가
  섞여 있어 이 방식이 불가능하지만 `backend_args` 는 순수한 gsplat 네임스페이스라 가능하다.
  reasoned about 하지 않은 키는 거부한다.
* **승격 시 재확인**: manifest 는 metric frame 도 재현 가능 여부도 기록하지 않으므로
  `build_depth_surface` 가 promotion 직전 `require_metric_outputs` 와
  `require_reproducible_render` 를 run record 에 대해 다시 돌린다. 가드가 없던 빌드가 만든
  depth 가 manifest 존재만으로 승격되는 경로를 닫는다.
* **reference cloud 의 frame 도 거부 대상**: `eval geometry --tls-ply` 가 `TLS_GLOBAL` 이
  아니면 (`UNKNOWN` 포함) claim 경로에서 거부한다. 예전에는 경고 한 줄이었고 JSON 에는 남지
  않았다. `dataset/init_points.ply`(LOCAL_METRIC) 는 `raw/tls_full.ply` 바로 옆에 있어서 흔한
  실수이고, 그 비교는 두 표면이 아니라 두 좌표계의 거리를 잰다. 예측 쪽은 `_to_tls` 가 이미
  같은 이유로 거부하고 있었다.
* **checkpoint 는 restricted unpickler 로 읽는다** (`weights_only=True`). run artifact 는 GPU
  호스트에서 가져오는 것이고, 기존 방식은 `check_checkpoint_blob` 이 보기 *전에* 임의 코드를
  실행할 수 있었다. upstream 은 tensor·dict·int 만 저장하므로 정상 checkpoint 는 모두 로드된다.
* **학습에 쓴 view 를 기록하고 검증한다**: depth 는 dataset 의 모든 view 에 대해 렌더되지만
  `max_images` 프로파일은 일부만 학습한다. manifest 의 `staged` 가 run.json 의 것을 복사해
  두 artifact 가 구별되게 하고, promotion 때 run 과 다시 대조한다 (렌더가 틀렸다는 뜻은 아니다).
* **renderer 도 신원을 검증한다**: `render_depths(renderer=...)` 주입 seam 은 GPU 없이 계약을
  테스트하기 위한 것이므로, manifest 의 `renderer.name` 이 이 빌드가 실제로 제공하는 renderer
  (`known_renderer_names()`) 가 아니면 승격하지 않는다. 기록만 하고 대조하지 않는 필드를 남기지
  않는다는 같은 원칙이다.
* **rasterizer 노브를 기본값에 기대지 않는다**: 거부 논리가 "이 renderer 는 classic mode 이고
  pinhole 로 투영한다" 를 근거로 삼으므로, 호출이 `rasterize_mode="classic"` 과
  `camera_model="pinhole"` 을 명시하고 `settings()` 에 기록한다.
* **`--no-holdout-only` 는 claim 을 내리지 않는다**: `geometry_accuracy` 는 정의상 holdout TLS
  에 대한 주장(`Claim` docstring)이고, judge 가 그것을 허용한 이유도 그 구간이 초기화에서
  제외되었기 때문이다. holdout mask 를 끄면 run 이 학습에 쓴 형상을 다시 재게 되므로 경고와
  함께 `geometry_diagnostic` 으로 강등한다.
* **camera model**: `RENDERABLE_CAMERA_MODELS = {PINHOLE, SIMPLE_PINHOLE}`. 왜곡 모델은
  pinhole 로 투영되어 모든 ray 가 조용히 틀어지므로 거부한다.
* **fail closed, CPU fallback 없음**: run != succeeded · dataset id/hash 불일치 · checkpoint
  미기록/부재 · `T_local_from_internal` 비항등 · backend 가 `depth_render` 미선언 · 지원하지 않는
  backend · torch/gsplat 부재 · CUDA 부재 · view 누락/중복/유령 · image stem 충돌 · 해상도
  불일치 · Inf/음수 depth · 전 픽셀 empty · 출력 디렉터리 존재 · render-critical 옵션 ·
  왜곡 camera model.
* **실행되지 않은 부분**: `GsplatDepthRenderer.render` 는 gsplat rasterization API 에 맞춰 작성했고
  이 저장소에서 **한 번도 실행된 적이 없다** — CI 에 CUDA 도 gsplat 도 없다. 주변 계약은 전부
  테스트되지만 rasterizer 호출은 아니다. `tests/test_depth_render.py` 는 adapter 를 대체하되
  validation path 를 우회하지 않으며, 실제 adapter 에 대해서는 "GPU 없이는 절대 돌지 않는다" 만
  단언한다.

TSDF(`eval/surface/tsdf.py`) 와 mesh 재구성은 여전히 미구현이고, Open3D 는 CI 의존성이 아니다.

Phase 1A/1B 는 **artifact 경계와 증거 경로**를 닫았을 뿐이다. 3DGS 형상이 정확하다거나 surface 가
과학적으로 검증되었다는 주장은 하지 않는다.

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
checkpoint, **exit-code 기반 status**, artifact 동기화 복귀, provenance, GPU 타입 기록.
resume 은 Phase 6 범위가 아니다 — runner 와 무관하게 Phase 0D.3 이다 (§Phase 0D).

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
| TSDF / mesh 추출 | `NotYetImplementedError` | 미구현 | Phase 1 후속 |
| CUDA·gsplat 없이 `eval render-depth` | `NoGpuError` / `MissingDependencyError` (exit 4) | rasterizer 는 CUDA 전용이고 CPU fallback 은 없다 | 해당 없음 (설계) |
| `T_local_from_internal` 이 항등이 아닌 run 의 depth 렌더 | `ContractError` (exit 2) | backend 단위가 미터라고 보장할 수 없다 | 해당 없음 (설계) |
| 검증에 실패하는 depth manifest | `ContractError` (exit 2) | 강등이 아니라 거부 — 무언가 움직였다는 신호다 | 해당 없음 (설계) |
| `pose_opt`/`antialiased` run 의 depth 렌더 | `ContractError` (exit 2) | renderer 가 재현하지 않는 설정이다 — equivalence 미검증 재현은 증거가 아니다 | pose 복원·mode 재현을 실제로 검증한 뒤 |
| 왜곡 camera model 의 depth 렌더 | `ContractError` (exit 2) | pinhole 로 투영되어 모든 ray 가 조용히 틀어진다 | 해당 없음 (설계) |
| manifest 의 `file` 이 naming contract 와 다름 | `ContractError` (exit 2) | 검사한 파일과 역투영할 파일이 달라진다 | 해당 없음 (설계) |
| reasoned about 하지 않은 `backend_args` 키의 depth 렌더 | `ContractError` (exit 2) | trainer 에 그대로 전달되는 옵션이 투영/frustum 을 바꿀 수 있다 | 해당 항목이 neutral 임을 보인 뒤 |
| `--no-holdout-only` 에 `geometry_accuracy` | claim 을 `geometry_diagnostic` 으로 강등 + 경고 | 학습에 쓴 형상을 다시 재는 수치다 | 해당 없음 (설계) |
| holdout 구간에 점이 없는 reference/prediction | `ContractError` (exit 2) | 빈 cloud 에 대한 accuracy/completeness 는 수치가 아니라 입력 누락이다 | 해당 없음 (설계) |
| run 의 최종 checkpoint 가 아닌 manifest | `ContractError` (exit 2) | 다른 모델을 기술하면서 나머지 검사를 통과한다 | 해당 없음 (설계) |
| claim 을 담는 `eval geometry` 에 원시 PLY | `ContractError` (exit 2) | 가우시안 중심은 표면이 아니다 (§1A) | 해당 없음 (설계) |
| claim 을 담는 `eval geometry` 에 `external_unverified` surface | `ContractError` (exit 2) | 외부 depth 는 기록된 run 과 묶여 있지 않다 | Phase 1B (`minegs_render`) |
| surface.json 과 내용이 다른 `point_file` | `ContractError` (exit 2) | record 는 surface 가 아니다 | 해당 없음 (설계) |
| camera 해상도와 다른 depth map | `ContractError` (exit 2) | intrinsics 불일치는 조용히 기하를 틀리게 한다 | 해당 없음 (설계) |
| 카메라 view 보다 적은 depth map 으로 surface 생성 | `ContractError` (exit 2) | 조용한 구멍은 미복원 형상과 구별되지 않는다 | 해당 없음 (설계) |
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
| 미지원 image representation (visual-reference, unknown) | `skipped_images` 에 이유 기록 | 다른 projection 으로 재해석하지 않는다 | 해당 없음 (설계) |
| 선언 없는 SOURCE → TLS_GLOBAL | build config 검증 실패 | identity 도 사용자의 선언이어야 한다 (0C §5) | 해당 없음 (설계) |
| 반사·scale·비강체 `T_tls_from_source` | `ContractError` | 거울 프레임은 모든 카메라의 좌우를 뒤집는다 | 해당 없음 (설계) |
| E57 pinhole intrinsic 누락 | `ContractError` | FOV/focal/principal point 를 추측하지 않는다 | 해당 없음 (설계) |
| image pose 없음/비단위 quaternion | `ContractError` | 파일명·index 는 orientation 근거가 아니다 | 해당 없음 (설계) |
| camera convention 미선언·미측정 | build config 검증 실패 | 추측한 축 규약으로 학습하면 수렴까지 하고 결과만 무의미하다 | 해당 없음 (설계) |
| calibration `ambiguous`/`inconsistent` | `ContractError` | 근거가 하나를 특정하지 못한다 | 해당 없음 (설계) |
| `unmapped/ambiguous/orphan/conflict` 이미지가 dataset 입력에 존재 | `ContractError` (드롭 아님) | 조용히 빠진 face 는 아무도 모른다 | 해당 없음 (설계) |
| centerline 없는 geometry holdout | build config 검증 실패 | scan 순서는 chainage 가 아니다 | 해당 없음 (설계) |
| cylindrical 을 spherical 로 | `ContractError` | 검증 없는 projection 재해석 금지 | 별도 설계 |
| `--raw` / `--no-hash` staging tree 로 dataset build | `ContractError` | 배치 불가 / provenance 불가 | 해당 없음 (설계) |
| init PLY 에 holdout point 잔존 | publish 거부 | manifest 필드가 아니라 실제 PLY 로 증명한다 | 해당 없음 (설계) |
| 추출 후 바뀐 scan/image 바이트 | `ContractError` (소비 시점 재해싱) | provenance 는 실제 소비한 바이트를 기술한다 | 해당 없음 (설계) |
| 서로 모순인 `pano_mapping.json` 과 carried mapping report | `ContractError` | 다른 이야기를 하는 두 artifact 는 합치지 않는다 | 해당 없음 (설계) |
| 다른 source 에서 측정한 calibration artifact | `ContractError` | 다른 survey 에 대한 증거다 | 해당 없음 (설계) |
| station 3 개 미만의 calibration | `insufficient` / `--stations < 3` 거부 | 한 station 의 규약은 규약이 아니다 | 해당 없음 (설계) |
| RGB 없는 scan 의 자동 calibration | `ContractError` | threshold 를 넘길 수 있는 색 없는 scoring 이 없다 | 명시적 규약 (사용자 책임) |
| centerline 끝을 넘는 holdout 구간 | `ContractError` | survey 에 없는 chainage 에 대한 주장을 연다 | 해당 없음 (설계) |
| provenance 와 다른 staging tree 로 golden-gate | `ContractError` | 같은 E57 ≠ 같은 입력 | 해당 없음 (설계) |
| golden-gate `structural_result=fail` | report/overlay 기록 후 exit 2 | 실패한 gate 가 성공처럼 끝나면 안 된다 | 해당 없음 (설계) |
| `--overwrite` 대상이 이 도구의 dataset 이 아니거나 외부 파일을 포함 | `ContractError` | 오타 난 경로가 데이터를 지울 수 없어야 한다 | 해당 없음 (설계) |
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
| gsplat 에 `--resume-from` | `ContractError` (upstream 근거 인용, run directory 생성 이전) | v1.5.3 은 학습을 이어붙일 수 없다 — §Phase 0D | Phase 0D.3 (보류) |
| resume=true 를 선언하는 backend 에 `--resume-from` | `NotYetImplementedError` (Phase 0D.3) | 완전한 training state 를 복원하는 checkpoint contract 가 아직 없다 | Phase 0D.3 (보류) |
| `backend_args` 로 들어온 `ckpt` (모든 철자) | `ContractError` | 학습이 아니라 evaluation pass 가 조용히 실행된다 | 해당 없음 (설계) |
| trainer exit 0 인데 checkpoint 없음 | `FAILED` + `failure_reason` | 아무것도 쓰지 않고 끝난 프로세스도 exit 0 이다 | 해당 없음 (설계) |
| trainer exit 0 인데 최종 PLY 없음 | `FAILED` + `failure_reason` | 위와 같다 | 해당 없음 (설계) |
| 출력 gaussian 좌표에 non-finite | `FAILED` | 학습이 발산했다 | 해당 없음 (설계) |
| `max_steps` 에 못 미친 step 에서 종료 | `FAILED` | 짧게 끝난 run 은 다른 실험이다 | 해당 없음 (설계) |
| trainer `cfg.yml` 이 `normalize_world_space: true` | `FAILED` | upstream 기본값이 true 다 — 출력이 metre 가 아니게 된다 | 해당 없음 (설계) |
| 출력 span 이 init span 대비 20배 밖 | `FAILED` | scale 이 바뀌었다는 정황 (보조 신호) | 해당 없음 (설계) |
| CUDA 없음 | `NoGpuError` (exit 4) | CPU 학습은 느린 GPU 학습이 아니라 다른 실험이다 | 해당 없음 (설계) |
| 이미 사용된 run directory | `ContractError` | 이전 run 의 artifact 를 이번 run 의 증거로 읽게 된다 | 해당 없음 (설계) |
| `runner.gpus` 가 GPU 를 2개 이상 노출 | `ContractError` | gsplat 이 device count 만 보고 distributed 로 가고, PLY 는 rank 로 구분되지 않아 서로 덮어쓴다 | multi-GPU 는 범위 밖 |
| container 안의 torch 가 CUDA 를 못 봄 | `ContractError` | 학습할 런타임이 GPU 를 못 보면 GPU baseline 이 아니다 | 해당 없음 (설계) |
| option name 이 될 수 없는 `backend_args` key (내부 공백) | `ContractError` | flag 로 넘길 수 없고, 출력된 command 에서는 인자 두 개로 읽힌다 | 해당 없음 (설계) |
| 읽을 수 없는/scan 없는 E57 | `E57*` (`ContractError`, exit 2) | 무엇이 문제인지 문장으로 보고 | 해당 없음 (설계) |
| GLUEMAP SfM | `NotYetImplementedError` | 의존성 무거움, 보류 | Phase 3 |
| `pgsr` / `2dgs` / `splatfacto` backend | `NotYetImplementedError` | 미구현 | Phase 4 |
| `inria` backend | `ContractError` | non-commercial 라이선스 | 해당 없음 |
| gsplat trainer 미탐지 | `ContractError` | wheel 에 trainer 없음 | 해당 없음 (docker 로 해결) |
| `raw/` push | `ContractError` | 원본은 로컬에만 (§1.4) | 해당 없음 (설계) |
| `scale.basis` 없는 metric claim | claim 거부 | §4 | 해당 없음 (설계) |
