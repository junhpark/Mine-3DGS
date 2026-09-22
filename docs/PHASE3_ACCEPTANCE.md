# Phase 3 — Manual Acceptance 절차

Phase 3 의 새 경로(영상/360 → SfM → 정합 → dataset)를 **실제 COLMAP 과 실제 사진**으로 한 번
사람이 직접 통과시키는 절차다. 목적은 "합성 fixture 안에서만 맞는 게 아니라 실물에서도 같은
계약을 수행하는가" 하나다.

**이것은 G2 가 아니다.** benchmark, threshold tuning, GPU 학습, 정확도 주장 어느 것도 여기서
하지 않는다. reconstruction 품질을 올리려는 tuning 을 시작하면 그 순간 acceptance 가 아니라
별도의 실험이다.

> Real-data scientific validation remains NOT VALIDATED. Phase 3 G2 remains PENDING.

---

## 0. 환경 요구사항

| 필요한 것 | 비고 |
|---|---|
| **COLMAP ≥ 4.0** | 아래 주의 참조 |
| Python 3.10–3.12, `pip install -e ".[dev,e57,video]"` | |
| ffmpeg | 영상에서 프레임을 뽑을 때만. 이미지 시퀀스면 불필요 |
| GPU | **불필요** — 이 절차는 학습을 하지 않는다 |

> **주의 — `apt install colmap` 로는 안 된다.** Ubuntu 24.04 LTS 가 주는 것은 **3.9.1** 이고,
> 3.x 는 같은 인터페이스의 옛 버전이 아니다: `global_mapper` 도 `rig_configurator` 도 없고,
> 특징 추출 옵션 이름이 `SiftExtraction.*`(4.0 은 `FeatureExtraction.*`) 이다. minegs 는 이제
> 실행 전에 버전을 확인하고 그 이유를 말하며 거부한다. 버전을 읽을 수 없으면 거부하지 않고
> "모름" 으로 기록한다. `docker/Dockerfile.cpu` 가 이 프로젝트가 구동하는 버전이다.

환경을 보고서에 기록한다.

```bash
git rev-parse HEAD
python -c "import minegs; print(minegs.__version__)"
python -V && colmap -h | head -1 && (. /etc/os-release && echo "$PRETTY_NAME")
```

## 1. 입력

일반 perspective 영상 또는 이미지 시퀀스. 20–100장이면 충분하다. 한 방향으로 진행한
복도/갱도형 공간. 실제 갱도 영상이 있으면 그것이 가장 좋고, 없으면 임시 복도 촬영도 된다.

**합성 렌더링은 manual acceptance 가 아니다.** 실제 카메라로 찍은 사진이어야 한다.

## 2. A1 — FrameSet

```bash
minegs ingest video frameset <video-or-image-dir> work/fs --kind video --fps 2 --blur 60 --hamming 6
```

사람이 `work/fs/images/` 에서 **몇 장을 직접 열어 본다.**

PASS: 이미지가 깨지지 않았고, 방향이 정상이고, dedup 이 과하지 않고(남은 수가 합리적),
쓸 수 없는 프레임이 대량으로 남아 있지 않고, `frameset.json` 의 수와 실제 파일이 일치한다.

## 3. A2 — 실제 COLMAP SfM

**이번 acceptance 에서 가장 중요한 단계.** stand-in 을 쓰지 않는다 (CLI 에는 그 경로가 없다).

```bash
minegs ingest video sfm work/fs work/sfm --backend colmap --mapper global
```

기록: backend, COLMAP 버전, 실행된 명령(`work/sfm/sfm.log`), 입력 이미지 수,
등록 이미지 수, sparse 점 수, component 수, 선택된 component.

```bash
python -c "
import json; r = json.load(open('work/sfm/sfm.json'))
for k in ('backend','backend_version','real_sfm_execution','registered_images','points',
          'selected_component','frame','metric_state'): print(f'{k:22}', r[k])
print('components            ', [c['name'] for c in r['components']])"
```

사람이 sparse reconstruction 을 **눈으로 본다** — `colmap gui --import_path work/sfm/sparse/0
--database_path work/sfm/database.db` 또는 임의의 뷰어.

PASS: `real_sfm_execution = true`, 재구성이 복도/갱도 형태를 대략 재현, camera trajectory 가
뒤집히거나 폭주하지 않음, 의미 있는 비율의 이미지가 등록, component 선택이 합리적.

SfM 품질이 나쁘면 **code bug 인지 data limitation 인지 구분해서** 기록한다.

## 4. A3 — Metric registration

```bash
minegs eval register work/sfm work/reg --basis known_target \
  --targets targets_sfm.csv --targets-tls targets_tls.csv \
  --reference-ply reference.ply \
  --max-rmse-m <m> --min-inlier-ratio <r> --min-correspondences <n>
```

- 전체 TLS ICP 를 claim 용으로 쓰지 않는다. 쓰면 결과는 진단용이고, 그렇게 기록된다.
- threshold 를 여기서 튜닝하지 않는다. 아직 정하지 않았다면 세 개를 모두 빼고 실행하면 되고,
  그 경우 `claim_allowed = false` 가 **정상**이다 — 이 절차의 목적은 claim 생성이 아니다.
- 세 threshold 는 전부 있거나 전부 없어야 한다. 일부만 주면 거부되고 이유가 나온다.

사람이 SfM 점과 reference 를 겹쳐 본다.

PASS: scale 이 명백히 틀리지 않음, 좌우/상하 반전 없음, translation 폭주 없음, trajectory 와
reference 가 같은 공간에서 겹침, `support_ranges_m` 이 실제 사용한 geometry 와 일치.

## 5. A4 — Dataset

```bash
minegs dataset from-sfm work/fs work/sfm work/reg data/acc/dataset \
  --dataset-id acc_v1 --source video --centerline <design.csv>
```

```text
data/acc/dataset/
  images/  sparse/0/  init_points.ply  manifest.json
  provenance/phase3/{frameset,sfm,registration,init_provenance}.json
  provenance/phase3/sfm_model/{cameras,images,points3D}.txt   ← 원본 SFM_INTERNAL 모델
```

PASS: `source=video`, `initialization.source=sfm_sparse`, scale·registration 블록 존재,
`sfm_model/` 에 원본 모델 보존, `T_tls_from_local` 이 합리적.

## 6. A5 — Validation / protocol

```bash
minegs dataset validate data/acc/dataset --strict
minegs eval protocol data/acc/dataset
```

PASS: strict layout 통과, 예상치 못한 consistency issue 없음, **증거보다 강한 claim 이 나오지
않음.** acceptance 를 했다는 이유만으로 `geometry_accuracy` / `volume_accuracy` 가 열리면 안
된다. 증거가 부족하면 fail-closed 가 정상이다.

## 7. A6 — Image-only Golden Gate (가능하면)

```bash
minegs dataset golden-gate data/acc/dataset --out work/gg --reference-ply reference.ply
```

결과는 **structural / manual diagnostic** 이지 scientific validation 이 아니다.
`real_data_validation_status` 는 언제나 `pending_human_inspection` 이다 — 사람이 볼 때까지.

## 8. 360 sanity (perspective 가 PASS 한 뒤에만)

작은 샘플로 ring crop 까지만 본다. 전체 E2E 를 완료할 필요 없다.

```bash
minegs ingest video frameset <pano-dir-or-video> work/fs360 --kind video360 \
  --n-yaw 8 --fov-deg 90 --size 1600 --pano-source Configured --pano-az-sign 1 --nadir-el-deg -35
```

확인: azimuth 방향, elevation 방향, crop 중복/뒤집힘 없음, crop 파일 stem 이 전부 유일,
mask 정렬, 같은 파노라마의 crop 들이 동일 광학 중심(rig) 의미 유지.

## 9. 판정

| | |
|---|---|
| **PASS** | 사람의 수동 확인에서도 정상 작동 |
| **PASS WITH MINOR ISSUES** | 작동하나 UX·logging·문서 등 non-blocking 문제 존재 |
| **FAIL** | real correctness 문제 — COLMAP 호출 실패, record 와 실제 artifact 불일치, frame 오류, axis flip, scale 오염, provenance 불일치, dataset 오류, 숨은 TLS 초기화, 증거보다 강한 claim, 문서대로 진행 불가 |

버그를 찾으면: 원인 기록 → **최소 수정** → 재현 regression test 하나 → acceptance 재수행.
구조적 재설계는 하지 않는다. 주변 코드를 정리하지 않는다.

## 10. Stop condition

perspective 입력 → 실제 COLMAP → 정합 → dataset → strict validation → protocol 이 **한 번**
성공하고 사람이 reconstruction 과 정합을 눈으로 확인했으면 끝이다. 더 나은 reconstruction 을
위한 tuning 은 acceptance 가 아니다.
