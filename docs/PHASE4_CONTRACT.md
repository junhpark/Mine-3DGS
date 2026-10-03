# Phase 4 — Advanced GS / Heavy Profile 계약 (freeze)

이 문서는 Phase 4 구현 **전에** 고정하는 계약이다. 구현이 이 문서를 바꾸는 것이 아니라, 이 문서가
구현의 판정 기준이다. 구현 중 계약이 비어 있던 곳은 §14 구현 기록에 남긴다.

source of truth: 범위와 gate 는 [docs/ROADMAP.md](ROADMAP.md) §Phase 4, 불변식은
[docs/ARCHITECTURE.md](ARCHITECTURE.md) §3·§7. Phase 3 의 dataset/registration 계약은
[docs/PHASE3_CONTRACT.md](PHASE3_CONTRACT.md).

## 0. 시작 기록 (2026-10-03)

| 항목 | 값 |
|---|---|
| base | `main @ eeedd04e3262ae94d6533953cd7936aeaef580c8` (PR #15 merge) |
| branch | `phase-4-advanced-gs` |
| 테스트 수 (base) | 691 collected |
| gsplat pin | `PINNED_GSPLAT = "1.5.3"`; trainer = `examples/simple_trainer.py` at tag `v1.5.3` (commit `937e299`), docker `/opt/gsplat/examples/simple_trainer.py` |
| upstream 의 COLMAP reader | `rmbrualla/pycolmap@cc7ea4b` (`examples/requirements.txt:4`) |
| heavy profile (base) | `default_runner: runpod`, `requests.depth_loss: true` → `DEPTH_LOSS_REFUSAL` 로 **실행 불가** |
| backend capability (base) | appearance_embedding, bilateral_grid, antialiasing, absgrad, mcmc_strategy, pose_refinement, depth_render = true; **depth_loss, normal_loss, resume = false** |

시작 시점의 성숙도 (이 문서가 바꾸지 않는다):

> Phase 3 implementation: COMPLETE · structural testing: PASS ·
> Phase 3 manual acceptance: **DEFERRED** · real COLMAP reconstruction: **NOT PERFORMED** ·
> real GPU training: **NOT PERFORMED** · scientific G2: **NOT VALIDATED** · Phase 3 G2: **PENDING**

Phase 4 는 implementation COMPLETE / structural PASS 까지 갈 수 있다. real GPU 실행과 G3 는
이 Phase 의 구조 작업이 증명할 수 없다 (§12).

---

## 1. 목적과 non-goals

### 1.1 만드는 것

과학적으로 방어 가능한 advanced/heavy 학습 경로. metric geometry, leakage 경계, provenance,
fail-closed 를 baseline 과 같은 강도로 지킨다.

* A. depth supervision 재설계 — 초기화 geometry 와 분리된, versioned, hash 로 묶인 artifact.
* B. heavy profile 을 구조적으로 실행 가능하게.
* C. appearance 처리를 truthful 하게 — 요청 / 실제 trainer config / evidence 가 일치.
* D. 학습 evidence / provenance 강화 — 요청과 실제의 불일치는 거부.
* E. baseline ↔ advanced 비교 경로 — 같은 dataset/protocol 일 때만.
* F. `normalize_world_space` 부채 감사 — 해제하지 않는다면 그 이유를 현재 사실로.

### 1.2 non-goals

2DGS, PGSR, GLUEMAP, surface-aware third-party backend, 2DGS 가 필요한 normal supervision, RunPod,
long-tunnel chunking, multi-epoch, viewer/web, TSDF/mesh, training resume. gsplat 을 교체하지 않는다.
기술은 입증된 요구를 따른다 — Phase 2/3 가 실제 GPU 에서 한 번도 돌지 않았으므로 **관측된 failure
mode 는 아직 없다** (§3.4). 그래서 이 Phase 가 고르는 기술은 최소이고, 성능 이득을 주장하지 않는다.

---

## 2. Reality audit — pinned gsplat v1.5.3

### 2.1 방법

* `v1.5.3` tag (commit `937e299`) 를 clone 해서 **코드를** 읽었다 (upstream main 이 아님). 대상:
  `examples/simple_trainer.py`, `examples/datasets/colmap.py`, `examples/datasets/normalize.py`,
  `examples/utils.py`, `examples/lib_bilagrid.py`, `gsplat/rendering.py`,
  `gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu`, `gsplat/strategy/{default,mcmc,ops}.py`,
  `gsplat/distributed.py`, `gsplat/exporter.py`.
* upstream 이 고정한 COLMAP reader `rmbrualla/pycolmap@cc7ea4b` 도 clone 해서 실행했다.
* 7 개 주제를 독립 auditor 가 file:line 인용으로 답하고, 각 답을 별도 skeptic 이 인용을 다시 읽어
  반박을 시도했다 (§2.5). 핵심 주장은 이 문서 작성자가 직접 재확인했다 (§2.2 의 ✔).

### 2.2 C0 질문 10 개 — 답 (코드 기준)

**Q1. upstream 은 무엇을 "depth" 라 부르는가.** ✔
`Config.depth_loss: bool = False`, `depth_lambda: float = 1e-2` (`simple_trainer.py:171-174`).
target 은 `Dataset.__getitem__` 이 만든다 (`colmap.py:411-432`): `parser.points[point_indices[name]]`
를 dataset camera 로 투영해 `points` (픽셀 좌표, 이름과 달리 2D) 와 `depths = points_cam[:, 2]` 를
돌려준다. prediction 은 `render_mode="RGB+ED"` 의 4 번째 채널 (`simple_trainer.py:648-654`).
loss (`simple_trainer.py:688-706`):

```
L = depth_lambda · scene_scale · mean_m | ρ(ED(u_m)) − 1/z_m |,   ρ(x) = 1/x (x>0), 0 otherwise
```

ED 는 `grid_sample(align_corners=True)` 로 `u/(W−1)·2−1` 에서 샘플한다. weight/confidence/mask 가
없고 (`F.l1_loss` mean), schedule 도 없다.

**Q2. camera-Z / ray distance / disparity?** ✔
target 과 prediction 모두 **camera-Z** (OpenCV +z, 광축 방향). 비교는 disparity (1/z) 공간.
`ED = Σ w_i z_i / Σ w_i`, `z_i` = Gaussian 중심의 camera-z (`rendering.py:760-768`, alpha 는
`clamp(min=1e-10)`). Euclidean ray distance 모드는 v1.5.3 에 없다 (`render_mode` ∈ RGB, D, ED, RGB+D,
RGB+ED). MineGS 의 `GsplatDepthRenderer` (`render_mode="ED"`) 와 `backproject_depth`
(`depth.py:47-56`, z 를 +z 방향 m 로 해석) 와 **같은 양**이다.

**Q3. track → sample 변환.** ✔
`Parser` 는 `manager.point3D_id_to_images` (points3D 의 TRACK) 에서 image → point index 맵을 만든다
(`colmap.py:201-215`). 2D keypoint (`POINTS2D`) 와 `point2D_idx` 는 **읽지 않는다**. 샘플 픽셀은
관측 keypoint 가 아니라 3D 점의 **재투영**이다. 필터는 `0 ≤ u < W`, `0 ≤ v < H`, `z > 0` 뿐
(`colmap.py:421-430`) — 재투영 오차·track 길이·가림 검사 없음. 이미지당 샘플 수 제한 없음.
track 이 없는 이미지는 `point_indices[image_name]` 에서 **KeyError** (`colmap.py:415`).

**Q4. frame 과 scale.** ✔
target 과 prediction 은 같은 학습 frame 에 있다 (`parser.points`, `parser.camtoworlds`). MineGS 는
`--no-normalize_world_space` 를 넘기므로 그 frame 은 staged `sparse/0` 의 frame = `LOCAL_METRIC`,
단위 m. `scene_scale = 1.1 · max‖c − mean(c)‖ · global_scale` (`colmap.py:345-348`,
`simple_trainer.py:347`) 은 정규화 여부와 무관하게 **항상** 계산되고 MineGS 에서는 m 이다.
disparity (1/m) × scene_scale (m) 이므로 loss 값은 world scale 에 불변이다 — 즉 `depth_lambda` 는
frame 간에 같은 뜻이다. 다만 scene_scale 은 **카메라 배치 길이**에 비례하므로, 긴 갱도에서는 같은
`depth_lambda` 의 실효 가중이 갱도 길이에 비례한다 (Phase 5 chunking 의 문제로 기록).

**Q5. 정규화가 하는 일.** ✔ (상세: §8)
`normalize_world_space` (upstream 기본값 **True**) 는 `Parser.__init__` 안에서
`T1 = similarity_from_cameras` (회전·재중심·`1/median` scale), `T2 = align_principal_axes` (강체),
조건부 `T3 = diag(1,−1,−1,1)` (`median(z) > mean(z)` 일 때) 를 카메라와 점에 적용한다
(`colmap.py:218-244`). 변환은 `parser.transform` 메모리에만 있고 **어디에도 쓰이지 않는다** —
cfg.yml 은 boolean 만, checkpoint/PLY 는 정규화 frame 그대로 (역변환 없음).

**Q6. depth 를 init 점과 독립적으로 공급할 수 있는가.** ✔
**없다.** init (`init_type=sfm`: `parser.points`, `simple_trainer.py:234-236`) 과 depth target
(`parser.points[point_indices]`) 은 **같은 배열**이다. CLI/config 에 depth 파일 경로 옵션도 없다
(depth 관련 필드는 `depth_loss`, `depth_lambda` 둘뿐). 유일한 경로는 points3D 의 track 이고, 그것은
init 점에 묶여 있다. 따라서 upstream `--depth_loss` 는 MineGS 에서 **영구히** 거부한다 (AD-5).

**Q7. appearance 호환성.** ✔
`--app_opt` 는 Gaussian 의 색 parameterisation 을 바꾼다: `sh0/shN` 대신 `features` (32-d) 와
`colors` (`simple_trainer.py:265-275`), 이미지별 embedding 을 가진 `AppearanceOptModule`
(`utils.py:51-115`, `len(trainset)` 개). 기하 parameter (`means`, `scales`, `quats`, `opacities`) 는
그대로이고 ED 깊이는 색과 무관하다 → **depth-neutral**. checkpoint 에 `app_module` 이 추가된다
(`simple_trainer.py:772-778`). PLY export 는 원점·zero-embedding·`dirs=0` 으로 색을 **bake** 하고
`shN` 은 비운다 (`simple_trainer.py:786-797`) — PLY 색은 근사다. eval (val 이미지) 은 `image_ids`
없이 렌더하므로 **zero embedding** 이다 (`utils.py:95-96`).

**Q8. bilateral flag.** ✔
`--use_bilateral_grid` (lib_bilagrid) 또는 `--use_fused_bilagrid` (true 로 강제하고 fused_bilagrid 를
쓴다, `simple_trainer.py:1228-1243`). 이미지별 grid 로 **색만** 후처리 (`simple_trainer.py:656-667`).
eval 의 `cc_psnr/cc_ssim/cc_lpips` 는 GT 에 affine 색 보정을 맞춘 뒤 잰 값이다
(`simple_trainer.py:958-962`) — held-out render quality 로 보고하면 안 된다.

**Q9. render semantics 를 바꾸는 기능.** ✔
| 기능 | 기하 | pose | 색만 | MineGS renderer 재현 |
|---|---|---|---|---|
| `antialiased` | opacity 보정 → ED 가중이 바뀜 | – | – | **아니오** (classic 고정) |
| `pose_opt` | – | 바뀜 (`pose_adjust`) | – | **아니오** (거부) |
| `camera_model` fisheye/ortho, `with_ut/with_eval3d` | 투영이 바뀜 | – | – | **아니오** (pinhole 고정) |
| `near_plane/far_plane` | cull 이 바뀜 | – | – | near 0.01 고정; 다르면 거부 |
| `app_opt` | 아니오 | 아니오 | 예 | 예 (기하만 읽음) |
| bilateral grid | 아니오 | 아니오 | 예 | 예 |
| MCMC / default strategy | 학습 동역학만 | – | – | 예 |
| `depth_loss`/depth supervision | 학습되는 기하가 달라질 뿐 렌더 규칙은 같다 | – | – | 예 |
| `packed`, `sparse_grad`, `visible_adam`, `absgrad` | 아니오 | – | – | 예 |

**Q10. trainer evidence 에 무엇이 남는가.** ✔
`cfg.yml` = `yaml.dump(vars(cfg))` (`simple_trainer.py:552-554`) — 모든 Config 필드, nested strategy
는 `!!python/object:gsplat.strategy...` tag 아래 들여쓴 mapping. checkpoint `ckpts/ckpt_{step}_rank{r}.pt`
= `{step, splats, [pose_adjust], [app_module]}` (`max_steps−1` 과 `save_steps−1` 에서). stats
`stats/train_step{step:04d}_rank{r}.json` = `{mem, ellipse_time, num_GS}`. depth loss 값은 tensorboard
에만 남는다. PLY `ply/point_cloud_{step}.ply`. `scene_scale` 과 정규화 변환은 stdout 에만 / 어디에도.

### 2.3 감사가 드러낸 기존 결함 (Phase 0D 부터, 실제 GPU 실행이 없어 숨어 있었다)

이것들은 Phase 4 를 위해 고친다. heavy 를 "실행 가능" 하게 하려면 피할 수 없고, 같은 staging 을
쓰는 baseline 도 같은 이유로 실제로는 실행 불가였다.

1. **upstream COLMAP reader 가 MineGS staged 모델을 읽지 못한다.** ✔ (직접 재현)
   staging 은 COLMAP **text** 만 쓴다. 고정된 pycolmap fork 의 text loader 는 Python 2 코드
   (`np.array(map(...))`, `scene_manager.py:198-203, 275-281`) 라 Python 3 에서
   `images.txt` → `Exception: Input quaternion should be a 3- or 4-vector`,
   `points3D.txt` → `ValueError: cannot reshape array of size 1 into shape (2)`. 또
   `iter(lambda: f.readline().strip(), '')` 는 첫 빈 줄에서 멈추는데, track 을 비운 이미지는 빈
   `POINTS2D` 줄을 쓴다. fork 는 `.bin` 을 먼저 찾는다 (`scene_manager.py:88-92, 127-137, 217-227`).
   → **staging 이 binary COLMAP 모델을 함께 쓴다** (§14).
2. **`data_factor > 1` 은 `images_<factor>/` 가 이미 있어야 한다.** ✔
   `Parser` 는 `images_<f>` 가 없으면 `ValueError("Image folder ... does not exist")` (`colmap.py:183-187`).
   `images_<f>` 의 첫 파일이 `.jpg` 일 때만 upstream 이 `images/` 에서 `images_<f>_png` 로 직접
   축소한다 (`colmap.py:193-197`, `_resize_image_folder` `colmap.py:31-53`: PIL BICUBIC, `round(w/f)`). light (factor 4)
   와 heavy (factor 2) 모두 해당. → staging 이 `images_<f>/` 를 만든다 (§14).
3. **upstream `test_every = 8` 이 staged train 이미지의 1/8 을 학습에서 뺀다.** ✔
   MineGS 는 `--test_every` 를 넘기지 않는다. `indices % 8 == 0` 인 이미지 (이름순) 는 upstream 의 val
   이 되어 photometric 도 depth 도 받지 않는다 (`colmap.py:366-369`). MineGS 의 holdout 은 dataset
   수준에서 이미 분리되어 있으므로 leakage 는 아니지만, "staged 이미지 수 = 학습된 이미지 수" 가
   아니다. light 는 그대로 두고 (수치 보존) evidence 에 실제 최적화 이미지 수를 기록한다. heavy 는
   `test_every` 를 이미지 수보다 크게 두어 upstream val 을 한 장으로 줄인다 (AD-6).
4. **`masks/` 는 upstream 이 읽지 않는다.** ✔ `Parser.mask_dict` 는 fisheye ROI 용으로만 채워진다
   (`colmap.py:136, 297-342`). staging 이 링크하는 `masks/` 는 학습에 쓰이지 않는다. Phase 4 는 이것을
   바꾸지 않고 기록한다 (upstream 의 `masks` 의미 — 렌더만 0 으로, GT 는 그대로 — 는 MineGS mask 에
   맞지 않는다).
5. **MCMC 의 noise 항은 단위에 의존한다.** (auditor, 차원 분석)
   `inject_noise_to_position(scaler = lr · noise_lr)` (`mcmc.py:143-145`), `lr` 은 `means_lr ·
   scene_scale` (길이), noise 는 `covar(길이²) · randn · scaler` → 장면 대비 noise 가 world scale 의
   제곱으로 커진다. `noise_lr = 5e5` 는 정규화 frame (scene ~1) 에서 맞춘 값이다. preset 의
   `scale_reg = 0.01` (`exp(scales).mean()`, 길이) 도 단위에 의존한다. LOCAL_METRIC 에서는 둘 다 다른
   실험이 된다. → AD-6 의 metric 보정.
6. **span-ratio backstop 의 "30-100x shrink" 주석은 신뢰할 수 없다.** (auditor) 정규화 scale 은
   LOCAL_METRIC 원점 위치에 의존하는 `focus` 점까지의 median 거리라, 원점이 가운데인 60 m 합성
   갱도에서 약 15x 만 줄어 통과한다. 주 guard 는 cfg.yml 의 `normalize_world_space` 이고 그것은 그대로
   유지된다. 주석은 정정한다.
7. **`gsplat_normalization()` 은 Parser 가 아니라 `normalize.py::normalize` (T2@T1) 를 흉내낸다.**
   (auditor) Parser 는 조건부 T3 flip 을 더한다. 사용처가 없으므로 equivalence 의 근거로 쓰면 안 된다는
   주석을 단다 (삭제는 unrelated refactor 라 하지 않는다).
8. **`init_type=random` 은 metric frame 에서 안전하지 않다.** (auditor) 무작위 cube 는 **world 원점**
   중심 (`simple_trainer.py:238`) 이다. LOCAL_METRIC 원점이 갱도 안에 있다는 보장이 없다. → 거부.
9. **`stage_dataset(use_init_points=False)` 는 빠진 이미지를 가리키는 track 을 남긴다.** (auditor)
   upstream `colmap.py:210` 에서 KeyError. 호출하는 곳은 없지만 binary 를 쓰면서 track 을 staged
   이미지로 거른다.
10. **`app_opt` 의 초기 색은 `torch.logit(rgb/255)` 다.** ✔ (`simple_trainer.py:236, 275`) TLS init 점의
    0 또는 255 채널은 ±inf parameter 가 된다 (gradient 0 으로 영구 포화, checkpoint 에 inf). →
    appearance 를 요청한 run 의 staging 은 init RGB 를 [1, 254] 로 clamp 하고 기록한다.
11. **`export_splats` 는 non-finite splat 을 조용히 버린다.** (auditor, `gsplat/exporter.py:515-538`)
    그래서 `verify_postconditions` 의 "중심이 유한한가" 검사는 gsplat PLY 에서 발화할 수 없다. →
    마지막 step 의 stats `num_GS` 와 PLY vertex 수가 같아야 한다 (다르면 FAILED).
12. **cfg.yml 은 run 밖으로 나오지 않는다.** (auditor) `normalize_outputs` 는 ckpts/stats/renders 만
    복사하고 `trainer_config` 는 run.json 에 저장되지 않는다. render gate 의 cfg.yml witness 는
    `if cfg.is_file()` 로만 동작한다. 정규식 reader 는 nested strategy 필드와 list 를 놓친다. →
    tag 를 실행하지 않는 YAML reader 로 읽어 핵심 값과 cfg.yml sha256 을 run.json 에 남기고, cfg.yml 이
    없으면 FAILED.
13. **depth renderer 는 학습 해상도가 아니라 full 해상도로 렌더한다.** (auditor) 학습은
    `K/data_factor`, `width//data_factor` 에서 했고 `eps2d = 0.3` 은 픽셀 단위다. 따라서 full 해상도
    ED/alpha 는 모델이 학습된 ED/alpha 와 정확히 같지 않다. **Phase 1B 부터 있던 차이**이고 baseline
    에도 해당한다. 고치려면 Phase 1B 의 "depth 해상도 = camera intrinsics" 계약을 바꿔야 하므로
    Phase 4 에서는 바꾸지 않고 **SHOULD_FIX (deferred)** 로 기록한다. run evidence 에 `data_factor` 를
    남기고, 비교는 서로 다른 `data_factor` 의 run 을 비교할 때 이 사실을 표시한다.
14. **ED 는 Gaussian 중심 camera-z 의 alpha 가중 평균이다** (`ProjectionEWA3DGSFused.cu:205`). ray 와
    표면의 교점 깊이가 아니다. 크거나 비스듬한 Gaussian 에서는 중심-z bias 가 있다. depth supervision
    과 평가 모두 같은 양을 쓰므로 일관되지만, 이 bias 는 depth 정의의 일부로 기록한다.

### 2.4 ROADMAP 과의 불일치

| ROADMAP | 사실 | 처리 |
|---|---|---|
| Phase 4 는 "실제로 관측된 failure mode" 를 근거로 | Phase 2/3 실제 GPU 실행 없음 → 관측된 failure mode 없음 | 최소 기술만. 성능 주장 없음. G3 PENDING |
| depth_loss 거부 근거 = "staging 이 track 을 비운다" | 더 근본적: upstream 은 depth target = init 점 (같은 배열). track 을 살려도 분리 불변식 위반 | upstream `--depth_loss` 영구 거부 + MineGS artifact |
| normalize: "재구현한 similarity_from_cameras + align_principle_axes 가 실제 파서와 동일한지" | 재구현은 Parser 가 아니라 `normalize()` 를 흉내 (T3 누락); upstream 은 변환을 저장하지 않음 | §8 — 7 조건 중 미충족 다수, 거부 유지 |
| heavy `default_runner: runpod` | RunPod 는 Phase 6 | `local` 로 |
| 후보: normal supervision, 2DGS, PGSR | 이 Phase 의 non-goal | §1.2 |
| fail-closed 표: "`--profile heavy` 실행 → ContractError (depth_loss)" | Phase 4 가 해제 | ROADMAP 갱신 |

### 2.5 감사 검증 상태

§14 에 결과를 남긴다 (각 주제의 skeptic verdict: confirmed / corrected / refuted).

---

## 3. Architecture decisions

### AD-1 — depth supervision 은 init 과 분리된 versioned artifact

`DepthSupervisionRecord` (schema `1.0`) + `samples.npy`. 위치는 기본
`<dataset>/supervision/depth/<supervision_id>/`. init geometry (`init_points.ply`, staged `points3D`)
와 **파일도, 해시도, provenance 도 따로**다. 가짜 COLMAP track 을 init 점에 붙이지 않는다.

* init 을 바꿔도 depth artifact 의 바이트는 바뀌지 않고 여전히 검증을 통과한다 (dataset binding 이
  init 을 포함하지 않는다).
* depth artifact 를 바꿔도 staged init 은 바뀌지 않는다.
* supervision 바이트가 바뀌면 **학습 identity 가 바뀐다**: run.json 의 `depth_supervision`
  (artifact sha256, samples sha256) 과 staged tree hash 에 들어간다.

### AD-2 — depth semantics

| 필드 | 값 | 뜻 |
|---|---|---|
| `frame` | `LOCAL_METRIC` | 샘플을 역투영한 점이 사는 frame |
| `depth_unit` | `m` | |
| `depth_semantics` | `camera_z` | dataset camera 의 OpenCV +z 좌표 (광축 방향 거리, ray 길이가 아님) |
| `pixel_convention` | `colmap_continuous` | dataset camera 의 **full resolution** 연속 좌표, 픽셀 (col,row) 의 중심 = (col+0.5, row+0.5). `colmap_io.project` 및 rasteriser (`px = j + 0.5`) 와 같은 규약 |
| `confidence_semantics` | `binary_mask` \| `unit_interval_weight` | §AD-4 |

pinhole (`PINHOLE`, `SIMPLE_PINHOLE`) 카메라만 받는다 — renderer 와 같은 제한.

### AD-3 — leakage (hard blocker)

holdout 의 깊이는 학습에 도달하지 않는다. 판정은 실제 support 로 한다:

1. **점 규칙**: 샘플의 3D 점 (LOCAL_METRIC) 의 chainage 가 holdout 범위 안이면 제외.
2. **ray 규칙**: 카메라 중심 → 점 선분이 holdout 을 지나면 제외. 판정은
   `[min(s_cam, s_pt) − margin, max(s_cam, s_pt) + margin]` 이 holdout 범위와 겹치는지로 한다
   (`margin` 기록, 기본 0.5 m). 직선 centerline 에서는 선분 위 chainage 가 단조이므로 정확하고,
   굽은 centerline 에서는 margin 이 덮는 근사다 (§14 에 기록). 이 규칙 때문에 builder 는 holdout
   점을 아예 읽지 않아도 된다 — holdout 을 지나는 ray 는 남지 않으므로 빠진 holdout 기하가 남은
   샘플을 틀리게 만들 수 없다.
3. **이미지 규칙**: 샘플은 manifest 의 `train_images()` 에만 달릴 수 있다. test group 이미지, holdout
   때문에 빠진 이미지, dataset 에 없는 이미지는 거부.
4. **track 규칙 (sfm_tracks)**: track 에 train 이 아닌 이미지가 하나라도 있으면 그 점 전체를 제외한다
   — 그 깊이는 held-out 이미지의 관측으로 삼각측량된 값이기 때문이다.
5. **support 를 모르면 fail closed**: holdout 이 선언되었는데 centerline 이 없으면 build 를 거부한다.
   centerline 끝 밖이거나 반경 `max_radial_m` (기록, 기본 10 m) 보다 먼 점·카메라는 "위치를 알 수
   없음" 이다. builder 는 그런 샘플을 내보내지 않고 (개수 기록), **검증기는 그런 샘플이 하나라도 있는
   artifact 를 거부**한다. 경고가 아니다.
6. 검증은 **재도출**이다: 저장된 (image, u, v, depth) 를 dataset camera 로 역투영해 3D 점과 카메라
   중심을 다시 계산하고 1–5 를 다시 판정한다. record 의 선언 (`excluded_holdout_ranges_m`, 개수) 을
   믿지 않는다.
7. **TLS depth 는 image-only dataset 에 붙일 수 없다** (`source ∈ {video, video360}` 이면 거부) —
   그 순간 image-only 경로가 TLS-assisted 가 된다 (Phase 3 독립성).

holdout 이 없는 dataset 에서는 1·2·5 가 할 일이 없다. centerline 이 있으면 support 범위는 그래도
기록한다.

### AD-4 — confidence / uncertainty

* `binary_mask`: 값은 정확히 0 또는 1. 0 인 샘플은 artifact 에 남지만 (감사용, 이유별 개수)
  loss 에 들어가지 않는다.
* `unit_interval_weight`: [0, 1] 의 유한값. loss 가 그대로 가중한다.
* 그 밖의 값 (NaN, 음수, >1, binary 인데 0.5) → artifact 거부.
* loss: `Σ w·|ρ(ED) − 1/z| / Σ w` (이미지별), `Σ w = 0` 인 이미지는 depth 항 0 (NaN 아님).
* Phase 4 의 두 builder 는 **`binary_mask` 만** 낸다. 연속 신뢰도를 보정할 데이터가 없으므로 연속
  가중치를 지어내지 않는다. 형식과 loss 는 `unit_interval_weight` 를 소비할 수 있다 (향후 sensor).

### AD-5 — trainer 소유권: upstream loop 그대로 + MineGS depth 항

upstream 은 별도 depth 증거를 먹을 수 없다 (Q6) — 그래서 얇은 MineGS adapter
`minegs/train/trainers/advanced_gs.py` 를 둔다. **upstream 학습 loop 를 복사하지 않는다.**

* adapter 는 upstream `simple_trainer.py` 를 그 파일 그대로 실행한다 (`runpy`, upstream 의 tyro CLI,
  preset, `adjust_steps`, bilateral import 까지 그대로). 실행 전에 `gsplat.distributed.cli` 를
  감싸서, upstream 이 파싱한 `cfg` 와 `main` 을 받은 뒤 `main` 의 전역 `Runner` 를 MineGS 하위 클래스로
  바꾼다.
* 하위 클래스가 바꾸는 것은 `rasterize_splats` **하나**다. 학습 호출 (grad 활성, `image_ids` 있음) 에서만
  `render_mode="RGB+ED"` 로 렌더해서, ED 채널로 MineGS depth 항을 계산하고, 그 항의 gradient 를
  autograd 함수로 색 출력에 **주입**한 뒤 RGB 3 채널만 upstream 에 돌려준다. upstream 이 계산하는
  loss 값은 바뀌지 않고, `loss.backward()` 가 depth 항의 gradient 를 정확히 한 번 더한다.
* upstream `depth_loss` 는 **항상 false** 다 (cfg.yml 이 증거). MineGS depth 항은 adapter 의 evidence
  (`minegs_trainer.json`) 가 증거다.
* adapter 가 소유하는 것: supervision 로딩·검증, 좌표 변환 (factor, half-pixel), 가중 loss,
  NaN-safe 역수, evidence 기록. 그 밖은 upstream.
* depth 항: `λ · scene_scale · mean_images[ Σ w|ρ(ED(û)) − 1/z| / Σ w ]`, `λ = cfg.depth_lambda`
  (upstream 필드를 그대로 쓰므로 cfg.yml 에 남는다), `scene_scale` 은 upstream Runner 의 값,
  `û = u/f − 0.5` (half-pixel 보정, `align_corners=True` 인덱스), 샘플은 `û ∈ [0, W_f−1]` 만 (zero
  padding 없음), `ρ(x) = 1/x (x > ε)`, 아니면 0 — gradient 가 NaN 이 되지 않게 분모를 먼저 안전화.

### AD-6 — heavy profile

| 항목 | 값 | 근거 |
|---|---|---|
| runner | `local` | RunPod 는 Phase 6, 명시 요청 시 계속 `NotYetImplementedError` |
| 이미지 | `max_images: null` (모든 train 이미지) | |
| `test_every` | 1 000 000 | upstream val 을 이름순 첫 이미지 한 장으로 줄인다 (§2.3-3). val 0 장은 upstream eval 이 0 으로 나눈다 |
| `data_factor` | 2 | |
| `max_steps` | 30 000 | |
| strategy | `mcmc` + **metric 보정** | `noise_lr ← 5e5 · s²`, `scale_reg ← 0.01 · s` (s = upstream `similarity_from_cameras` 의 scale 을 staged 카메라에서 계산). 이것으로 MCMC 의 길이 단위 항이 upstream 이 튜닝한 정규화 frame 과 같은 상대 크기가 된다. profile 이 두 값을 직접 주면 그 값을 쓰고 `explicit` 로 기록 |
| appearance | `appearance_embedding: true` | |
| bilateral | `false` | appearance 와 동시에 켜지 않는다 |
| antialiasing | **`false`** | MineGS depth renderer 는 classic 만 재현한다 (Q9). 켜면 모든 heavy run 의 depth 가 render gate 에서 거부되어 surface/비교 사슬에 못 들어간다. 요구가 입증되지 않았다 |
| depth | `depth_loss: true` (= MineGS depth supervision), `depth_lambda: 0.01` | |
| SH | 3 | |

ablation 은 코드 변경 없이 builtin profile 로: `heavy-base` (appearance·depth 끔), `heavy-appearance`,
`heavy-depth`, `heavy` (둘 다). 네 profile 은 `requests.{appearance_embedding, depth_loss}` 만 다르다.

### AD-7 — appearance

capability 로 명시 요청 → `--app_opt`. 실제 cfg.yml `app_opt` 과 비교. evidence 의
`appearance_mode ∈ {none, embedding, bilateral_grid, embedding+bilateral_grid}` 는 **cfg.yml 에서**
도출한다. held-out view 는 upstream 처럼 zero embedding 으로 렌더된다는 사실을 기록한다. render
quality 비교에서 appearance run 은 이 정책을 함께 보고한다.

### AD-8 — evidence 와 요청 ↔ 실제 대조

run.json (schema 1.2) 이 추가로 기록: `trainer` (entrypoint, adapter sha256, upstream trainer 경로/
sha256), `capabilities` (요청/해결), `trainer_config` (실제 cfg.yml 에서 읽은 핵심 값), `depth_supervision`
(id, artifact/record/samples sha256, source kind, 샘플 수, confidence semantics, depth semantics),
`metric_compensation`, `optimised_images`. 이미 있던 것: backend/version, profile, argv, env,
T_local_from_internal, dataset hash, staged hash, git SHA (provenance), runtime.

run 이 끝나면 cfg.yml 과 adapter evidence 를 다시 읽어 대조한다. **불일치는 FAILED**:
cfg.yml 없음 / 요청한 capability 가 cfg 에서 꺼져 있음 / 요청 안 한 capability 가 켜져 있음 /
strategy·data_factor·max_steps·sh_degree·test_every·init_type·normalize 불일치 / renderer 가 가정하는
값과 다름 (`camera_model: pinhole`, `near_plane: 0.01`, `far_plane: 1e10`, `with_ut/with_eval3d: false`,
`pose_noise: 0`, `patch_size: null`) / upstream `depth_loss: true` / depth 요청인데 adapter evidence 없음 /
adapter evidence 의 supervision sha256 ≠ run 이 기록한 값 / adapter sha256 ≠ 이 MineGS 의 adapter
(이미지 안 MineGS 와 host MineGS 의 version skew) / MCMC 보정값 불일치 / 마지막 stats `num_GS` ≠ PLY
vertex 수 / container 가 보고한 gsplat 버전 ≠ pin.
"real" 은 선언으로 바뀌지 않는다: real GPU 여부는 기존처럼 run 의 runtime evidence 와 seam 대체
여부로만 정해진다.

### AD-9 — `normalize_world_space`: 거부 유지 (§8)

### AD-10 — `normal_loss`: false 유지

v1.5.3 `simple_trainer.Config` 에 normal loss 가 없다 (normal consistency 는 `simple_trainer_2dgs.py`
에만 있고 2DGS 는 non-goal). capability note 로 이유를 남긴다.

### AD-11 — render 호환성

Phase 1 render gate 를 우회하지 않는다. `RENDER_NEUTRAL_BACKEND_ARGS` 에 `test_every` (어느 이미지를
최적화할지만), `depth_lambda` (loss 가중), `strategy.noise_lr`·`scale_reg` (최적화 동역학) 를 이유와
함께 추가한다. antialiased, pose_opt, camera_model 등은 그대로 거부. pose refinement 는 claim-bearing
렌더를 계속 거부한다.

### AD-12 — baseline ↔ advanced 비교

같은 dataset identity (dataset hash), 같은 평가 protocol, 같은 holdout, 같은 metric frame
(`LOCAL_METRIC`, `T_local_from_internal = I`), 같은 평가 support 일 때만. 기존 metric (render/geometry/
section/volume/coverage/runtime/peak memory) 을 재사용한다. 없는 값은 `null` (0 아님). loss 감소를
개선으로 읽지 않는다 — 비교는 training loss 를 아예 읽지 않는다. verdict 를 내리지 않고 차이만,
`G3: PENDING`.

---

## 4. `DepthSupervisionRecord` schema 1.0

```
schema_version, supervision_id, dataset_id,
dataset_binding: {cameras_sha256, poses_sha256, split_sha256, centerline_sha256|null,
                  T_tls_from_local, binding_sha256},
source_kind: sfm_tracks | tls_projection | sensor_depth,
source_assets: [{role, path, sha256}],
frame: LOCAL_METRIC, depth_unit: m, depth_semantics: camera_z, pixel_convention: colmap_continuous,
confidence_semantics: binary_mask | unit_interval_weight,
images: [name, ...]                      # samples.image 의 index 공간, 모두 train 이미지
n_samples, n_samples_in_loss, per_image_counts: [int, ...],
support_ranges_m: [[lo, hi], ...] | null, excluded_holdout_ranges_m: [[lo, hi], ...],
exclusions: {reason: count},
creation_params: {...}, minegs_version, git_commit, created_at,
samples_file: samples.npy, samples_sha256
```

`samples.npy`: structured array `image <u4, u <f4, v <f4, depth_m <f4, confidence <f4`
(`np.save`, `allow_pickle=False` 로 읽힌다). artifact identity = 디렉터리의 `sha256_tree`.

dataset binding 은 카메라 내부 parameter, 이미지 pose (이름·camera·qvec·tvec), split (train/test
group, holdout), centerline 파일, `T_tls_from_local` 의 digest 다. **init 은 포함하지 않는다** (AD-1).

---

## 5. Source 계약

### 5.1 `sfm_tracks`

* dataset `source ∈ {video, video360}`, `provenance/phase3/sfm_model/` (SFM_INTERNAL, 원본 SfM 모델) 과
  manifest `registration` 이 있어야 한다. `check_image_only_dataset` 를 먼저 통과해야 한다.
* 점은 `T_local_from_sfm = T_local_from_tls · registration.sim3` 로 metric 화한다 — SfM 내부 scale 은
  임의이므로 registration 없이는 거부. 원본 SfM 모델과 registration record 가 `source_assets`.
* 샘플 = 관측 이미지의 dataset camera 로 metric 점을 **재투영**한 픽셀 + camera-z (upstream 과 같은
  재투영 방식). 관측 keypoint 와의 재투영 오차 > `max_reproj_error_px` 또는 track 길이 <
  `min_track_length` 이면 confidence 0. leakage 규칙 (AD-3) 위반은 제외.

### 5.2 `tls_projection`

* TLS dataset 전용 (image-only 거부). TLS cloud 는 **명시한 materialized PLY** (frame 선언 필수:
  LOCAL_METRIC 또는 TLS_GLOBAL → manifest 의 `T_local_from_tls`). trainer 는 이 artifact 만 읽고 raw
  TLS 로 돌아가지 않는다: raw → dataset/materialized evidence → artifact → training.
* builder 는 holdout·위치 불명 점을 먼저 버리고, train 이미지마다 투영한다. 가림: `cell_px` 격자에서
  셀별 최소 깊이, `k×k` 셀 이웃 최소 깊이보다 `(1+rel_tol)` 배 넘게 먼 셀은 가려진 것으로 버린다
  (sparse 점 사이로 뒷벽이 비치는 것을 막는다). 셀마다 가장 가까운 점 하나를 그 점의 sub-pixel 위치로.
  이미지당 `max_samples_per_image` 를 seed 고정으로 균등 추출. 모두 기록.
* depth = TLS 점의 camera-z.

### 5.3 `sensor_depth`

schema 에 이름만 예약한다. builder 가 없고, 검증기는 이 source 를 **거부**한다 — 깊이 의미와 불확실성
이 정의되지 않은 source 를 지원하는 척하지 않는다.

---

## 6. 실행 경로

```
dataset ──(build)──> supervision/depth/<id>/   (record + samples.npy)
   │                         │
   │   train run --depth-supervision <dir>      │  verify (재도출, AD-3/AD-4)
   ▼                         ▼
staged/ (images, images_<f>/, sparse/0 txt+bin, init as points3D [RGB clamp if app_opt], supervision/depth/)
   │
   ▼
python -m minegs.train.trainers.advanced_gs --supervision ... --supervision-sha256 ... -- mcmc --data_dir ...
   │   (upstream simple_trainer.py, upstream depth_loss=false)
   ▼
backend_out/ (cfg.yml, ckpts, stats, ply, minegs_trainer.json) ──> postconditions + 요청↔실제 대조
```

depth 를 요청하지 않고 strategy 가 `default` 인 profile (light) 은 지금과 **같은 argv** 로 upstream 을
직접 실행한다. depth 요청 또는 MCMC 이면 adapter.

---

## 7. Fail-closed 목록 (Phase 4 추가)

| 요청 | 결과 |
|---|---|
| depth 요청인데 `--depth-supervision` 없음 | `ContractError` |
| supervision 을 줬는데 profile 이 depth 를 요청하지 않음 | `ContractError` (조용히 무시하지 않는다) |
| samples 바이트가 record 의 sha256 과 다름 | `ContractError` |
| 다른 dataset 의 artifact (dataset_id 또는 binding 불일치) | `ContractError` |
| frame/unit/semantics/pixel convention 이 계약 값이 아님 | `ContractError` |
| confidence 값이 semantics 와 맞지 않음 | `ContractError` |
| dataset 에 없는 이미지 / train 이 아닌 이미지 | `ContractError` |
| holdout 안의 점 / holdout 을 지나는 ray / 위치 불명 support | `ContractError` |
| holdout 이 있는데 centerline 없음 | `ContractError` |
| `sensor_depth` source | `ContractError` |
| image-only dataset 에 `tls_projection` | `ContractError` |
| upstream `--depth_loss` (어떤 profile 이든) | `ContractError` (영구) |
| `init_type: random` | `ContractError` |
| `normalize_world_space: true` | `ContractError` (§8 의 현재 이유) |
| run 후: 요청 ↔ cfg.yml / adapter evidence 불일치 | run FAILED |
| `--runner runpod` | `NotYetImplementedError` (Phase 6) |

---

## 8. `normalize_world_space` — 7 조건 감사

| 조건 | 상태 |
|---|---|
| 1. 정확한 변환을 재현하거나 얻는다 | ✗ upstream 은 `parser.transform` 을 메모리에만 둔다. 재현하려면 같은 staged 입력으로 Parser 를 다시 돌려야 하고, `gsplat_normalization()` 은 T3 를 빠뜨린다 |
| 2. 기록한다 | ✗ cfg.yml 은 boolean 만 |
| 3. 가역이다 | △ 수학적으로 Sim3 는 가역이지만 SH band ≥1 회전, log-scale 보정, pose_adjust 가 함께 필요하고 MineGS 는 SH 회전을 하지 않는다 |
| 4. host/container 경로 identity | ✗ 변환을 계산할 쪽 (host) 과 실행되는 쪽 (container) 이 같은 staged 바이트를 본다는 증명이 없다 |
| 5. 출력을 LOCAL_METRIC 으로 되돌린다 | ✗ `normalize_outputs` 는 ckpt 를 그대로 복사한다 |
| 6. renderer 가 그 의미를 재현한다 | ✗ renderer 는 dataset pose (LOCAL_METRIC) 로 렌더하고 `require_metric_outputs` 가 비항등을 거부한다 |
| 7. 실제 equivalence test | ✗ GPU 없음 |

→ 거부 유지. 거부 문구는 "Phase 4 로 미룬다" 가 아니라 위 사실로 바꾼다. metric 학습에서 생기는
단위 의존 항은 정규화 대신 AD-6 의 보정으로 다룬다.

---

## 9. 필수 negative test (지시서 §20 대응)

| 묶음 | 테스트 |
|---|---|
| depth 계약 | 요청인데 artifact 없음 · 변조된 samples · dataset 불일치 · 비 metric/모르는 semantics · 잘못된 confidence · 없는 이미지 |
| leakage | TLS 샘플 holdout 안 · SfM 샘플 holdout 안 · held-out 이미지 supervision · support 위치 불명 → fail closed (모두 고의 오염) |
| init 분리 | init 변경이 depth artifact 를 다시 쓰지 않고 검증도 통과 · 반대 방향 · 가짜 COLMAP track 이 선언된 supervision 을 대신하지 못함 |
| heavy | capability 해결 · pinned flag 만 · local runner 구조 실행 · RunPod 거부 · baseline argv 불변 |
| evidence | 요청한 capability 가 cfg 에서 꺼짐 → 거부 · supervision hash 불일치 → 거부 · substituted 는 선언으로 real 이 되지 않음 |
| normalization | 미지원 → 현재 이유로 fail closed |

---

## 10. Claim 경계

* Phase 4 는 heavy 가 **구조적으로** 실행된다는 것을 보인다. 실제 GPU 실행: NOT PERFORMED.
* 어떤 성능 향상도 주장하지 않는다. 합성 데이터의 수치를 performance win 으로 쓰지 않는다.
* 금지 표현: geometry validated, surface scientifically validated, 3DGS geometry accurate,
  volume validated, real mine validated, G2 passed, G3 passed.

## 11. 성숙도 목표 (Phase 4 종료 시)

```
Phase 4 implementation: COMPLETE (목표)
Structural testing: PASS (목표)
Heavy profile / depth supervision / appearance: IMPLEMENTED
Real GPU heavy training: NOT PERFORMED
Baseline vs heavy real comparison: NOT PERFORMED
G3: PENDING
Scientific validation: NOT VALIDATED
Phase 3 manual acceptance: DEFERRED
Phase 3 G2: PENDING
```

## 12. G3 는 구조적으로 증명할 수 없다

G3 = 같은 dataset/protocol 에서 baseline 대비 정량적 개선. 실제 데이터·GPU 가 필요하다. Phase 4 의
비교 경로 (AD-12) 는 그 비교를 **할 수 있게** 만들 뿐이다.

## 13. 열린 질문 (C0 에서 닫지 않는다)

* MCMC metric 보정이 실제 갱도에서 upstream 정규화 실행과 같은 거동을 내는가 — GPU 필요.
* `scene_scale` 이 갱도 길이에 비례하는 문제 (depth 가중, DefaultStrategy 임계값) — Phase 5.
* TLS projection 의 가림 규칙 parameter 가 실제 스캔 밀도에서 적절한가 — 실데이터 필요.

---

## 14. 구현 기록 — `phase-4-advanced-gs`

### 14.1 C0 감사의 독립 검증 결과

7 개 주제 (정규화, depth 의미, depth 독립성, appearance, render 의미, evidence, adapter flag) 의 모든
답을 별도 skeptic 이 인용을 다시 읽어 반박을 시도했다. **반박(refuted) 된 답은 없다.** 정정
(corrected) 은 세부 사항이고, 이 계약의 결정을 바꾸는 것은 없었다. 기록할 정정:

* `noise_lr = 5e5` 가 "정규화 frame (scene ~1) 에서 맞춘 값" 이라는 말은 코드에 있는 사실이 아니라
  `normalize_world_space=True` 기본값에서 나온 **추론**이다. MCMC noise 가 닿는 것은 opacity ≲ 0.05 의
  Gaussian 이다 (`ops.py:360-365`, opacity 0.005 에서 gate 0.5, 0.05 에서 약 0.011). AD-6 의 보정은
  차원 분석으로 맞고, 실제 갱도에서의 효과는 GPU 없이 측정되지 않았다 (§13).
* 고정된 pycolmap fork 는 **numpy ≥ 2 에서 import 자체가 실패**한다 (`np.uint64(-1)`); 그래서
  `examples/requirements.txt` 가 `numpy<2.0.0` 을 고정한다. `docker/Dockerfile.gpu` 는 그 뒤에
  `pip install -e .` (extra 없음) 를 하므로 numpy 1.x 가 유지된다. MineGS 의 `video`/`all` extra 는
  같은 배포 이름의 공식 `pycolmap` 을 끌어와 fork 를 대체할 수 있다 — GPU 이미지에 그 extra 를 넣지
  말 것 (변경하지 않고 기록).
* upstream depth loss 에서 이미지의 track 이 **아예 없으면** KeyError, track 은 있는데 필터 후 0 개면
  `F.l1_loss` 가 빈 텐서에서 **NaN** 이 된다. MineGS 항은 둘 다 0 기여로 처리한다 (AD-4).
* 정수 픽셀 인덱스의 샘플이 덮이지 않은 픽셀 (ED = 0) 위에 있으면 upstream 의 `torch.where(1/d)` 는
  `grid_sample` backward 에서 이웃 픽셀로 NaN 을 흘릴 수 있다 (CPU torch 로 재현). MineGS 항은 분모를
  먼저 안전화하므로 NaN gradient 가 없다 (테스트).
* ED 는 alpha 정규화 값이라 alpha 가 아주 작은 픽셀도 큰 깊이를 낸다. upstream 은 alpha 로 거르지
  않고 MineGS 항도 거르지 않는다 (upstream 의미 유지). `d ≤ 0` 샘플은 disparity 0 으로 평균에 남아 값만
  키우고 gradient 는 없다 — upstream 과 같다.
* 정규화 T2 의 고유벡터 부호는 numpy/LAPACK 구현에 따라 다를 수 있다 (`np.linalg.eigh`). §8 의
  equivalence test 는 같은 수치 스택이거나 부호에 견고해야 한다.
* `init_type=random` 의 위험은 과장되었다: MineGS 기본 원점은 station centroid 라 random cube 가 보통
  카메라를 덮는다. 그래도 chunk 학습이나 명시 원점에서는 아니므로 거부는 유지한다 (fail-closed).
* `gsplat.distributed.cli` 는 `OMPI_COMM_WORLD_SIZE` 가 있으면 device 수를 무시한다. adapter 경로의
  `MineGSRunner` 는 `world_size != 1` 을 거부한다. plain upstream 경로 (light) 에는 같은 방어가 없다 —
  상속된 OMPI 환경을 쓰지 말 것 (기록).
* render 해상도 차이 (§2.3-13) 를 고치려면 upstream 이 **첫 이미지 하나의** 크기 비율로 모든 카메라의
  K 를 다시 맞추는 것 (`colmap.py:262-273`) 까지 재현해야 한다.
* upstream 그대로의 결함이라 MineGS 가 고치지 않고 기록하는 것 (모두 light/heavy 공통):
  * DefaultStrategy 의 opacity reset 은 v1.5.3 에서 **발화하지 않는다** —
    `step % self.reset_every == 0 & step > 0` (`default.py:195`) 는 연산자 우선순위 때문에 항상
    거짓이다.
  * `export_splats` 는 1-D opacity 에 `isnan(...).any(dim=0)` 을 써서, opacity 하나가 non-finite 면
    **모든** splat 을 버린다 (`exporter.py:523-524`). 그 run 은 "model holds no gaussians" 로 FAILED
    가 된다 (fail-closed 이지만 원인은 발산이다).
  * MineGS 는 `--disable_video` 를 넘기지 않으므로 eval step 마다 `render_traj` 가
    `camtoworlds[5:-5]` 로 경로를 만든다. staged 이미지가 11 장 이하이면 실패할 수 있다 (작은 합성
    데이터에서만 해당, 실데이터는 아님).

### 14.2 C1 — depth supervision artifact

`minegs/train/supervision/{support,depth,build}.py`, CLI `minegs dataset depth-supervision` /
`depth-supervision-verify`. 계약 §4·§5 그대로. 구현 중 정한 것:

* 검증기는 record 의 `max_radial_m` / `ray_margin_m` 를 쓰되 **계약보다 느슨하면 거부**한다
  (`max_radial_m ≤ 10`, `ray_margin_m ≥ 0.5`). 그렇지 않으면 record 가 자기에게 준 관용으로 판정된다.
* builder 는 자기가 쓴 float32 샘플을 다시 역투영해서 support 를 판정한다 — 검증기와 같은 숫자로.
* TLS projection 의 가림 허용치 기본값은 0.15 (비스듬한 벽에서 인접 셀 깊이가 5 % 이상 다르다).
* sfm_tracks 의 샘플 픽셀은 metric 점을 dataset camera 로 재투영한 위치, 품질 기준 (track 길이,
  관측 keypoint 와의 재투영 오차) 위반은 confidence 0.

### 14.3 C2 — adapter, staging, heavy

* adapter: `minegs/train/trainers/{advanced_gs,depth_term}.py`. depth 항은 numpy 정의와 torch 구현이
  같고 (테스트), 주입한 gradient 는 `loss + term` 의 gradient 와 **정확히** 같으며 loss 값은 바뀌지
  않는다 (테스트, CPU torch).
* CI 에 CPU torch 를 설치하고, `tests/fake_upstream/` (upstream 모양의 stand-in trainer + stub
  `gsplat.distributed`) 로 **실제 adapter 코드**를 끝까지 실행한다: hook, Runner 교체, MCMC 보정
  (stand-in 이 upstream 의 `similarity_from_cameras` 원문을 쓰고 host 는 port 를 써서 비교), depth
  gradient, evidence. stand-in 의 rasteriser 는 toy 다 — real gsplat 학습은 수행되지 않았다.
* staging: `sparse/0/*.bin` (고정된 pycolmap fork 로 직접 읽어 확인: 카메라·이미지·빈 2D 점·track),
  `images_<f>/`, appearance 시 RGB clamp, `use_init_points=False` track 필터, supervision 복사.
  staged hash 패턴에 `*.bin`, `images_*/`, `supervision/` 추가.
* run.json schema 1.2 (1.1 → 1.2 migration 은 아무것도 지어내지 않는다).
* light 의 argv 는 Phase 0D 와 **바이트 단위로 같다** (테스트). light run 도 이제 cfg.yml 이 없거나
  요청과 다르면 FAILED 다 — 이것은 baseline 의 수치를 바꾸지 않고 evidence 만 강화한다.

### 14.4 C3 — baseline ↔ advanced 비교

`minegs/eval/compare/runs.py` (`compare_runs`, `RunComparison`), CLI `minegs eval compare-runs`.

* 입력은 명시적인 artifact 다 (Phase 3 `compare-paths` 와 같은 방식): 두 run 디렉터리, 각 run 의
  surface 에서 자른 section record, 기준 section, 선택적으로 geometry/render 보고서와 Phase 2 e2e 보고서.
* 거부 순서: run 이 succeeded 가 아님 → 다른 dataset (id·hash 를 지금의 dataset 에서 재도출) →
  LOCAL_METRIC 아님 / `T_local_from_internal ≠ I` → 같은 run 두 번 → section 이 그 run 의 surface 에서
  온 것이 아님 → 다른 grid → 다른 기준 → geometry/render 측정 방식 다름 → 선언된 holdout 이 없는데
  범위도 주지 않음.
* 모든 수치는 두 run 이 **모두** 관측한 구간에서 `compare_to_reference` 를 다시 불러 얻는다. 한쪽에 없는
  값은 그쪽과 차이 모두 `null`. training loss 는 읽지 않는다. verdict 필드는 없고 `g3_status:
  PENDING`.
* "real" 은 각 쪽의 e2e 보고서가 `real_gpu_execution` 과 `real_renderer_execution` 을 **둘 다** 참으로
  말할 때만이다. 보고서가 없거나 하나라도 대체되었으면 structural.
* 주석으로 남기는 것: data_factor 가 다르면 renderer 해상도 차이 (§2.3-13), strategy 가 다르면 mcmc
  preset 전체 (init_opa/init_scale/opacity_reg/scale_reg) 가 차이에 섞임, appearance run 의 render
  지표는 zero-embedding 정책을 포함함, 명시 범위면 diagnostic.
