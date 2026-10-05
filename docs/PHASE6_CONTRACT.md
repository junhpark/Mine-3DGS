# Phase 6 — RunPod / Reproducible Compute 계약 (freeze)

구현 **전에** 고정한다. 비어 있던 곳은 §13 구현 기록에 남긴다. 범위·gate 는
[ROADMAP.md](ROADMAP.md) §Phase 6, 불변식은 [ARCHITECTURE.md](ARCHITECTURE.md).

## 0. 시작 기록 (2026-10-05)

| 항목 | 값 |
|---|---|
| base | `main @ 03affe85402bd4c8cf8c41f5b58c2b639e6e742e` (PR #17 merge) |
| branch | `phase-6-runpod-reproducible-compute` |
| 테스트 수 (base) | 905 collected (torch 없이) + torch 모듈 16 |
| `RunRecord` | schema 1.3 (Phase 5 chunk binding) |
| `RunnerConfig` | schema 1.0, migration 없음. `network_volume_id`·`volume_mount`·`gpu_types`·`container_disk_gb`·`sync` 필드는 있으나 아무도 읽지 않음 |
| GPU image | `docker/Dockerfile.gpu` (CUDA 12.4, torch 2.4.1, gsplat 1.5.3, `TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"`), LocalRunner docker 경로만 digest 를 강제 |
| RunPod SDK | `runpod>=1.6` (optional, 상한 없음), CI·테스트 환경에 미설치 |
| sync | `minegs/train/runner/sync.py`: `rclone sync --include` 6 패턴 (provenance 없음), pull 은 `point_cloud/log/run.json/stats` 만 (ckpt 없음), 무결성 검증 없음 |
| `RunPodRunner` | `submit`/`terminate` 가 `NotYetImplementedError` |

시작 성숙도 (이 문서가 바꾸지 않는다): Phase 3 manual acceptance DEFERRED · G2 PENDING; Phase 4
COMPLETE / structural PASS / real GPU NOT PERFORMED / G3 PENDING; Phase 5 COMPLETE / structural PASS /
real GPU chunk NOT PERFORMED / real long tunnel NOT VALIDATED / G3 PENDING.

## 1. 목표와 non-goals

**목표.** 같은 dataset·code·GPU image digest·profile·과학 설정이면 LocalRunner 와 RunPodRunner 가
같은 계산을 한다. RunPod 는 compute provider 이지 새 과학 backend 가 아니다: pod 안에서 도는 것은
LocalRunner 와 **같은** staging·backend·trainer·postcondition·`RunRecord` 코드다.

**non-goals.** training resume·checkpoint continuation (0D.3 fail-closed 유지), multi-GPU·multi-node,
spot recovery·pod migration·자동 retry·job queue·DAG·Kubernetes·Serverless·Flash·Slurm·타 클라우드,
GPU 가격 최적화·입찰, Phase 7/8, 새 backend·loss·metric.

## 2. C0 reality audit

| Q | 실제 | 결정 |
|---|---|---|
| Q1 control API | `runpod==1.12.0` 소스 (`runpod/api/ctl_commands.py`) 를 설치해 읽었다. `create_pod(name, image_name, gpu_type_id, cloud_type, support_public_ip, start_ssh, data_center_id, country_code, gpu_count, volume_in_gb, container_disk_in_gb, min_vcpu_count, min_memory_in_gb, docker_args: str, ports, volume_mount_path, env: dict, template_id, network_volume_id, allowed_cuda_versions, min_download, min_upload, instance_id) -> dict`, `get_pod(pod_id, api_key=None)`, `terminate_pod(pod_id)`. 인증은 모듈 전역 `runpod.api_key` (없으면 `AuthenticationError`), HTTP `Authorization: Bearer`. `get_pod` 응답: `id, desiredStatus, imageName, gpuCount, machine{gpuDisplayName}, costPerHr, uptimeSeconds, lastStatusChange, containerDiskInGb, volumeMountPath, env, runtime` — **container exit code 가 없다**. `docker_args`·`env` 값은 GraphQL 문자열에 **escape 없이** 끼워진다. `network_volume_id` 를 주면 SDK 가 `get_user()` 로 data center 를 찾는다 | 얇은 provider seam `RunPodClient{create_pod, get_pod, terminate_pod}` (§7). SDK dict 는 seam 안에서 `PodInfo` 로 바꾸고 core 로 퍼뜨리지 않는다. 성공은 provider 가 아니라 durable job status 로 (§6). worker 인자와 env 는 `[A-Za-z0-9_./:-]` 만 (따옴표·공백 금지) — escape 없는 interpolation 때문 |
| Q1' SDK version | 1.6 과 1.12 사이에 `get_pod(api_key=)`, `allowed_cuda_versions`, `min_download/upload`, `instance_id` 가 추가됐다 — 호출 signature 가 version 의존 | `runpod>=1.12,<1.13`. 읽은 소스와 테스트한 seam 이 그 범위다. 최신이라서가 아니라 signature 가 근거 |
| Q2 storage | `network_volume_id`·`volume_mount` 는 config 에만 있고 sync 는 별개의 rclone remote 로 push — 그 remote 가 pod 안에 나타나는 단계가 없었다 | **canonical topology 하나: RunPod network volume 이 persistent truth.** pod 는 그 volume 을 `volume_mount` 에 mount 해 직접 읽고 쓴다 (pod 쪽 rclone 없음). 로컬은 같은 volume 을 가리키는 storage remote (`sync.remote`: RunPod network volume 의 S3 호환 endpoint 를 rclone remote 로, 또는 로컬에 mount 된 같은 volume 의 절대 경로) 로 올리고 내린다. container disk 는 scratch 일 뿐 |
| Q2' namespace | 없음 | §4 의 결정적 경로. dataset 은 **content-addressed** (`datasets/<dataset_hash>/`), sidecar 는 digest 로, job/run 은 run id 로 — 서로 다른 dataset·run 이 충돌할 수 없고, 다른 run 이 쓰는 dataset 바이트를 새 push 가 덮을 수 없다 |
| Q3 sync gap | `DATASET_INCLUDE` 에 `provenance/**` 가 없다 (hash 에는 있음): Phase 3 image-only dataset 의 SfM/registration evidence 가 빠져 remote hash 가 달라지고 `sfm_tracks` depth 검증이 실패한다. `sparse/**` 는 hash (`sparse/0/*.txt`) 보다 넓다. pattern 목록이 두 곳에 복제 | upload 대상 = `DATASET_HASH_PATTERNS` 가 hash 하는 **바로 그 파일 목록** (`tree_files`, 한 함수) 을 `--files-from` 으로. pattern 목록 복제 제거 |
| Q3' sidecar | `DepthSupervisionRecord` (`depth_supervision.json` + `samples.npy`; `tls_projection` 의 TLS cloud 는 sha 만 기록 — pod 에 필요 없음), `ChunkPlanRecord` 는 dataset hash 밖 | `RunInputBundle` 에 identity (artifact sha256 / plan digest) 와 remote 경로를 묶는다. pod 가 `verify_depth_supervision` / `verify_chunk_plan` 을 다시 돌리고 identity 를 비교 |
| Q4 image | Dockerfile 은 "같은 digest" 를 말하지만 RunPod 쪽 강제는 없음. image 는 `.git` 없이 `COPY` 로 빌드 → pod 안 `git_commit()` 은 `unknown` | `image_digest() is None` 이면 RunPod 거부 (pod 생성 전). create 요청·bundle·run.json 모두 같은 `repo@sha256:` 참조. build-arg `MINEGS_GIT_SHA` → `ENV MINEGS_GIT_COMMIT`, `git_commit()` 은 git 이 없을 때만 그 값을 쓴다 (없으면 `unknown` — 그러면 재현성 쌍이 아니다) |
| Q5 worker | pod 에서 쓸 진입점 없음 | `minegs train remote-worker <inputs.json>` (hidden). `RemoteWorkerRunner(LocalRunner)` 가 native 경로를 **그대로** 실행하고 `name = "runpod"` 이라 `run.json.runner = "runpod"`. LocalRunner 는 trainer exit code 를 handle 로 노출하는 것 말고 바뀌지 않는다 |
| Q6 exit code | provider 가 주지 않는다 (Q1) | worker 가 volume 에 `jobs/<run_id>/status.json` 을 atomic 하게 쓴다 (§6). pod 가 EXITED/TERMINATED/사라짐 = 성공이 아니다 |
| Q7 restart | RunPod GPU pod 는 container 가 끝나면 다시 시작할 수 있다 | worker 는 `claim` 을 O_EXCL 로 만든다. 두 번째 시작은 학습하지 않는다 (§6.3) |

## 3. 비용·credential 안전

* 개발·`pytest`·CI 는 billable 호출을 하지 않는다: provider 는 seam 뒤의 fake, network volume 은 temp
  directory. 실제 provider 테스트는 `pytest -m runpod_live` **그리고** `MINEGS_RUNPOD_LIVE=1` 둘 다
  있을 때만 (credential 이 있다는 것만으로는 돌지 않는다).
* API key 는 환경변수 `RUNPOD_API_KEY` (SDK/CLI 관례). 없으면 `ContractError` — SDK 인증 오류까지 가지
  않는다. 값은 SDK 호출 직전에 `runpod.api_key` 에만 넣는다.
* 기록 금지 값: `RUNPOD_API_KEY`, `RCLONE_CONFIG_PASS`, `RCLONE_*_SECRET*`/`*_KEY*`/`*_PASS*`,
  `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, registry token, SSH key, Authorization header — run.json,
  status.json, remote.json, bundle, log, command, provenance, 예외 메시지 어디에도. 이름은 기록 가능
  (`credential_source: "RUNPOD_API_KEY"`). provider·rclone 의 오류 문자열은 기록 전 redaction.
* pod 에는 secret 을 넘기지 않는다: worker 는 volume 의 파일만 읽고, pod env 는 비어 있다.

## 4. Remote 경로 (volume root 기준, pod 에서는 `<volume_mount>/` 아래)

```
minegs/<dataset_id>/datasets/<dataset_hash>/        dataset (hash pattern 파일 그대로) + dataset.json (claim)
minegs/<dataset_id>/sidecars/chunk_plans/<plan_digest>/chunk_plan.json
minegs/<dataset_id>/sidecars/depth/<artifact_sha256>/   DepthSupervisionRecord 디렉터리
minegs/<dataset_id>/jobs/<run_id>/inputs.json       RunInputBundle (마지막에 publish)
                                  status.json        worker 의 durable status
                                  claim  worker.log  cancel_requested.json
minegs/<dataset_id>/runs/<run_id>/                  run 디렉터리 (run.json, 산출물, output_manifest.json)
```

* `dataset_id`, `run_id`: `^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$`. 아니면 거부 (경로·worker 인자에 안전하지
  않다). `volume_mount`: 절대 경로, `/` 아님, `..` 없음, 같은 문자 집합.
* write-once: `jobs/<run_id>/` 또는 `runs/<run_id>/` 가 이미 있으면 거부. 같은 hash 의 dataset 재업로드는
  idempotent; `dataset.json` 이 다른 hash 를 말하면 거부.
* stale 읽기 없음: worker 는 bundle 이 이름 붙인 content-addressed 경로만 읽는다.

## 5. `RunInputBundle` (schema 1.0, `inputs.json`)

`run_id, dataset_id, dataset_hash, dataset_files (개수), profile, overrides, backend, chunk {chunk_id, plan_id,
plan_digest, path} | null, depth_supervision {supervision_id, artifact_sha256, path} | null, image (digest
참조), requested_gpu_types, gpu_count (=1), cuda_archs, volume_mount, paths {dataset, job, run}, submitter
{git_commit}, created_at, bundle_digest` — `bundle_digest = sha256(canonical_json(나머지))`.

* 순서: 로컬 검증 → dataset 파일 목록·hash (prepare 의 hash 와 같아야 함; 다르면 그 사이에 바뀐 것) →
  remote 충돌 검사 → dataset upload → sidecar upload → `inputs.partial.json` → `inputs.json` (move). worker
  는 `inputs.json` 없이는 시작하지 않는다.
* raw 거부: 대상은 `manifest.json` 이 있는 dataset 디렉터리여야 하고 (`raw/`, project root 거부), upload
  목록에 raw 확장자 (`.e57 .mp4 .mov .avi .mkv .insv .insp .360 .las .laz`) 파일이 있으면 거부.

## 6. Job status 와 성공

### 6.1 `status.json` (`JobStatus`, schema 1.0, worker 가 atomic write)

`run_id, state ∈ {STARTING, RUNNING, SUCCEEDED, FAILED}, started_at, completed_at, exit_code, trainer_exit_code,
failure_stage, message, input_bundle_digest, pod_dataset_hash, run_record_path, output_manifest_path,
output_manifest_sha256`. `failure_stage ∈ {input_bundle, input_verification, gpu_unsupported,
training_setup, trainer_exit, output_verification, worker_restarted, worker_error}`.

### 6.2 성공 = 전부

`status.json` final **그리고** `state == SUCCEEDED` **그리고** `exit_code == 0` **그리고** `trainer_exit_code == 0`
**그리고** 가져온 `run.json.status == SUCCEEDED` **그리고** output manifest 가 status 의 sha 와 같고 모든 항목이
로컬에서 다시 hash 되어 일치. 하나라도 아니면 성공이 아니다.

### 6.3 Provider lifecycle 대응 (`RunPodHandle.status()`)

| remote | provider | → RunStatus |
|---|---|---|
| final SUCCEEDED + exit 0 | 무관 | pull·검증 통과 시 SUCCEEDED, 아니면 FAILED (stage `pull_verification`) |
| final FAILED / exit ≠ 0 | 무관 | FAILED (worker 의 stage·message) |
| 없음 / STARTING / RUNNING | RUNNING | RUNNING (status 없으면 PENDING) |
| 사용자 terminate, final 없음 | 무관 | CANCELLED |
| 없음 / 비-final | EXITED · TERMINATED · 사라짐 | FAILED (stage `pod_ended_without_final_status`) |
| provider 조회 실패·timeout | — | UNKNOWN (상태를 바꾸지 않는다) |

* `wait(timeout_s)` 의 timeout 은 상태를 바꾸지 않고 pod 를 끄지 않는다.
* worker 재시작: `claim` 이 이미 있으면 학습하지 않는다. final 이 있으면 그 exit code 로 끝나고, 없으면
  FAILED `worker_restarted` (resume 없음) 를 쓴다.
* 자동 종료 (`terminate_on_completion`, 기본 true): final status 와 output manifest 가 volume 에 있을 때만
  pod 를 끈다. 증거는 volume 에 남는다.
* `terminate()`: 의도를 `remote.json` 과 `jobs/<run_id>/cancel_requested.json` 에 기록하고 pod 를 끈다.
  이미 final 이면 그 결과를 덮지 않는다.

## 7. Provider seam

`RunPodClient`: `create_pod(PodSpec) -> PodInfo`, `get_pod(pod_id) -> PodInfo | None`,
`terminate_pod(pod_id)`. `PodSpec{name, image, gpu_type_id, gpu_count=1, network_volume_id,
volume_mount_path, container_disk_gb, docker_args, env={}, cloud_type, allowed_cuda_versions}`;
`PodInfo{pod_id, desired_status, gpu_display_name, gpu_count, image, uptime_seconds, cost_per_hr,
last_status_change}`. `SdkRunPodClient` 만 `runpod` 을 import 한다. 오류는 `ProviderAllocationError` /
`ProviderUnavailable` / `ProviderError` 로 (redaction 후). GPU type 은 요청 목록 순서대로 시도하고 할당
실패 (`ProviderAllocationError`) 일 때만 다음으로 — retry scheduler 가 아니라 GPU 선택이다.

Pod 요청: `image=<repo@sha256:...>`, `gpu_count=1`, `network_volume_id`, `volume_mount_path=volume_mount`,
`container_disk_in_gb`, `start_ssh=False`, `support_public_ip=False`, `docker_args="train remote-worker
<volume_mount>/minegs/<dataset_id>/jobs/<run_id>/inputs.json"` (image ENTRYPOINT 이 `minegs` 다), `env={}`.

GPU 호환: `RunnerConfig.cuda_archs` (기본값 = Dockerfile `TORCH_CUDA_ARCH_LIST`, 테스트가 drift 를 잡는다).
worker 가 학습 전에 `torch.cuda.get_device_capability()` 를 확인하고 보이는 GPU 가 정확히 1 개인지 본다.
GPU 표 hard-code 없음. requested GPU type 과 실제 (`machine.gpuDisplayName`, pod 의 `nvidia-smi`) 는 따로
기록한다 — 이름 체계가 달라 비교하지 않는다.

## 8. Pod 안의 검증 (trainer 전)

dataset hash 재계산 = bundle 의 hash · bundle digest · chunk plan (`verify_chunk_plan` + digest) · depth
(`verify_depth_supervision` + sha) · 보이는 GPU 1 개, 지원 arch · run 디렉터리 없음. 실패하면 trainer 를
실행하지 않고 FAILED + 0 아닌 exit code.

## 9. 산출물과 pull

* worker 는 끝날 때 (성공·실패 모두, run 디렉터리가 있으면) `runs/<run_id>/output_manifest.json`
  (`RemoteOutputRecord` 1.0: `run_id, status, exit_code, dataset_hash, image, run_json_sha256, entries
  [{path, size, sha256}], output_tree_digest, created_at`) 를 atomic 하게 쓰고, 그 sha 를 status 에 넣는다.
  `backend_out/` (scratch) 만 제외.
* 로컬 pull: `<run_dir>` 옆 임시 디렉터리로 받는다 → manifest sha = status 의 sha → 모든 항목 size·sha
  일치, manifest 에 없는 파일 없음 → `output_tree_digest` 재계산 일치 → `run.json` 의 `run_id`·
  `dataset_hash`·`docker_digest`·`runner` 가 bundle 과 일치 → 로컬 run 디렉터리에 우리 것 (`run.json`
  placeholder, `remote.json`) 말고 아무것도 없을 때만 publish (`run.json` 마지막). 하나라도 어긋나면 publish
  하지 않는다.

## 10. Evidence

* `RunRecord` 1.4 (no-op migration, 이전 run 은 null): `remote_execution` (provider, pod_id (pod env
  `RUNPOD_POD_ID`, 없으면 null), requested_gpu_types, gpu_count, network_volume_id, volume_mount, job/run
  path) 와 `remote_sync` (input_bundle_digest, local_dataset_hash (bundle), pod_dataset_hash,
  depth_artifact_sha256, chunk_plan_digest). pod 가 아는 사실만 — run.json 은 output manifest 안에 있으므로
  manifest sha 를 담을 수 없다.
* 로컬 `remote.json` (`RemoteExecutionRecord` 1.0): provider 가 말한 것 (pod id, desired status, GPU 표시명,
  cost_per_hr 는 **요율**로만, uptime), job status 사본, output_manifest_sha256, pulled_output_sha256,
  `artifact_sync_verified`, cancel/terminate 기록, failure_stage. 비용 합계는 만들지 않는다.
* compute provenance (GPU, pod id) 와 과학 provenance (dataset hash, trainer config, depth, frame) 를
  섞지 않는다. 플래그는 셋: `real_gpu_training` (`real_gpu_evidence`, 기존 규칙), `remote_provider_execution`
  (runner runpod + remote_execution), `artifact_sync_verified` (pull 검증). renderer 는 별개다.

## 11. 재현성 비교 (`minegs eval compare-execution`)

fingerprint: dataset_hash, code SHA (`provenance.git_commit`; `unknown`·`-dirty` 면 쌍이 아님), docker digest,
backend name/version, profile (canonical), expected trainer config (canonical), max_steps, depth artifact
sha, chunk (plan_digest, chunk_id). 하나라도 다르면 `reproducibility_pair: false` 와 차이 목록만 — 수치를
나란히 놓지 않는다. 쌍이면 기존 evidence 만 나란히: gaussian_count, observed_final_step, scene_scale,
train_seconds, duration_s, peak_gpu_memory_gb, GPU. 허용 오차는 정하지 않는다 (live G3 commissioning).
bitwise 동일성은 요구하지 않는다. G3: PENDING.

## 12. Chunk 호환

`train chunks --runner runpod` 는 기존 순차 루프가 chunk 마다 `RunPodRunner.submit` 을 부른다 (pod 하나 =
run 하나, pod 재사용 없음). 첫 실패에서 멈춤 (Phase 5). pod 는 bundle 의 remote plan 경로를 받고 identity
(plan_id, plan_digest, dataset hash, centerline sha) 로 검증한다. 가져온 chunk run 의 `chunk.plan_path` 는
pod 경로다 — 로컬 render/surface (`run_views`) 는 기록된 경로가 없으면 dataset 의
`chunks/<plan_id>/chunk_plan.json` 을 digest 로 확인해 쓴다 (경로가 아니라 identity).

## 13. 구현 기록

(C1–C4 에서 채운다.)

## 14. 종료 성숙도 (live RunPod 없이)

```
Phase 6 implementation:             COMPLETE
Structural validation:              PASS
RunPod runner:                      IMPLEMENTED
Verified remote input sync:         IMPLEMENTED
Network-volume workflow:            IMPLEMENTED
Exit-code-based job status:         IMPLEMENTED
Verified artifact return:           IMPLEMENTED
Chunk/advanced-profile support:     IMPLEMENTED
Live RunPod execution:              NOT PERFORMED
Real RunPod GPU training:           NOT PERFORMED
Local↔RunPod reproducibility:       NOT VALIDATED
Phase 6 G3:                         PENDING
Phase 5 G3:                         PENDING
Phase 4 G3:                         PENDING
Phase 3 G2:                         PENDING
```

첫 실제 pod 성공은 `Live RunPod execution: PERFORMED` 일 뿐이다. 같은 실험의 Local↔RunPod 비교를 해야
"evaluated", 허용범위를 freeze 하고 통과해야 G3 PASS 다.
