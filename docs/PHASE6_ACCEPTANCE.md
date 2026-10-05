# Phase 6 live acceptance runbook — RunPod

Product Owner 가 **한 번** 실제 RunPod 에서 수행하는 최소 절차. 비용이 발생한다. 이 절차를 마치면
`Live RunPod execution: PERFORMED` 이고, 4 단계 비교까지 하면 `Local↔RunPod reproducibility: evaluated`
다. **G3 PASS 는 아니다** — 허용범위를 freeze 하고 그것을 통과해야 G3 다 (§6).

계약: [PHASE6_CONTRACT.md](PHASE6_CONTRACT.md).

## 0. 준비 (비용 없음)

1. **이미지.** 한 번 빌드해 push 하고 digest 로 고정한다.
   ```bash
   docker build -f docker/Dockerfile.gpu --build-arg MINEGS_GIT_SHA="$(git rev-parse HEAD)" \
     -t <registry>/minegs:gpu .
   docker push <registry>/minegs:gpu        # 출력된 sha256 digest 를 적는다
   ```
   `configs/runner/local.yaml` 과 `runpod.yaml` 의 `image:` 를 **같은** `<registry>/minegs:gpu@sha256:<digest>`
   로. tag 만 쓰면 RunPod 실행은 거부된다. `MINEGS_GIT_SHA` 없이 빌드한 이미지는 code 가 `unknown` 이라 재현성
   비교에서 쌍이 되지 않는다.
2. **Network volume.** RunPod 콘솔에서 network volume 하나 (데이터 센터는 쓸 GPU 가 있는 곳). id 를
   `runpod.yaml` 의 `network_volume_id` 에. `volume_mount: /data`.
3. **Storage remote.** 이 머신에서 같은 volume 에 닿는 rclone remote (RunPod 의 network volume S3 호환
   endpoint) 를 `rclone config` 로 만들고 `sync.remote: <remote>:<volume id>` 에. 자격증명은 rclone 설정에만
   있고 MineGS 는 기록하지 않는다.
4. **API key.** `export RUNPOD_API_KEY=...` (파일에 쓰지 않는다).
5. **dataset.** 짧은 합성 dataset:
   ```bash
   minegs dataset synthetic data/p6 --length-m 60 --n-stations 4
   ```
6. **dry run** — 비용 없이 요청 전체를 확인한다 (upload 없음, pod 없음):
   ```bash
   minegs train run data/p6/dataset --profile light --runner runpod \
     --config configs/runner/runpod.yaml --dry-run
   ```
   `pod_requests[].image` 가 digest 참조인지, `gpu_count: 1`, `docker_args` 가
   `train remote-worker /data/minegs/<dataset_id>/jobs/<run_id>/inputs.json` 인지, `inputs.dataset_hash` 가
   `minegs dataset validate` 의 hash 와 같은지 본다.

## 1. 로컬 GPU light run

```bash
minegs train run data/p6/dataset --profile light --runner local --config configs/runner/local.yaml
```

`runs/<local_id>/run.json`: `status: succeeded`, `docker_digest` = 위 digest, `runtime.gpu_model`.

## 2. RunPod light run (비용 발생)

```bash
minegs train run data/p6/dataset --profile light --runner runpod --config configs/runner/runpod.yaml
```

확인:

* `runs/<remote_id>/run.json`: `runner: runpod`, `status: succeeded`, 같은 `docker_digest`,
  `remote_sync.pod_dataset_hash == remote_sync.local_dataset_hash == dataset_hash`.
* `runs/<remote_id>/remote.json`: `artifact_sync_verified: true`, `job_status.exit_code: 0`,
  `job_status.trainer_exit_code: 0`, `provider.gpu_display_name` (실제 GPU), `terminated_at` (자동 종료).
* RunPod 콘솔에서 pod 가 종료됐는지 확인한다 (`terminate_on_completion: true`). 종료되지 않았다면
  `minegs train cancel runs/<remote_id> --config configs/runner/runpod.yaml`.
* 중단·실패 시: `minegs train fetch runs/<remote_id> --config ...` 가 volume 의 `status.json` 을 읽어
  상태를 정한다. pod 가 사라진 것은 성공이 아니다.

## 3. 비교

```bash
minegs eval compare-execution runs/<local_id> runs/<remote_id> --out runs/p6_compare
```

`reproducibility_pair: true` 여야 한다 (같은 dataset hash · code SHA · image digest · profile · trainer 요청 ·
step). 아니면 `differences` 를 고치고 다시 한다. 쌍이면 `sides` 에 gaussian 수, 최종 step, 시간, 메모리,
GPU 가 나란히 나온다. **허용범위는 아직 없다** — 이 결과로 정한다.

## 4. 기록

PR 또는 ROADMAP 에: 날짜, image digest, code SHA, dataset hash, local GPU, RunPod GPU, 두 run id, 비교
결과, 이상 소견. 그 뒤 성숙도:

```
Live RunPod execution:              PERFORMED
Real RunPod GPU training:           PERFORMED
Local↔RunPod reproducibility:       EVALUATED (tolerance not frozen)
Phase 6 G3:                         PENDING
```

## 5. 선택: 테스트 형태의 live smoke

```bash
MINEGS_RUNPOD_LIVE=1 MINEGS_RUNPOD_CONFIG=configs/runner/runpod.yaml \
MINEGS_RUNPOD_DATASET=data/p6/dataset pytest -m runpod_live tests/test_phase6_live.py
```

`-m runpod_live` 와 `MINEGS_RUNPOD_LIVE=1` 둘 다 있어야 돈다. 일반 `pytest` 와 CI 에서는 skip 된다.

## 6. G3 (이 runbook 다음)

허용범위 freeze (예: gaussian 수·geometry/volume 평가 차이) → 같은 실험을 다시 local/RunPod 에서 →
허용범위 안이면 Phase 6 G3 PASS. heavy·chunk 는 그 뒤에 같은 절차로 확장한다.
