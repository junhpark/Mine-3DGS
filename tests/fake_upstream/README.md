# fake_upstream — a stand-in for gsplat v1.5.3 `examples/`, for tests only

The Phase 4 trainer adapter (`minegs.train.trainers.advanced_gs`) runs upstream's
`examples/simple_trainer.py` as it is and hooks `gsplat.distributed.cli`. CI has no CUDA and no
gsplat, so these tests point it at this directory instead:

* `simple_trainer.py` has upstream's shape: a `Config` dataclass with v1.5.3 defaults and the
  `default`/`mcmc` presets, a `Parser` that refuses what upstream's refuses (no `.bin` model,
  no `images_<factor>/`), a `Runner` with `rasterize_splats` / `train`, and a `main` launched
  through `gsplat.distributed.cli`. Its rasteriser is a toy: a differentiable function of the
  splats, not gsplat's. It writes the files v1.5.3 writes.
* `gsplat/` is a stub of the two modules the adapter and the trainer import.
* `datasets/normalize.py` is v1.5.3's `similarity_from_cameras`, copied verbatim
  (Apache-2.0, nerfstudio-project/gsplat), so the adapter's MCMC rescaling runs on upstream's
  own function while the host check uses MineGS's port.

Nothing here is evidence that real gsplat trains. Runs launched against it are substituted
runs and are recorded as such.
