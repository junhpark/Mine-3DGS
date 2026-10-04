"""Phase 5 test scene: a long straight synthetic tunnel cut into three chunks.

About 300 m of axis, a TLS station every 15 m (a six-crop ring each, every fourth one a test
group), and one video segment from 75 to 165 m that crosses the 100 m chunk boundary. A geometry
holdout sits inside chunk 0's core, inside chunk 1's overlap. Structure only: nothing here is a
performance result.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from minegs.core.manifest import Manifest
from minegs.core.synthetic import SyntheticSpec, generate

CORE_M = 100.0
OVERLAP_M = 20.0
HOLDOUT = (92.0, 98.0)


def long_tunnel(root: Path, *, images_excluded: bool = False) -> SimpleNamespace:
    r = generate(
        Path(root),
        SyntheticSpec(
            dataset_id="p5_long_tunnel",
            length_m=300.0,
            station_spacing_m=15.0,
            image_size=32,
            points_per_m=300,
            curvature_deg_per_m=0.0,
            holdout_ranges_m=(HOLDOUT,),
        ),
    )
    if images_excluded:
        m = Manifest.load_dataset(r.dataset_dir)
        assert m.split.geometry_holdout is not None
        m.split.geometry_holdout.images_excluded = True
        m.save_dataset(r.dataset_dir)
    return SimpleNamespace(
        dataset_dir=r.dataset_dir, cloud=r.root / "raw" / "tls_full.ply", result=r
    )
