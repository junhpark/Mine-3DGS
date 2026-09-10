"""Viser research viewer (§12): frustums, init points, splats, TLS toggle, distance heatmap,
section slider, chainage scrubber. Python API over WebGL; FastAPI multi-user later (Phase 4).

Optional dependency: ``pip install 'minegs[viz]'``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from minegs.core.errors import MissingDependencyError
from minegs.core.manifest import Manifest
from minegs.core.pointcloud import read_ply
from minegs.ingest.common import colmap_io


def launch(
    dataset_dir: str | Path,
    run_ply: str | Path | None = None,
    tls_ply: str | Path | None = None,
    port: int = 8080,
    max_points: int = 1_000_000,
    block: bool = True,
):
    try:
        import viser
    except ImportError as e:
        raise MissingDependencyError("viser", "viz", "the web viewer") from e

    dataset_dir = Path(dataset_dir)
    manifest = Manifest.load_dataset(dataset_dir)
    model = colmap_io.read_model(dataset_dir / "sparse" / "0")
    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("+z")

    init = read_ply(dataset_dir / manifest.initialization.file).subsample(max_points)
    h_init = server.scene.add_point_cloud(
        "/init_points",
        init.xyz.astype(np.float32),
        init.rgb if init.rgb is not None else np.full((len(init), 3), 180, np.uint8),
        point_size=0.01,
    )

    h_frusta = []
    test_set = set(manifest.test_images())
    for im in model.images.values():
        cam = model.cameras[im.camera_id]
        K = cam.K()
        fov = 2 * np.arctan(cam.height / (2 * K[1, 1]))
        wfc = im.world_from_cam
        h_frusta.append(
            server.scene.add_camera_frustum(
                f"/cameras/{im.name}",
                fov=float(fov),
                aspect=cam.width / cam.height,
                scale=0.3,
                wxyz=wfc.quat(),
                position=wfc.t,
                color=(255, 80, 80) if im.name in test_set else (80, 160, 255),
            )
        )

    h_splat = None
    if run_ply:
        pc = read_ply(run_ply)
        if pc.is_gaussian_cloud():
            from minegs.viz.export import gaussian_attributes

            a = gaussian_attributes(pc)
            cov = _covariances(a["scale"], a["rot"])
            h_splat = server.scene.add_gaussian_splats(
                "/splats",
                centers=a["xyz"],
                rgbs=a["rgb"].astype(np.float32) / 255,
                opacities=a["opacity"][:, None],
                covariances=cov,
            )
        else:
            h_splat = server.scene.add_point_cloud(
                "/run_points",
                pc.xyz.astype(np.float32),
                pc.rgb if pc.rgb is not None else np.full((len(pc), 3), 200, np.uint8),
                point_size=0.01,
            )

    h_tls = None
    if tls_ply:
        tls = read_ply(tls_ply)
        if tls.frame == "TLS_GLOBAL":
            tls = tls.transformed(manifest.T_local_from_tls, "LOCAL_METRIC")
        tls = tls.subsample(max_points)
        h_tls = server.scene.add_point_cloud(
            "/tls",
            tls.xyz.astype(np.float32),
            np.full((len(tls), 3), (120, 220, 120), np.uint8),
            point_size=0.008,
        )

    with server.gui.add_folder("Layers"):
        cb_init = server.gui.add_checkbox("init points", True)
        cb_cam = server.gui.add_checkbox("frustums", True)
        cb_splat = server.gui.add_checkbox("splats / run", h_splat is not None)
        cb_tls = server.gui.add_checkbox("TLS", h_tls is not None)

    @cb_init.on_update
    def _(_):
        h_init.visible = cb_init.value

    @cb_cam.on_update
    def _(_):
        for h in h_frusta:
            h.visible = cb_cam.value

    @cb_splat.on_update
    def _(_):
        if h_splat is not None:
            h_splat.visible = cb_splat.value

    @cb_tls.on_update
    def _(_):
        if h_tls is not None:
            h_tls.visible = cb_tls.value

    if manifest.centerline is not None:
        from minegs.core.centerline import Centerline

        cl_path = dataset_dir / manifest.centerline.file
        if cl_path.exists():
            cl = Centerline.from_csv(cl_path, manifest.centerline.frame)
            if manifest.centerline.frame == "TLS_GLOBAL":
                cl = cl.transformed(manifest.T_local_from_tls, "LOCAL_METRIC")
            server.scene.add_spline_catmull_rom(
                "/centerline", cl.vertices.astype(np.float32), color=(255, 200, 0), line_width=3
            )
            with server.gui.add_folder("Chainage"):
                slider = server.gui.add_slider(
                    "s (m)", min=cl.s_start, max=cl.s_end, step=0.5, initial_value=cl.s_start
                )
                marker = server.scene.add_icosphere(
                    "/chainage_marker",
                    radius=0.15,
                    color=(255, 200, 0),
                    position=cl.point_at(cl.s_start),
                )

            @slider.on_update
            def _(_):
                marker.position = cl.point_at(float(slider.value))

    print(f"[minegs viz] http://localhost:{port}  dataset={manifest.dataset_id}")
    if block:
        import time

        while True:
            time.sleep(1.0)
    return server


def _covariances(scale: np.ndarray, rot_wxyz: np.ndarray) -> np.ndarray:
    from minegs.core.frames import quat_to_rotmat

    n = len(scale)
    cov = np.empty((n, 3, 3), dtype=np.float32)
    for i in range(n):
        R = quat_to_rotmat(rot_wxyz[i])
        S = np.diag(scale[i])
        cov[i] = R @ S @ S @ R.T
    return cov
