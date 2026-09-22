import json

import pytest
from minegs.cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


def test_cli_phase0a_gate(tmp_path):
    """Synthetic dataset must pass validate / info / protocol / train command / sections / volume."""
    root = tmp_path / "syn"
    r = runner.invoke(
        app,
        [
            "dataset",
            "synthetic",
            str(root),
            "--length-m",
            "60",
            "--n-stations",
            "4",
            "--image-size",
            "32",
        ],
    )
    assert r.exit_code == 0, r.output
    ds = root / "dataset"
    assert runner.invoke(app, ["dataset", "validate", str(ds), "--strict"]).exit_code == 0
    r = runner.invoke(app, ["dataset", "info", str(ds), "--json"])
    assert r.exit_code == 0 and json.loads(r.output)["judgement"]["protocols"] == [
        "novel_view",
        "geometry_holdout",
    ]
    r = runner.invoke(app, ["eval", "protocol", str(ds), "--json"])
    claims = json.loads(r.output)["claims"]
    assert r.exit_code == 0 and "geometry_accuracy" in claims
    # a single-epoch dataset must never advertise a two-epoch claim (Phase 7 pair protocol)
    assert "change_volume" not in claims and "change" not in json.loads(r.output)["protocols"]
    assert "change_volume" not in runner.invoke(app, ["eval", "protocol", str(ds)]).output
    r = runner.invoke(app, ["train", "command", str(ds), "--profile", "light"])
    assert r.exit_code == 0 and "simple_trainer" in r.output
    sec = tmp_path / "sec.json"
    r = runner.invoke(
        app,
        [
            "eval",
            "sections",
            str(root / "raw" / "tls_full.ply"),
            str(ds),
            "--interval-m",
            "2",
            "--thickness-m",
            "0.5",
            "--angle-bins",
            "72",
            "--out",
            str(sec),
        ],
    )
    assert r.exit_code == 0, r.output
    volume_argv = [
        "eval",
        "volume",
        str(sec),
        str(ds),
        "--design-radius-m",
        "2.4",
        "--out",
        str(tmp_path / "vol.json"),
    ]
    # Since Phase 1C a volume_accuracy claim needs sections cut from a verified surface. These
    # were cut from raw/tls_full.ply, which is the TLS reference itself: the dataset protocol
    # allows the claim, the *input* is what cannot carry it (§1C).
    r = runner.invoke(app, volume_argv)
    assert r.exit_code == 2, r.output
    assert "raw point cloud" in r.output and "--diagnostic" in r.output
    r = runner.invoke(app, [*volume_argv, "--diagnostic"])
    assert r.exit_code == 0, r.output
    vol = json.loads((tmp_path / "vol.json").read_text())["volume"]
    assert vol["claim"] == "geometry_diagnostic" and vol["valid_section_count"] > 20
    # ...and the diagnostic number still says exactly which spans it integrated.
    assert vol["coverage"]["coverage_fraction"] == pytest.approx(1.0)
    assert vol["segments"] and not vol["coverage"]["missing_intervals_m"]
    # A bare PLY is a diagnostic input since Phase 1A: claim-bearing geometry needs a surface
    # artifact (§1.7), and init_points.ply is TLS initialisation, not reconstructed surface.
    r = runner.invoke(
        app,
        [
            "eval",
            "geometry",
            str(ds / "init_points.ply"),
            str(ds),
            "--tls-ply",
            str(root / "raw" / "tls_full.ply"),
            "--diagnostic",
            "--out",
            str(tmp_path / "geo.json"),
        ],
    )
    assert r.exit_code == 0, r.output
    geo = json.loads((tmp_path / "geo.json").read_text())
    assert (
        geo["claim"] == "geometry_diagnostic" and geo["completeness"]["clipped_ratio"] > 0.5
    )  # init has a hole in the holdout
    r = runner.invoke(app, ["dataset", "chunks", str(ds), "--length-m", "40", "--overlap-m", "10"])
    assert r.exit_code == 0 and "C02" in r.output


def test_cli_refuses_geometry_claim_on_reconstruction(tmp_path):
    root = tmp_path / "syn"
    assert (
        runner.invoke(
            app,
            [
                "dataset",
                "synthetic",
                str(root),
                "--length-m",
                "45",
                "--n-stations",
                "3",
                "--image-size",
                "32",
            ],
        ).exit_code
        == 0
    )
    ds = root / "dataset"
    m = json.loads((ds / "manifest.json").read_text())
    m["split"] = {"train_groups": list(m["capture_groups"]), "test_groups": []}
    m["initialization"]["groups"] = [
        g for g, v in m["capture_groups"].items() if v["type"] == "tls_station"
    ]
    m["initialization"]["excluded_chainage_ranges_m"] = []
    (ds / "manifest.json").write_text(json.dumps(m))
    r = runner.invoke(
        app,
        [
            "eval",
            "geometry",
            str(ds / "init_points.ply"),
            str(ds),
            "--tls-ply",
            str(root / "raw" / "tls_full.ply"),
        ],
    )
    assert r.exit_code == 3 and "protocol" in r.output.lower()
    r = runner.invoke(
        app,
        [
            "eval",
            "geometry",
            str(ds / "init_points.ply"),
            str(ds),
            "--tls-ply",
            str(root / "raw" / "tls_full.ply"),
            "--diagnostic",
            "--out",
            str(tmp_path / "g.json"),
        ],
    )
    assert (
        r.exit_code == 0
        and json.loads((tmp_path / "g.json").read_text())["claim"] == "geometry_diagnostic"
    )


def test_cli_missing_dependency_exit_code(tmp_path):
    r = runner.invoke(app, ["ingest", "e57", "inventory", str(tmp_path / "x.e57")])
    assert r.exit_code in (4, 2)  # 4 = pye57 missing (CI), 2 = file missing when pye57 present


def test_cli_dry_runs(tmp_path):
    # `video sfm` takes a frame set, not a directory of images: the set is what says which
    # frames selection kept and what they hashed to, and reconstructing from a bare directory
    # is exactly the step that used to let a rejected frame back in (Phase 3 §5).
    refused = runner.invoke(
        app,
        ["ingest", "video", "sfm", str(tmp_path), str(tmp_path / "w"), "--dry-run"],
    )
    assert refused.exit_code == 2
    assert "frame set" in refused.output or "frameset" in refused.output
    assert (
        runner.invoke(
            app, ["ingest", "e57", "tiles", "a.e57", str(tmp_path), "--dry-run"]
        ).exit_code
        == 0
    )
    assert runner.invoke(app, ["ingest", "video", "rig", str(tmp_path / "rig.json")]).exit_code == 0
    assert runner.invoke(app, ["train", "profiles"]).exit_code == 0


def test_cli_prints_the_claim_label(tmp_path):
    """rich treats [foo] as markup: the claim label must be escaped or it never reaches the user."""
    root = tmp_path / "syn"
    assert (
        runner.invoke(
            app,
            [
                "dataset",
                "synthetic",
                str(root),
                "--length-m",
                "45",
                "--n-stations",
                "3",
                "--image-size",
                "32",
            ],
        ).exit_code
        == 0
    )
    ds = root / "dataset"
    r = runner.invoke(
        app,
        [
            "eval",
            "geometry",
            str(ds / "init_points.ply"),
            str(ds),
            "--tls-ply",
            str(root / "raw" / "tls_full.ply"),
            "--diagnostic",
        ],
    )
    assert r.exit_code == 0 and "[geometry_diagnostic]" in r.output


def test_cli_heavy_profile_fails_closed(tmp_path):
    """heavy requires depth_loss, refused under TLS staging; the message names the reason."""
    root = tmp_path / "syn"
    assert (
        runner.invoke(
            app,
            [
                "dataset",
                "synthetic",
                str(root),
                "--length-m",
                "45",
                "--n-stations",
                "3",
                "--image-size",
                "32",
            ],
        ).exit_code
        == 0
    )
    r = runner.invoke(app, ["train", "command", str(root / "dataset"), "--profile", "heavy"])
    assert r.exit_code == 2 and "depth_loss" in r.output and "Phase 4" in r.output
    # light still builds a command
    r = runner.invoke(app, ["train", "command", str(root / "dataset"), "--profile", "light"])
    assert r.exit_code == 0 and "--no-normalize_world_space" in r.output


def test_cli_runpod_is_not_runnable(tmp_path):
    root = tmp_path / "syn"
    assert (
        runner.invoke(
            app,
            [
                "dataset",
                "synthetic",
                str(root),
                "--length-m",
                "45",
                "--n-stations",
                "3",
                "--image-size",
                "32",
            ],
        ).exit_code
        == 0
    )
    # light is runnable in principle, so the RunPod path is what refuses it
    r = runner.invoke(
        app, ["train", "run", str(root / "dataset"), "--profile", "light", "--runner", "runpod"]
    )
    assert r.exit_code == 4 and "Phase 6" in r.output
    # heavy must report ITS OWN reason (depth_loss), not whatever its default runner says first
    r = runner.invoke(app, ["train", "run", str(root / "dataset"), "--profile", "heavy"])
    assert r.exit_code == 2 and "depth_loss" in r.output and "Phase 4" in r.output


def test_cli_phase0c_pipeline(tmp_path):
    """synthetic-staging -> calibrate-camera -> from-e57 -> validate -> info -> golden-gate."""
    import json

    root = tmp_path / "s"
    r = runner.invoke(
        app,
        [
            "dataset",
            "synthetic-staging",
            str(root),
            "--length-m",
            "45",
            "--station-spacing-m",
            "15",
            "--image-size",
            "48",
        ],
    )
    assert r.exit_code == 0, r.output
    conv = tmp_path / "camera_convention.json"
    r = runner.invoke(
        app, ["dataset", "calibrate-camera", str(root / "staging"), "--out", str(conv)]
    )
    assert r.exit_code == 0, r.output
    assert "status=selected" in r.output.replace("\n", "") and conv.is_file()
    cfg = tmp_path / "build.yaml"
    cfg.write_text(
        "schema_version: '1.0'\n"
        "dataset_id: cli_0c\n"
        "source_frame: {mode: explicit_identity}\n"
        f"camera: {{mode: e57_pinhole, convention_file: {conv}}}\n"
        "initialization: {voxel_m: 0.1, max_points: 20000, sparse_max_points: 2000}\n"
    )
    ds = tmp_path / "dataset"
    r = runner.invoke(
        app, ["dataset", "from-e57", str(root / "staging"), str(ds), "--config", str(cfg)]
    )
    assert r.exit_code == 0, r.output
    assert runner.invoke(app, ["dataset", "validate", str(ds), "--strict"]).exit_code == 0
    r = runner.invoke(app, ["dataset", "info", str(ds), "--json"])
    assert r.exit_code == 0 and json.loads(r.output)["judgement"]["protocols"] == ["reconstruction"]
    # a second build into the same place is refused without --overwrite
    r = runner.invoke(
        app, ["dataset", "from-e57", str(root / "staging"), str(ds), "--config", str(cfg)]
    )
    assert r.exit_code == 2 and "--overwrite" in r.output
    gg = tmp_path / "gg"
    r = runner.invoke(
        app,
        ["dataset", "golden-gate", str(ds), "--staging", str(root / "staging"), "--out", str(gg)],
    )
    assert r.exit_code == 0, r.output
    rep = json.loads((gg / "report.json").read_text())
    assert rep["structural_result"] == "pass"
    assert rep["real_data_validation_status"] == "pending_human_inspection"
    assert (gg / "tls_local_metric.ply").is_file()
    # a config without a declared SOURCE frame is refused up front
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "schema_version: '1.0'\ndataset_id: x\ncamera: {mode: e57_pinhole, R_e57cam_from_cam: [[1,0,0],[0,1,0],[0,0,1]]}\n"
    )
    r = runner.invoke(
        app,
        ["dataset", "from-e57", str(root / "staging"), str(tmp_path / "d2"), "--config", str(bad)],
    )
    assert r.exit_code == 2 and "source_frame" in r.output.replace("\n", "")


def test_cli_golden_gate_fails_loudly_on_a_wrong_convention(tmp_path):
    """MAJOR 1: a failing gate exits non-zero, after writing its diagnostics."""
    import json

    root = tmp_path / "s"
    assert (
        runner.invoke(
            app,
            [
                "dataset",
                "synthetic-staging",
                str(root),
                "--length-m",
                "45",
                "--station-spacing-m",
                "15",
                "--image-size",
                "40",
            ],
        ).exit_code
        == 0
    )
    cfg = tmp_path / "wrong.yaml"
    cfg.write_text(
        "schema_version: '1.0'\ndataset_id: wrong\nsource_frame: {mode: explicit_identity}\n"
        "camera: {mode: e57_pinhole, R_e57cam_from_cam: [[1,0,0],[0,1,0],[0,0,1]]}\n"
        "initialization: {voxel_m: 0.1, max_points: 20000, sparse_max_points: 1000}\n"
    )
    ds = tmp_path / "dataset"
    r = runner.invoke(
        app, ["dataset", "from-e57", str(root / "staging"), str(ds), "--config", str(cfg)]
    )
    assert r.exit_code == 0, r.output
    gg = tmp_path / "gg"
    r = runner.invoke(
        app,
        ["dataset", "golden-gate", str(ds), "--staging", str(root / "staging"), "--out", str(gg)],
    )
    assert r.exit_code == 2, r.output
    out = " ".join(r.output.split())
    assert "structural_result=fail" in out and "diagnostics written" in out
    rep = json.loads((gg / "report.json").read_text())
    assert rep["structural_result"] == "fail" and rep["overlays"]
    assert all((gg / p).is_file() for p in rep["overlays"])
    # and a calibration request with too few stations is refused at the CLI
    r = runner.invoke(
        app,
        [
            "dataset",
            "calibrate-camera",
            str(root / "staging"),
            "--out",
            str(tmp_path / "c.json"),
            "--stations",
            "2",
        ],
    )
    assert r.exit_code == 2 and "at least 3" in " ".join(r.output.split())


def test_cli_resume_target_is_explicit_and_fails_closed(tmp_path):
    """--resume-from names a run; a target that cannot be honoured is an error, not a fresh run."""
    r = runner.invoke(app, ["train", "run", "--help"])
    assert r.exit_code == 0 and "--resume-from" in r.output
    # the removed boolean must not linger as a silently-accepted alias
    assert "--resume " not in r.output and "--no-resume" not in r.output
    root = tmp_path / "syn"
    assert (
        runner.invoke(
            app,
            [
                "dataset",
                "synthetic",
                str(root),
                "--length-m",
                "45",
                "--n-stations",
                "3",
                "--image-size",
                "32",
            ],
        ).exit_code
        == 0
    )
    r = runner.invoke(
        app,
        [
            "train",
            "run",
            str(root / "dataset"),
            "--profile",
            "light",
            "--native",
            "--resume-from",
            str(tmp_path / "runs" / "does-not-exist"),
        ],
    )
    # Exit 2, a contract error, and a message that says what to do about it. The refusal comes
    # from the backend declaring it cannot resume, not from the missing directory: gsplat v1.5.3
    # cannot continue training at all, so nothing looks the target up (docs/ROADMAP.md §Phase 0D).
    # Whitespace-normalised because rich wraps to the terminal width, and an assertion that
    # depends on COLUMNS tests the environment rather than the contract.
    out = " ".join(r.output.split())
    assert r.exit_code == 2
    assert "does not support resuming training" in out
    assert "evaluation only" in out and "Phase 0D" in out
    # ...and the refusal happens before anything is written: this invocation used the default
    # run directory (<dataset>/../runs/<run_id>), the path every real user takes.
    assert not (root / "runs").exists()


@pytest.mark.parametrize(
    ("exc_factory", "needle"),
    [
        (lambda: _dep("pye57", "e57", "reading E57 files"), "pip install 'minegs[e57]'"),
        (lambda: _dep("viser", "viz", "the web viewer"), "pip install 'minegs[viz]'"),
        (lambda: _contract("holdout [20.0, 26.0] is outside"), "[20.0, 26.0]"),
    ],
)
def test_an_error_message_reaches_the_user_with_its_brackets_intact(capsys, exc_factory, needle):
    """Rich must not parse the error text: every install remedy names an extra in brackets.

    ``pip install 'minegs[e57]'`` rendered as ``pip install 'minegs'`` — rich read ``[e57]`` as
    a style tag and dropped it, so the one line telling the user how to fix their install told
    them to run a command that changes nothing.
    """
    import typer
    from minegs.cli._common import run_guarded

    exc = exc_factory()

    def boom():
        raise exc

    with pytest.raises(typer.Exit):
        run_guarded(boom)
    printed = " ".join(capsys.readouterr().err.split())  # rich wraps to the terminal width
    assert needle in printed, printed


def _dep(module: str, extra: str, purpose: str):
    from minegs.core.errors import MissingDependencyError

    return MissingDependencyError(module, extra, purpose)


def _contract(message: str):
    from minegs.core.errors import ContractError

    return ContractError(message)
