import json

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
    r = runner.invoke(
        app,
        [
            "eval",
            "volume",
            str(sec),
            str(ds),
            "--design-radius-m",
            "2.4",
            "--out",
            str(tmp_path / "vol.json"),
        ],
    )
    assert r.exit_code == 0, r.output
    vol = json.loads((tmp_path / "vol.json").read_text())["volume"]
    assert vol["claim"] == "volume_accuracy" and vol["valid_section_count"] > 20
    r = runner.invoke(
        app,
        [
            "eval",
            "geometry",
            str(ds / "init_points.ply"),
            str(ds),
            "--tls-ply",
            str(root / "raw" / "tls_full.ply"),
            "--out",
            str(tmp_path / "geo.json"),
        ],
    )
    assert r.exit_code == 0, r.output
    geo = json.loads((tmp_path / "geo.json").read_text())
    assert (
        geo["claim"] == "geometry_accuracy" and geo["completeness"]["clipped_ratio"] > 0.5
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
    assert (
        runner.invoke(
            app,
            [
                "ingest",
                "video",
                "sfm",
                str(tmp_path),
                str(tmp_path / "w"),
                "--dry-run",
                "--mapper",
                "global",
            ],
        ).exit_code
        == 0
    )
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
        ],
    )
    assert r.exit_code == 0 and "[geometry_accuracy]" in r.output


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
