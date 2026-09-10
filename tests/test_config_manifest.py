import json

import numpy as np
import pydantic
import pytest
from minegs.core.config import MigrationRegistry, VersionedModel, config_hash
from minegs.core.errors import ContractError
from minegs.core.manifest import Manifest, validate_layout


def _minimal(**over):
    d = {
        "schema_version": "1.0",
        "dataset_id": "t",
        "coordinate_frames": {"T_tls_from_local": np.eye(4).tolist()},
        "capture_groups": {
            "S01": {"type": "tls_station", "members": ["a.jpg"], "chainage_m": 1.0},
            "S02": {"type": "tls_station", "members": ["b.jpg"], "chainage_m": 20.0},
        },
        "split": {"train_groups": ["S01"], "test_groups": ["S02"]},
        "initialization": {"source": "tls", "groups": ["S01"]},
        "provenance": {"minegs_version": "0.1.0"},
    }
    d.update(over)
    return d


def test_manifest_minimal_and_hash_stable():
    m = Manifest.from_dict(_minimal())
    assert m.T_tls_from_local.is_identity()
    assert config_hash(m.model_dump(mode="json")) == config_hash(
        Manifest.from_dict(_minimal()).model_dump(mode="json")
    )
    assert m.consistency_issues() == ["scale.basis missing: geometry evaluation will refuse to run"]


def test_manifest_rejects_bad_references_and_overlap():
    with pytest.raises(Exception, match="unknown groups"):
        Manifest.from_dict(_minimal(split={"train_groups": ["S01"], "test_groups": ["S09"]}))
    with pytest.raises(Exception, match="both train and test"):
        Manifest.from_dict(_minimal(split={"train_groups": ["S01"], "test_groups": ["S01"]}))
    bad = _minimal()
    bad["capture_groups"]["S02"]["members"] = ["a.jpg"]
    with pytest.raises(Exception, match="belongs to groups"):
        Manifest.from_dict(bad)
    with pytest.raises(Exception, match="rigid"):
        Manifest.from_dict(
            _minimal(coordinate_frames={"T_tls_from_local": (np.eye(4) * 2).tolist()})
        )


def test_manifest_extra_fields_forbidden():
    with pytest.raises(pydantic.ValidationError):
        Manifest.from_dict(_minimal(stations={}))


def test_manifest_leak_detection():
    m = Manifest.from_dict(
        _minimal(
            initialization={"source": "tls", "groups": ["S01", "S02"]}, scale={"basis": "tls_pose"}
        )
    )
    issues = m.consistency_issues()
    assert any("test groups" in i for i in issues)
    m2 = Manifest.from_dict(
        _minimal(
            scale={"basis": "tls_pose"},
            split={
                "train_groups": ["S01"],
                "test_groups": ["S02"],
                "geometry_holdout": {"chainage_ranges_m": [[5, 10]], "points_excluded": True},
            },
        )
    )
    assert any("excluded_chainage_ranges_m" in i for i in m2.consistency_issues())


def test_manifest_migration_0_9_to_1_0(tmp_path):
    old = {
        "schema_version": "0.9",
        "dataset_id": "old",
        "stations": {"S01": {"images": ["a.jpg"], "chainage_m": 3.0}, "S02": {"images": ["b.jpg"]}},
        "split": {"train_groups": ["S01"], "test_groups": ["S02"], "holdout_stations": ["S02"]},
        "provenance": {"minegs_version": "0.0.1"},
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(old))
    m = Manifest.load(p)
    assert m.schema_version == "1.0"
    assert m.capture_groups["S01"].type == "tls_station"
    assert m.capture_groups["S01"].chainage_m == 3.0
    assert m.split.geometry_holdout is None  # station-level holdout dropped (not leak-proof)


def test_newer_schema_refused(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps(_minimal(schema_version="9.0")))
    with pytest.raises(ContractError, match="newer"):
        Manifest.load(p)


def test_migration_registry_chain():
    reg = MigrationRegistry("x")

    @reg.register("1.0", "1.1")
    def a(d):
        d["a"] = 1
        return d

    @reg.register("1.1", "2.0")
    def b(d):
        d["b"] = d["a"] + 1
        return d

    out = reg.migrate({"schema_version": "1.0"}, "2.0")
    assert out == {"schema_version": "2.0", "a": 1, "b": 2}
    with pytest.raises(ContractError):
        reg.migrate({"schema_version": "0.5"}, "2.0")


def test_versioned_model_yaml_roundtrip(tmp_path):
    class Cfg(VersionedModel):
        SCHEMA_VERSION = "1.0"
        name: str
        n: int = 3

    c = Cfg(name="x")
    p = c.save(tmp_path / "c.yaml")
    assert Cfg.load(p) == c
    assert Cfg.load(p).config_hash() == c.config_hash()


def test_layout_validation(synthetic):
    assert validate_layout(synthetic.dataset_dir, synthetic.manifest) == []
    (synthetic.dataset_dir / "sparse" / "0" / "points3D.txt").rename(
        synthetic.dataset_dir / "sparse" / "0" / "p.bak"
    )
    try:
        assert any("points3D" in p for p in validate_layout(synthetic.dataset_dir))
        with pytest.raises(ContractError):
            Manifest.load_dataset(synthetic.dataset_dir)
    finally:
        (synthetic.dataset_dir / "sparse" / "0" / "p.bak").rename(
            synthetic.dataset_dir / "sparse" / "0" / "points3D.txt"
        )
