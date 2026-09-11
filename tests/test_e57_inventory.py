"""Phase 0B.1 contract tests: E57 inventory, scan identity, pose validation, stations.

Two layers, both running in CI:

* **Fake libE57 tree** — exercises the contract against vendor variations that would be
  tedious or impossible to obtain as real files (no pose, no colour, no name, no guid,
  mirrored rotation, garbage bounds, unreadable nodes). No pye57 needed.
* **Real E57 written at test time** — pye57 can write E57 files, so the adapter is also run
  against genuine libE57 output without committing a binary fixture to the repository.

No real survey data is used here. Phase 0B's G2 gate is a manual run on the user's own file.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from minegs.core.errors import MissingDependencyError
from minegs.ingest.e57 import inventory as inv_mod
from minegs.ingest.e57.exceptions import (
    E57FileNotFoundError,
    E57NoScansError,
    E57NotAFileError,
    E57ReadError,
    E57UnsupportedStructureError,
)
from minegs.ingest.e57.inventory import inventory
from minegs.ingest.e57.models import (
    POSE_INVALID_TOL,
    E57Inventory,
    make_bounds,
    make_scan_pose,
    scan_id_for,
    station_id_for,
    validate_rotation,
)

# ---------------------------------------------------------------- fake libE57 node tree


class FakeE57Error(Exception):
    """Stands in for libe57.E57Exception: raised for a node that is not defined."""


class FakeLeaf:
    def __init__(self, name: str, value: Any) -> None:
        self._name, self._value = name, value

    def elementName(self) -> str:
        return self._name

    def value(self) -> Any:
        return self._value

    def childCount(self) -> int:
        return 0

    def isDefined(self, key: str) -> bool:
        return False

    def __getitem__(self, key: str) -> Any:
        raise FakeE57Error(key)


class FakeNode:
    """A structure node: ordered children, libE57-style accessors."""

    def __init__(self, name: str = "", children: dict[str, Any] | None = None) -> None:
        self._name = name
        self._children: dict[str, Any] = {}
        for k, v in (children or {}).items():
            self._children[k] = v if isinstance(v, (FakeNode, FakeLeaf)) else FakeLeaf(k, v)

    def elementName(self) -> str:
        return self._name

    def childCount(self) -> int:
        return len(self._children)

    def get(self, index_or_name: Any) -> Any:
        if isinstance(index_or_name, int):
            return list(self._children.values())[index_or_name]
        return self[index_or_name]

    def isDefined(self, key: str) -> bool:
        return key in self._children

    def __getitem__(self, key: str) -> Any:
        if key not in self._children:
            raise FakeE57Error(f"node {key!r} is not defined")
        return self._children[key]

    def __len__(self) -> int:
        return len(self._children)


class ExplodingNode(FakeNode):
    """A node whose enumeration fails, as a corrupt or exotic file's would."""

    def childCount(self) -> int:
        raise FakeE57Error("cannot enumerate")


class FakeHeader:
    def __init__(self, node: FakeNode, point_fields: list[str], point_count: int | None) -> None:
        self.node = node
        self.point_fields = point_fields
        self._point_count = point_count

    @property
    def point_count(self) -> int:
        if self._point_count is None:
            raise FakeE57Error("point count unavailable")
        return self._point_count

    def __getitem__(self, key: str) -> Any:
        return self.node[key]


class FakeE57:
    """Stands in for pye57.E57. Reading point data from it is a test failure by construction."""

    def __init__(self, headers: list[FakeHeader], root: FakeNode) -> None:
        self._headers = headers
        self.root = root
        self.closed = False

    @property
    def scan_count(self) -> int:
        return len(self._headers)

    def get_header(self, index: int) -> FakeHeader:
        return self._headers[index]

    def read_scan(self, *a: Any, **k: Any) -> Any:
        raise AssertionError("inventory must never read point data (§17)")

    def read_scan_raw(self, *a: Any, **k: Any) -> Any:
        raise AssertionError("inventory must never read point data (§17)")

    def close(self) -> None:
        self.closed = True


def _pose_node(quat_wxyz: tuple[float, ...], xyz: tuple[float, ...]) -> FakeNode:
    w, x, y, z = quat_wxyz
    return FakeNode(
        "pose",
        {
            "rotation": FakeNode("rotation", {"w": w, "x": x, "y": y, "z": z}),
            "translation": FakeNode("translation", {"x": xyz[0], "y": xyz[1], "z": xyz[2]}),
        },
    )


def make_scan(
    name: str | None = "Setup 1",
    guid: str | None = "{abc}",
    point_fields: list[str] | None = None,
    point_count: int | None = 1_000,
    pose: FakeNode | None = None,
    bounds: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> FakeHeader:
    children: dict[str, Any] = {}
    if guid is not None:
        children["guid"] = guid
    if name is not None:
        children["name"] = name
    if bounds is not None:
        children["cartesianBounds"] = FakeNode("cartesianBounds", bounds)
    if pose is not None:
        children["pose"] = pose
    children.update(extra or {})
    children["points"] = FakeNode("points", {})
    fields = (
        point_fields
        if point_fields is not None
        else [
            "cartesianX",
            "cartesianY",
            "cartesianZ",
            "intensity",
        ]
    )
    return FakeHeader(FakeNode("data3D", children), fields, point_count)


CARTESIAN_BOUNDS_OK = {
    "xMinimum": -5.0,
    "xMaximum": 5.0,
    "yMinimum": -4.0,
    "yMaximum": 4.0,
    "zMinimum": -1.0,
    "zMaximum": 2.0,
}


@pytest.fixture
def fake_e57(tmp_path, monkeypatch):
    """Install a fake pye57 whose E57 class is built from the headers a test supplies."""

    def install(headers: list[FakeHeader], root: FakeNode | None = None, name: str = "f.e57"):
        path = tmp_path / name
        path.write_bytes(b"not really an e57, the reader is faked")
        opened: list[FakeE57] = []

        class _Module:
            @staticmethod
            def E57(p: str, mode: str = "r") -> FakeE57:
                obj = FakeE57(headers, root if root is not None else FakeNode("root", {}))
                opened.append(obj)
                return obj

        monkeypatch.setattr(inv_mod, "_pye57", lambda: _Module)
        return path, opened

    return install


# ---------------------------------------------------------------- identifiers


def test_scan_and_station_ids_are_deterministic_and_zero_padded():
    assert [scan_id_for(i) for i in (0, 1, 12, 345)] == [
        "scan_000",
        "scan_001",
        "scan_012",
        "scan_345",
    ]
    assert [station_id_for(i) for i in (0, 7)] == ["S000", "S007"]
    with pytest.raises(ValueError):
        scan_id_for(-1)
    with pytest.raises(ValueError):
        station_id_for(-1)


def test_ids_do_not_depend_on_vendor_metadata(fake_e57):
    """A file with no names and no guids still gets stable, usable identifiers."""
    path, _ = fake_e57([make_scan(name=None, guid=None), make_scan(name=None, guid=None)])
    inv = inventory(path, compute_hash=False)
    assert [s.scan_id for s in inv.scans] == ["scan_000", "scan_001"]
    assert [s.name for s in inv.scans] == [None, None]
    assert [s.guid for s in inv.scans] == [None, None]
    assert [c.station_id for c in inv.station_candidates] == ["S000", "S001"]


# ---------------------------------------------------------------- enumeration and flags


def test_multi_scan_inventory_reports_declared_capabilities(fake_e57):
    path, opened = fake_e57(
        [
            make_scan(
                name="Setup 1",
                point_fields=[
                    "cartesianX",
                    "cartesianY",
                    "cartesianZ",
                    "colorRed",
                    "colorGreen",
                    "colorBlue",
                    "intensity",
                    "rowIndex",
                    "columnIndex",
                ],
                point_count=42_312_443,
                pose=_pose_node((1.0, 0.0, 0.0, 0.0), (10.0, 20.0, 1.5)),
                bounds=CARTESIAN_BOUNDS_OK,
            ),
            make_scan(
                name="Setup 2",
                point_fields=["sphericalRange", "sphericalAzimuth", "sphericalElevation"],
                point_count=7,
            ),
        ]
    )
    inv = inventory(path, compute_hash=False)

    assert inv.scan_count == 2 and len(inv.scans) == 2
    a, b = inv.scans
    assert a.point_count == 42_312_443
    assert (a.has_cartesian_xyz, a.has_spherical) == (True, False)
    assert (a.has_rgb, a.has_intensity, a.has_row_column) == (True, True, True)
    assert a.bounds is not None and a.bounds.valid and a.bounds.extent() == (10.0, 8.0, 3.0)

    assert (b.has_cartesian_xyz, b.has_spherical) == (False, True)
    assert (b.has_rgb, b.has_intensity, b.has_row_column) == (False, False, False)
    assert b.bounds is None and b.is_usable_for_points
    assert opened[0].closed, "the E57 handle must be closed"


def test_partial_coordinate_triple_is_not_claimed_as_usable(fake_e57):
    """Two of three cartesian fields is not a coordinate system."""
    path, _ = fake_e57([make_scan(point_fields=["cartesianX", "cartesianY", "intensity"])])
    inv = inventory(path, compute_hash=False)
    s = inv.scans[0]
    assert not s.has_cartesian_xyz and not s.has_spherical and not s.is_usable_for_points
    assert any("neither a full cartesian" in i for i in s.issues)
    assert any("no usable coordinate fields" in i for i in inv.issues)


def test_missing_optional_metadata_does_not_crash(fake_e57):
    """No name, no guid, no pose, no bounds, unknown point count: all recorded as absent."""
    path, _ = fake_e57([make_scan(name=None, guid=None, point_count=None, bounds=None)])
    inv = inventory(path, compute_hash=False)
    s = inv.scans[0]
    assert s.name is None and s.guid is None and s.point_count is None
    assert s.pose is None and s.has_pose is False and s.bounds is None
    assert any("point count unavailable" in i for i in s.issues)
    assert any("declare no pose" in n for n in inv.notes)


def test_raw_fields_are_preserved_for_diagnosis(fake_e57):
    path, _ = fake_e57(
        [make_scan(point_fields=["cartesianX", "cartesianY", "cartesianZ", "wobble"])]
    )
    s = inventory(path, compute_hash=False).scans[0]
    assert "wobble" in s.raw_point_fields
    assert "points" in s.raw_scan_fields and "guid" in s.raw_scan_fields


def test_vendor_metadata_is_captured_when_present(fake_e57):
    path, _ = fake_e57(
        [make_scan(extra={"sensorVendor": "Leica", "sensorModel": "RTC360", "temperature": 14.5})]
    )
    s = inventory(path, compute_hash=False).scans[0]
    assert s.vendor_metadata == {
        "sensorVendor": "Leica",
        "sensorModel": "RTC360",
        "temperature": 14.5,
    }


# ---------------------------------------------------------------- pose contract


def test_pose_absent_is_none_never_identity(fake_e57):
    """The forbidden behaviour: silently substituting identity for a missing pose (§10)."""
    path, _ = fake_e57([make_scan(pose=None)])
    s = inventory(path, compute_hash=False).scans[0]
    assert s.has_pose is False and s.pose is None


def test_valid_pose_becomes_named_transform(fake_e57):
    """90 deg about z, 3 m translation -> T_source_from_scan, direction stated by the name."""
    c = np.sqrt(0.5)
    path, _ = fake_e57([make_scan(pose=_pose_node((c, 0.0, 0.0, c), (1.0, 2.0, 3.0)))])
    s = inventory(path, compute_hash=False).scans[0]
    assert s.has_pose and s.pose is not None and s.pose.validation.valid
    assert s.pose.source_frame == "SOURCE" and s.pose.unit == "m"
    assert s.pose.translation_m == [1.0, 2.0, 3.0]
    T = s.pose.se3()  # raises unless the stored matrix is genuinely rigid
    assert np.allclose(T.apply([1.0, 0.0, 0.0]), [1.0, 3.0, 3.0], atol=1e-9)
    assert np.allclose(T.t, [1.0, 2.0, 3.0])


def test_invalid_pose_is_detected_and_reported(fake_e57):
    """A non-unit quaternion well beyond rounding is a problem, not a note."""
    path, _ = fake_e57([make_scan(pose=_pose_node((2.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))])
    inv = inventory(path, compute_hash=False)
    s = inv.scans[0]
    assert s.has_pose and s.pose is not None
    assert s.pose.validation.valid is False
    assert any("not unit length" in i for i in s.pose.validation.issues)
    assert s.issues, "an invalid pose must surface on the scan"
    assert any("invalid pose" in i for i in inv.issues)
    assert inv.usable_scan_count() == 0


def test_non_finite_pose_is_rejected(fake_e57):
    path, _ = fake_e57([make_scan(pose=_pose_node((1.0, 0.0, 0.0, 0.0), (float("nan"), 0.0, 0.0)))])
    s = inventory(path, compute_hash=False).scans[0]
    assert s.pose is not None and not s.pose.validation.valid
    assert any("non-finite" in i for i in s.pose.validation.issues)


def test_rounded_rotation_is_a_note_not_a_failure(fake_e57):
    """Vendors store 4-6 decimals; that must not read the same as a corrupt rotation."""
    path, _ = fake_e57([make_scan(pose=_pose_node((0.7071, 0.0, 0.0, 0.7071), (0.0, 0.0, 0.0)))])
    s = inventory(path, compute_hash=False).scans[0]
    assert s.pose is not None and s.pose.validation.valid
    assert s.pose.validation.normalised and s.pose.validation.warnings
    assert not s.issues
    s.pose.se3()  # orthonormalised, so it is a usable rigid transform


def test_partial_pose_node_is_reported(fake_e57):
    broken = FakeNode(
        "pose", {"translation": FakeNode("translation", {"x": 1.0, "y": 2.0, "z": 3.0})}
    )
    path, _ = fake_e57([make_scan(pose=broken)])
    s = inventory(path, compute_hash=False).scans[0]
    assert any("no rotation" in i for i in s.issues)
    # an undeclared rotation must not be quietly completed with identity
    assert s.pose is not None and not s.pose.validation.valid
    assert any("no rotation" in i for i in s.pose.validation.issues)


def test_identity_pose_is_flagged_as_a_note(fake_e57):
    path, _ = fake_e57(
        [
            make_scan(pose=_pose_node((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0))),
            make_scan(pose=_pose_node((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0))),
        ]
    )
    inv = inventory(path, compute_hash=False)
    assert all(s.pose is not None and s.pose.is_identity for s in inv.scans)
    assert all(not s.issues for s in inv.scans)
    assert any("appears to carry no registration" in n for n in inv.notes)


def test_validate_rotation_grades_deviation():
    assert validate_rotation(np.eye(3), np.array([1.0, 0, 0, 0])).valid
    mirrored = np.diag([1.0, 1.0, -1.0])
    v = validate_rotation(mirrored)
    assert not v.valid and any("determinant" in i for i in v.issues)
    big = validate_rotation(np.eye(3) * (1 + 10 * POSE_INVALID_TOL))
    assert not big.valid
    assert validate_rotation(np.zeros((2, 2))).valid is False


def test_make_scan_pose_never_returns_a_silent_identity():
    """A broken rotation must not be repaired into something that looks fine.

    3*I orthonormalises to exactly I, so a repair step applied to an invalid pose would hand
    back a plausible identity and hide the defect. Repair is for rounding only.
    """
    pose = make_scan_pose(np.eye(3) * 3, [0.0, 0.0, 0.0], np.array([3.0, 0, 0, 0]))
    assert not pose.validation.valid and pose.validation.issues
    assert pose.validation.normalised is False
    assert pose.is_identity is False
    assert pose.T_source_from_scan[0][0] == 3.0, "the file's own numbers are preserved"


# ---------------------------------------------------------------- bounds


def test_impossible_header_bounds_are_reported_not_trusted(fake_e57):
    """pye57's own writer emits min > max; a reader must not propagate that as fact."""
    bad = dict(CARTESIAN_BOUNDS_OK, xMinimum=1.0, xMaximum=-11.0)
    path, _ = fake_e57([make_scan(bounds=bad)])
    s = inventory(path, compute_hash=False).scans[0]
    assert s.bounds is not None and s.bounds.valid is False
    assert any("not credible" in i for i in s.issues)


def test_make_bounds_requires_all_six_values():
    assert make_bounds({"min_x": 0.0, "max_x": 1.0}) is None
    b = make_bounds(
        {"min_x": 0.0, "max_x": 1.0, "min_y": 0.0, "max_y": 2.0, "min_z": 0.0, "max_z": 3.0}
    )
    assert b is not None and b.valid and b.extent() == (1.0, 2.0, 3.0)


# ---------------------------------------------------------------- stations


def test_station_candidates_are_inferred_never_confirmed(fake_e57):
    path, _ = fake_e57([make_scan(), make_scan(), make_scan()])
    inv = inventory(path, compute_hash=False)
    assert len(inv.station_candidates) == 3
    for i, c in enumerate(inv.station_candidates):
        assert c.station_id == station_id_for(i)
        assert c.scan_ids == [scan_id_for(i)]
        assert c.origin == "e57_scan"
        assert c.mapping_status == "inferred_from_scan", "0B.1 must not claim a confirmed mapping"
        assert c.note and "0B.2" in c.note


# ---------------------------------------------------------------- images (detection only)


def test_images_are_detected_but_not_mapped(fake_e57):
    root = FakeNode(
        "root", {"images2D": FakeNode("images2D", {"a": FakeNode("a", {}), "b": FakeNode("b", {})})}
    )
    path, _ = fake_e57([make_scan()], root=root)
    inv = inventory(path, compute_hash=False)
    assert inv.images.has_images2d and inv.images.image_count == 2
    assert "0B.2" in (inv.images.detection_note or "")
    # detection is a count and a flag; nothing associates an image with a station or scan
    assert set(inv.images.model_dump()) == {"has_images2d", "image_count", "detection_note"}
    assert all(c.mapping_status == "inferred_from_scan" for c in inv.station_candidates)
    assert not any("image" in k or "pano" in k for k in inv.station_candidates[0].model_dump())
    assert not any("image" in k or "pano" in k for k in inv.scans[0].model_dump())


def test_absent_and_unreadable_images_structures(fake_e57):
    path, _ = fake_e57([make_scan()], root=FakeNode("root", {}))
    inv = inventory(path, compute_hash=False)
    assert not inv.images.has_images2d and inv.images.image_count == 0
    assert any("no images2D" in n for n in inv.notes)

    path2, _ = fake_e57(
        [make_scan()], root=FakeNode("root", {"images2D": ExplodingNode("images2D")}), name="g.e57"
    )
    inv2 = inventory(path2, compute_hash=False)
    assert inv2.images.detection_note and "not enumerable" in inv2.images.detection_note


# ---------------------------------------------------------------- errors


def test_missing_file_and_directory_errors(tmp_path):
    with pytest.raises(E57FileNotFoundError):
        inventory(tmp_path / "nope.e57")
    with pytest.raises(E57NotAFileError):
        inventory(tmp_path)


def test_zero_scans_is_an_explicit_error(fake_e57):
    path, _ = fake_e57([])
    with pytest.raises(E57NoScansError, match="no readable Data3D scans"):
        inventory(path, compute_hash=False)


def test_unopenable_file_reports_which_file(tmp_path, monkeypatch):
    path = tmp_path / "corrupt.e57"
    path.write_bytes(b"garbage")

    class _Module:
        @staticmethod
        def E57(p: str, mode: str = "r") -> Any:
            raise FakeE57Error("bad file signature")

    monkeypatch.setattr(inv_mod, "_pye57", lambda: _Module)
    with pytest.raises(E57ReadError, match=r"corrupt\.e57"):
        inventory(path)


def test_unreadable_scan_structure_is_reported(fake_e57):
    class NoPrototype(FakeHeader):
        @property
        def point_fields(self) -> list[str]:
            raise FakeE57Error("no prototype")

        @point_fields.setter
        def point_fields(self, v: list[str]) -> None:
            pass

    h = make_scan()
    broken = NoPrototype(h.node, [], 10)
    path, _ = fake_e57([broken])
    with pytest.raises(E57UnsupportedStructureError, match="point prototype"):
        inventory(path, compute_hash=False)


def test_missing_pye57_says_how_to_install(tmp_path, monkeypatch):
    path = tmp_path / "x.e57"
    path.write_bytes(b"")

    def _boom():
        raise MissingDependencyError("pye57", "e57", "reading E57 files")

    monkeypatch.setattr(inv_mod, "_pye57", _boom)
    with pytest.raises(MissingDependencyError, match=r"minegs\[e57\]"):
        inventory(path)


# ---------------------------------------------------------------- safety and provenance


def test_inventory_never_reads_point_data(fake_e57):
    """FakeE57.read_scan* raise; an inventory that touched them would fail here (§17)."""
    path, _ = fake_e57([make_scan(point_count=500_000_000) for _ in range(8)])
    inv = inventory(path, compute_hash=False)
    assert inv.scan_count == 8 and inv.scans[0].point_count == 500_000_000


def test_provenance_records_the_source_file(fake_e57):
    path, _ = fake_e57([make_scan()])
    inv = inventory(path, compute_hash=True)
    assert inv.file.sha256 and len(inv.file.sha256) == 64
    assert inv.file.size_bytes == path.stat().st_size
    assert inv.file.file_name == path.name and inv.file.path == str(path.resolve())
    asset = inv.provenance.source_assets[0]
    assert asset.sha256 == inv.file.sha256 and asset.size_bytes == inv.file.size_bytes
    assert inv.provenance.minegs_version and inv.provenance.created_at
    assert "python" in inv.provenance.tool_versions


def test_skipped_hash_is_recorded_as_skipped(fake_e57):
    path, _ = fake_e57([make_scan()])
    inv = inventory(path, compute_hash=False)
    assert inv.file.sha256 is None
    assert inv.file.hash_skipped_reason, "a report without a hash must say why"


def test_report_round_trips_through_json(fake_e57, tmp_path):
    path, _ = fake_e57(
        [
            make_scan(
                pose=_pose_node((1.0, 0.0, 0.0, 0.0), (1.0, 2.0, 3.0)), bounds=CARTESIAN_BOUNDS_OK
            )
        ]
    )
    inv = inventory(path, compute_hash=False)
    out = inv.save(tmp_path / "inventory.json")
    reloaded = E57Inventory.load(out)
    assert reloaded.schema_version == "1.0"
    assert reloaded.model_dump() == inv.model_dump()
    raw = json.loads(out.read_text())
    assert raw["scans"][0]["pose"]["T_source_from_scan"][0] == [1.0, 0.0, 0.0, 1.0]
    assert raw["scans"][0]["pose"]["source_frame"] == "SOURCE"


def test_report_never_claims_a_tls_global_frame(fake_e57):
    path, _ = fake_e57([make_scan(pose=_pose_node((1.0, 0.0, 0.0, 0.0), (1.0, 2.0, 3.0)))])
    dumped = inventory(path, compute_hash=False).model_dump_json()
    assert "TLS_GLOBAL" not in dumped and "LOCAL_METRIC" not in dumped
    assert "SOURCE" in dumped


# ---------------------------------------------------------------- real libE57 round trip

pye57 = pytest.importorskip("pye57", reason="real-E57 round trip needs minegs[e57]")


def _write_real_e57(path, scans):
    """Write a genuine (tiny) E57 with pye57 so the adapter meets real libE57 output."""
    f = pye57.E57(str(path), mode="w")
    try:
        for spec in scans:
            n = spec.get("n", 5)
            data = {
                "cartesianX": np.linspace(-1, 1, n),
                "cartesianY": np.linspace(0, 2, n),
                "cartesianZ": np.zeros(n),
            }
            if spec.get("rgb"):
                data.update(
                    colorRed=np.full(n, 10.0),
                    colorGreen=np.full(n, 20.0),
                    colorBlue=np.full(n, 30.0),
                )
            if spec.get("intensity"):
                data["intensity"] = np.linspace(0, 1, n)
            f.write_scan_raw(
                data,
                name=spec.get("name", "Scan"),
                rotation=np.asarray(spec.get("rotation", [1.0, 0.0, 0.0, 0.0])),
                translation=np.asarray(spec.get("translation", [0.0, 0.0, 0.0])),
            )
    finally:
        f.close()
    return path


def test_real_e57_round_trip(tmp_path):
    path = _write_real_e57(
        tmp_path / "real.e57",
        [
            {
                "name": "Setup A",
                "rgb": True,
                "intensity": True,
                "n": 9,
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "translation": [12.0, -3.0, 1.25],
            },
            {"name": "Setup B", "n": 4},
        ],
    )
    inv = inventory(path)

    assert inv.scan_count == 2
    assert [s.scan_id for s in inv.scans] == ["scan_000", "scan_001"]
    assert [c.station_id for c in inv.station_candidates] == ["S000", "S001"]
    a, b = inv.scans
    assert a.name == "Setup A" and a.point_count == 9 and a.guid
    assert a.has_cartesian_xyz and a.has_rgb and a.has_intensity and not a.has_spherical
    assert not b.has_rgb and b.point_count == 4
    assert a.pose is not None and a.pose.translation_m == [12.0, -3.0, 1.25]
    assert a.pose.validation.valid and a.pose.source_frame == "SOURCE"
    assert np.allclose(a.pose.se3().t, [12.0, -3.0, 1.25])
    assert b.pose is not None and b.pose.is_identity
    assert inv.file.sha256 and inv.file.size_bytes > 0
    assert inv.file.e57_library_version and "pye57" in inv.file.e57_library_version


def test_real_e57_with_zero_scans(tmp_path):
    path = tmp_path / "empty.e57"
    f = pye57.E57(str(path), mode="w")
    f.close()
    with pytest.raises(E57NoScansError):
        inventory(path)


def test_real_e57_inventory_does_not_read_points(tmp_path, monkeypatch):
    path = _write_real_e57(tmp_path / "guard.e57", [{"name": "A", "n": 6}])

    def _forbidden(*a: Any, **k: Any) -> Any:
        raise AssertionError("inventory must never read point data (§17)")

    monkeypatch.setattr(pye57.E57, "read_scan", _forbidden, raising=False)
    monkeypatch.setattr(pye57.E57, "read_scan_raw", _forbidden, raising=False)
    assert inventory(path, compute_hash=False).scans[0].point_count == 6
