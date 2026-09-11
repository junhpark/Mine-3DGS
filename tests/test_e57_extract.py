"""Phase 0B.3 contract tests: reading point payloads, masks, poses, images, manifest.

This is the first phase that reads point data, and the defect it exists to prevent is
invisible: a scan whose invalid points sit in the middle of the record array, whose colours
are then taken as "the first N", produces a cloud that opens fine in any viewer with every
colour shifted onto the wrong point. Several tests below construct exactly that file.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from minegs.core.errors import ContractError
from minegs.core.pointcloud import read_ply
from minegs.ingest.e57.exceptions import E57PoseUnusableError, E57ReadScanError
from minegs.ingest.e57.extract import (
    E57ExtractionManifest,
    apply_invalid_state,
    extract,
    read_scan_points,
)

from e57_fakes import (
    ExplodingBlob,
    ExplodingNode,
    FakeNode,
    image_node,
    make_scan,
    pose_node,
    root_with_images,
)

GUID_A = "{aaaa}"
GUID_B = "{bbbb}"

COLOR_LIMITS_255 = FakeNode(
    "colorLimits",
    {
        "colorRedMinimum": 0,
        "colorRedMaximum": 255,
        "colorGreenMinimum": 0,
        "colorGreenMaximum": 255,
        "colorBlueMinimum": 0,
        "colorBlueMaximum": 255,
    },
)
COLOR_LIMITS_16BIT = FakeNode(
    "colorLimits",
    {
        "colorRedMinimum": 0,
        "colorRedMaximum": 65535,
        "colorGreenMinimum": 0,
        "colorGreenMaximum": 65535,
        "colorBlueMinimum": 0,
        "colorBlueMaximum": 65535,
    },
)

IDENTITY = (1.0, 0.0, 0.0, 0.0)


def cartesian_data(n=6, invalid=None, rgb=False, intensity=False, row_column=False):
    data = {
        "cartesianX": np.arange(n, dtype=np.float64),
        "cartesianY": np.zeros(n),
        "cartesianZ": np.zeros(n),
    }
    if rgb:
        data["colorRed"] = np.arange(n, dtype=np.uint8)
        data["colorGreen"] = np.arange(n, dtype=np.uint8) + 100
        data["colorBlue"] = np.arange(n, dtype=np.uint8) + 200
    if intensity:
        data["intensity"] = np.arange(n, dtype=np.float64) / 10.0
    if row_column:
        data["rowIndex"] = np.arange(n, dtype=np.int32)
        data["columnIndex"] = np.arange(n, dtype=np.int32)[::-1].copy()
    if invalid is not None:
        data["cartesianInvalidState"] = np.asarray(invalid, dtype=np.int8)
    return data


def spherical_data(n=4):
    return {
        "sphericalRange": np.full(n, 2.0),
        "sphericalAzimuth": np.linspace(0.0, np.pi / 2, n),
        "sphericalElevation": np.zeros(n),
    }


def _scan(pose=pose_node(IDENTITY, (0.0, 0.0, 0.0)), guid=GUID_A, data=None, extra=None, **kw):
    return make_scan(guid=guid, pose=pose, point_data=data or cartesian_data(), extra=extra, **kw)


# ---------------------------------------------------------------- the mask (§19)


def test_invalid_state_mask_is_applied_to_every_column_identically(fake_e57):
    """The whole point of Phase 0B.3: colour must follow the coordinates, not the index.

    Invalid points sit at positions 1 and 3, so the surviving points are 0, 2, 4, 5. The
    defect this replaces kept ``rgb[:4]`` — points 0, 1, 2, 3 — which is off by one from the
    second invalid point onwards and looks perfectly normal in a viewer.
    """
    data = cartesian_data(6, invalid=[0, 1, 0, 2, 0, 0], rgb=True, intensity=True, row_column=True)
    path, _ = fake_e57([_scan(data=data, extra={"colorLimits": COLOR_LIMITS_255})])

    from minegs.ingest.e57 import _nodes

    with _nodes.open_e57(path) as handle:
        cloud, meta = read_scan_points(handle, 0)

    assert meta["invalid_state_field"] == "cartesianInvalidState"
    assert meta["invalid_points_removed"] == 2
    kept = [0, 2, 4, 5]
    assert cloud.xyz[:, 0].tolist() == [float(i) for i in kept]
    assert cloud.rgb[:, 0].tolist() == kept, "red must follow the surviving points"
    assert cloud.rgb[:, 1].tolist() == [i + 100 for i in kept]
    assert cloud.rgb[:, 2].tolist() == [i + 200 for i in kept]
    assert cloud.extra["intensity"].tolist() == [i / 10.0 for i in kept]
    assert cloud.extra["rowIndex"].tolist() == kept
    assert cloud.extra["columnIndex"].tolist() == [5 - i for i in kept]


def test_invalid_state_one_and_two_are_both_dropped():
    """E57 state 1 means the direction is valid but the range is not — no usable position."""
    columns = {
        "cartesianX": np.arange(5, dtype=np.float64),
        "cartesianInvalidState": np.array([0, 1, 2, 0, 0], dtype=np.int8),
    }
    out, field, removed = apply_invalid_state(columns)
    assert field == "cartesianInvalidState" and removed == 2
    assert out["cartesianX"].tolist() == [0.0, 3.0, 4.0]


def test_spherical_invalid_state_is_honoured():
    columns = {
        "sphericalRange": np.arange(4, dtype=np.float64),
        "sphericalInvalidState": np.array([0, 0, 1, 0], dtype=np.int8),
    }
    out, field, removed = apply_invalid_state(columns)
    assert field == "sphericalInvalidState" and removed == 1
    assert out["sphericalRange"].tolist() == [0.0, 1.0, 3.0]


def test_a_scan_with_no_invalid_state_keeps_every_point():
    columns = {"cartesianX": np.arange(3, dtype=np.float64)}
    out, field, removed = apply_invalid_state(columns)
    assert field is None and removed == 0 and out["cartesianX"].tolist() == [0.0, 1.0, 2.0]


def test_mismatched_column_lengths_are_rejected(fake_e57):
    """A scan is a fixed-length record array; parallel columns of different lengths cannot be
    masked safely, and picking one to truncate is the bug, not the fix."""
    with pytest.raises(ContractError, match="different lengths"):
        apply_invalid_state({"cartesianX": np.zeros(5), "colorRed": np.zeros(4, dtype=np.uint8)})

    data = cartesian_data(4, rgb=True)
    data["colorRed"] = np.zeros(3, dtype=np.uint8)
    path, _ = fake_e57([_scan(data=data, extra={"colorLimits": COLOR_LIMITS_255})])
    from minegs.ingest.e57 import _nodes

    with _nodes.open_e57(path) as handle, pytest.raises(ContractError, match="different lengths"):
        read_scan_points(handle, 0)


# ---------------------------------------------------------------- colour (§20)


def test_declared_8bit_colour_is_passed_through_unchanged(fake_e57, tmp_path):
    path, _ = fake_e57(
        [_scan(data=cartesian_data(3, rgb=True), extra={"colorLimits": COLOR_LIMITS_255})]
    )
    m = extract(path, tmp_path / "out", compute_hash=False)
    s = m.scan_output("scan_000")
    assert s.color_source_range == [0.0, 255.0]
    assert s.color_conversion == "none (already 0..255)"
    assert read_ply(s.path).rgb[:, 0].tolist() == [0, 1, 2]


def test_declared_16bit_colour_is_converted_and_the_conversion_recorded(fake_e57, tmp_path):
    data = cartesian_data(3)
    data["colorRed"] = np.array([0, 32768, 65535], dtype=np.uint16)
    data["colorGreen"] = np.array([0, 0, 0], dtype=np.uint16)
    data["colorBlue"] = np.array([65535, 65535, 65535], dtype=np.uint16)
    path, _ = fake_e57([_scan(data=data, extra={"colorLimits": COLOR_LIMITS_16BIT})])
    m = extract(path, tmp_path / "out", compute_hash=False)
    s = m.scan_output("scan_000")
    assert s.color_source_range == [0.0, 65535.0]
    assert s.color_conversion == "linear 0.0..65535.0 -> 0..255"
    rgb = read_ply(s.path).rgb
    assert rgb[:, 0].tolist() == [0, 128, 255] and rgb[:, 2].tolist() == [255, 255, 255]


def test_unknown_colour_range_is_preserved_raw_not_guessed(fake_e57, tmp_path):
    """No colorLimits: 10 could be a dark grey or almost black. Neither is assumed."""
    path, _ = fake_e57([_scan(data=cartesian_data(3, rgb=True))])
    m = extract(path, tmp_path / "out", compute_hash=False)
    s = m.scan_output("scan_000")
    assert s.color_source_range is None
    assert "unknown" in (s.color_conversion or "")
    assert any("colour range unknown" in i or "unknown" in i for i in s.issues)
    assert "color_red" in s.attributes and "red" not in s.attributes
    cloud = read_ply(s.path)
    assert cloud.rgb is None
    assert cloud.extra["color_red"].tolist() == [0, 1, 2]


# ---------------------------------------------------------------- coordinates (§18)


def test_cartesian_scan_extraction(fake_e57, tmp_path):
    path, _ = fake_e57([_scan(data=cartesian_data(5))])
    m = extract(path, tmp_path / "out", compute_hash=False)
    s = m.scan_output("scan_000")
    assert s.point_count_masked == s.point_count_output == 5
    assert "cartesian coordinates" in s.notes
    assert read_ply(s.path).xyz[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_spherical_only_extraction(fake_e57, tmp_path):
    """range/azimuth/elevation becomes scanner Cartesian; it is still the SOURCE chain."""
    path, _ = fake_e57([_scan(data=spherical_data(4))])
    m = extract(path, tmp_path / "out", compute_hash=False)
    s = m.scan_output("scan_000")
    assert "spherical coordinates" in s.notes and s.point_count_output == 4
    xyz = read_ply(s.path).xyz
    assert np.allclose(np.linalg.norm(xyz, axis=1), 2.0)
    assert s.source_frame == "SOURCE"


def test_a_scan_with_no_usable_coordinates_is_refused(fake_e57, tmp_path):
    path, _ = fake_e57([_scan(data={"cartesianX": np.zeros(3), "cartesianY": np.zeros(3)})])
    with pytest.raises(ContractError, match="no usable coordinate fields"):
        extract(path, tmp_path / "out", compute_hash=False)
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------- poses (§17)


def test_valid_pose_is_applied_and_preserved(fake_e57, tmp_path):
    """90 deg about z, then +10 in x: the transform must be the one the file declared."""
    c = np.sqrt(0.5)
    path, _ = fake_e57(
        [_scan(pose=pose_node((c, 0.0, 0.0, c), (10.0, 0.0, 0.0)), data=cartesian_data(3))]
    )
    m = extract(path, tmp_path / "out", compute_hash=False)
    s = m.scan_output("scan_000")
    assert s.pose_status == "valid" and s.registration_status == "registered"
    assert s.source_frame == "SOURCE"
    # scanner (i, 0, 0) -> source (10, i, 0)
    assert np.allclose(read_ply(s.path).xyz, [[10, 0, 0], [10, 1, 0], [10, 2, 0]], atol=1e-9)
    assert s.pose is not None and s.pose.translation_m == [10.0, 0.0, 0.0]

    meta = json.loads((tmp_path / "out" / "scans" / "scan_000.pose.json").read_text())
    assert meta["point_frame"] == "SOURCE" and meta["registration_status"] == "registered"
    assert [row[3] for row in meta["T_source_from_scanner"][:3]] == [10.0, 0.0, 0.0]


def test_raw_extraction_keeps_the_scanner_frame(fake_e57, tmp_path):
    c = np.sqrt(0.5)
    path, _ = fake_e57(
        [_scan(pose=pose_node((c, 0.0, 0.0, c), (10.0, 0.0, 0.0)), data=cartesian_data(3))]
    )
    m = extract(path, tmp_path / "out", registered=False, compute_hash=False)
    s = m.scan_output("scan_000")
    assert s.source_frame == "SCANNER" and s.registration_status == "unregistered"
    assert m.output_frame == "SCANNER" and m.registration == "unregistered"
    assert read_ply(s.path).xyz[:, 0].tolist() == [0.0, 1.0, 2.0], "the pose was not applied"
    assert read_ply(s.path).frame == "SCANNER"
    meta = json.loads((tmp_path / "out" / "scans" / "scan_000.pose.json").read_text())
    assert meta["point_frame"] == "SCANNER" and meta["registration_status"] == "unregistered"
    assert meta["T_source_from_scanner"] is not None, "the pose is still reported, just not used"


def test_invalid_pose_is_refused(fake_e57, tmp_path):
    path, _ = fake_e57(
        [_scan(pose=pose_node((2.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), data=cartesian_data(3))]
    )
    with pytest.raises(E57PoseUnusableError, match="cannot be used"):
        extract(path, tmp_path / "out", compute_hash=False)
    assert not (tmp_path / "out").exists()


def test_unreadable_pose_is_refused(fake_e57, tmp_path):
    path, _ = fake_e57([_scan(pose=ExplodingNode("pose"), data=cartesian_data(3))])
    with pytest.raises(E57PoseUnusableError, match="unreadable"):
        extract(path, tmp_path / "out", compute_hash=False)
    assert not (tmp_path / "out").exists()


def test_a_broken_pose_is_refused_even_for_raw_extraction(fake_e57, tmp_path):
    """§17: unreadable and invalid poses stop extraction on either path."""
    path, _ = fake_e57([_scan(pose=ExplodingNode("pose"), data=cartesian_data(3))])
    with pytest.raises(E57PoseUnusableError):
        extract(path, tmp_path / "out", registered=False, compute_hash=False)


def test_absent_pose_matches_the_chosen_contract(fake_e57, tmp_path):
    """The decision this PR makes explicit (§17): registered needs a pose, raw does not.

    An unregistered cloud and a registered one are indistinguishable by looking at them, so
    the raw path marks its output and never pretends the missing pose was identity.
    """
    path, _ = fake_e57([_scan(pose=None, data=cartesian_data(3))])

    with pytest.raises(E57PoseUnusableError, match="declare no pose") as err:
        extract(path, tmp_path / "registered", compute_hash=False)
    assert "--raw" in str(err.value)
    assert not (tmp_path / "registered").exists()

    m = extract(path, tmp_path / "raw", registered=False, compute_hash=False)
    s = m.scan_output("scan_000")
    assert s.pose_status == "absent" and s.pose is None
    assert s.registration_status == "unregistered" and s.source_frame == "SCANNER"
    meta = json.loads((tmp_path / "raw" / "scans" / "scan_000.pose.json").read_text())
    assert meta["T_source_from_scanner"] is None, "no identity may be invented"


def test_identity_pose_registers_and_says_so(fake_e57, tmp_path):
    path, _ = fake_e57([_scan(pose=pose_node(IDENTITY, (0.0, 0.0, 0.0)), data=cartesian_data(3))])
    s = extract(path, tmp_path / "out", compute_hash=False).scan_output("scan_000")
    assert s.pose_status == "identity" and s.registration_status == "registered"


def test_extraction_never_uses_pye57s_transforming_reader():
    """read_scan(transform=True) builds its transform from pye57's falling-back properties."""
    import inspect

    import minegs.ingest.e57.extract as extract_mod

    # The module docstring names the trap on purpose; the code below it must not use it.
    code = inspect.getsource(extract_mod).split('"""', 2)[2]
    assert "read_scan_raw" in code
    assert "rotation_matrix" not in code and ".translation" not in code
    assert ".read_scan(" not in code


# ---------------------------------------------------------------- selection and preflight


def test_only_the_selected_scans_are_extracted(fake_e57, tmp_path):
    path, _ = fake_e57([_scan(data=cartesian_data(3)) for _ in range(3)])
    m = extract(path, tmp_path / "out", scan_ids=["scan_001"], compute_hash=False)
    assert [s.scan_id for s in m.scan_outputs] == ["scan_001"]
    assert sorted(p.name for p in (tmp_path / "out" / "scans").iterdir()) == [
        "scan_001.ply",
        "scan_001.pose.json",
    ]


def test_an_unknown_scan_id_is_refused(fake_e57, tmp_path):
    path, _ = fake_e57([_scan(data=cartesian_data(3))])
    with pytest.raises(ContractError, match="no such scan"):
        extract(path, tmp_path / "out", scan_ids=["scan_007"], compute_hash=False)


def test_nothing_is_written_when_any_selected_scan_fails_preflight(fake_e57, tmp_path):
    """A run that writes two scans and then refuses the third leaves a plausible-looking
    staging directory that is missing a scan nobody will notice."""
    path, _ = fake_e57(
        [
            _scan(data=cartesian_data(3)),
            _scan(data=cartesian_data(3)),
            _scan(pose=None, data=cartesian_data(3)),
        ]
    )
    out = tmp_path / "out"
    with pytest.raises(E57PoseUnusableError):
        extract(path, out, compute_hash=False)
    assert not out.exists(), "no partial staging directory may survive a refusal"


def test_existing_output_is_not_silently_mixed(fake_e57, tmp_path):
    path, _ = fake_e57([_scan(data=cartesian_data(3))])
    out = tmp_path / "out"
    extract(path, out, compute_hash=False)
    with pytest.raises(ContractError, match="already holds extraction output"):
        extract(path, out, compute_hash=False)
    extract(path, out, compute_hash=False, overwrite=True)


def test_overwrite_refuses_to_delete_files_it_did_not_write(fake_e57, tmp_path):
    """--overwrite means "replace my last extraction", never "empty this directory"."""
    path, _ = fake_e57([_scan(data=cartesian_data(3))])
    out = tmp_path / "out"
    extract(path, out, compute_hash=False)
    keepsake = out / "scans" / "field_notes.txt"
    keepsake.write_text("someone's data")
    with pytest.raises(ContractError, match="did not write"):
        extract(path, out, overwrite=True, compute_hash=False)
    assert keepsake.exists()


def test_overwrite_replaces_rather_than_merges(fake_e57, tmp_path):
    """A stale scan from an earlier run, in the other frame, must not survive --overwrite."""
    path, _ = fake_e57([_scan(data=cartesian_data(3)) for _ in range(3)])
    out = tmp_path / "out"
    extract(path, out, compute_hash=False)
    assert len(list((out / "scans").glob("*.ply"))) == 3

    m = extract(
        path, out, scan_ids=["scan_001"], registered=False, overwrite=True, compute_hash=False
    )
    assert [p.name for p in (out / "scans").glob("*.ply")] == ["scan_001.ply"]
    assert [s.scan_id for s in m.scan_outputs] == ["scan_001"]


def test_a_scan_too_large_for_memory_can_be_refused(fake_e57, tmp_path):
    path, _ = fake_e57([make_scan(point_count=10_000_000, pose=pose_node(IDENTITY, (0, 0, 0)))])
    with pytest.raises(ContractError, match="max-scan-points"):
        extract(path, tmp_path / "out", max_scan_points=1_000, compute_hash=False)
    assert not (tmp_path / "out").exists()


def test_unreadable_point_payload_is_an_explaining_error(fake_e57, tmp_path):
    path, _ = fake_e57([_scan(data=cartesian_data(3))])
    from minegs.ingest.e57 import _nodes

    with _nodes.open_e57(path) as handle:
        handle._headers[0].point_data = None
        with pytest.raises(AssertionError):
            read_scan_points(handle, 0)

    path2, _ = fake_e57([_scan(data=cartesian_data(3))], name="b.e57")
    with _nodes.open_e57(path2) as handle:

        def boom(*a, **k):
            raise RuntimeError("libe57 says no")

        handle.read_scan_raw = boom
        with pytest.raises(E57ReadScanError, match="libe57 says no"):
            read_scan_points(handle, 0)


# ---------------------------------------------------------------- voxel downsampling


def test_voxel_downsample_is_deterministic(fake_e57, tmp_path):
    data = cartesian_data(6)
    data["cartesianX"] = np.array([0.0, 0.001, 0.002, 1.0, 1.001, 2.0])
    path, _ = fake_e57([_scan(data=data)])
    a = extract(path, tmp_path / "a", voxel_m=0.5, compute_hash=False).scan_output("scan_000")
    b = extract(path, tmp_path / "b", voxel_m=0.5, compute_hash=False).scan_output("scan_000")
    assert a.point_count_masked == 6 and a.point_count_output == 3
    assert a.voxel_m == 0.5 and a.sha256 == b.sha256, "same input, byte-identical output"
    assert read_ply(a.path).xyz[:, 0].tolist() == [0.0, 1.0, 2.0]


def test_no_voxel_keeps_every_masked_point(fake_e57, tmp_path):
    path, _ = fake_e57([_scan(data=cartesian_data(6, invalid=[0, 0, 1, 0, 0, 0]))])
    s = extract(path, tmp_path / "out", compute_hash=False).scan_output("scan_000")
    assert s.voxel_m is None and s.point_count_masked == s.point_count_output == 5


# ---------------------------------------------------------------- images (§23, §24)


def test_supported_embedded_images_are_written(fake_e57, tmp_path):
    payload = b"\xff\xd8\xff\xdb" + b"JPEGBODY"
    path, _ = fake_e57(
        [_scan(data=cartesian_data(3))],
        root=root_with_images(
            image_node(associated_scan_guid=GUID_A, name="pano_a", payload=payload)
        ),
    )
    m = extract(path, tmp_path / "out", compute_hash=False)
    (im,) = m.image_outputs
    assert im.image_id == "image_000" and im.representation == "spherical"
    assert im.image_format == "jpeg" and im.blob_field == "jpegImage"
    assert im.bytes_written == len(payload) and im.sha256
    written = tmp_path / "out" / "images" / "image_000.jpg"
    assert written.read_bytes() == payload and im.path == str(written)
    assert (im.width, im.height) == (8, 4)


def test_cylindrical_images_are_supported(fake_e57, tmp_path):
    path, _ = fake_e57(
        [_scan(data=cartesian_data(3))],
        root=root_with_images(
            image_node(
                representation="cylindricalRepresentation",
                blob_field="pngImage",
                payload=b"\x89PNG\r\n\x1a\nBODY",
                associated_scan_guid=GUID_A,
            )
        ),
    )
    (im,) = extract(path, tmp_path / "out", compute_hash=False).image_outputs
    assert im.representation == "cylindrical" and im.image_format == "png"
    assert (tmp_path / "out" / "images" / "image_000.png").exists()


def test_unsupported_representations_are_recorded_not_reinterpreted(fake_e57, tmp_path):
    path, _ = fake_e57(
        [_scan(data=cartesian_data(3))],
        root=root_with_images(
            image_node(representation="pinholeRepresentation", associated_scan_guid=GUID_A),
            image_node(representation="cubeMapRepresentation", associated_scan_guid=GUID_A),
        ),
    )
    m = extract(path, tmp_path / "out", compute_hash=False)
    assert m.image_outputs == []
    assert [s.image_id for s in m.skipped_images] == ["image_000", "image_001"]
    pinhole, unknown = m.skipped_images
    assert pinhole.representation == "pinhole" and "unsupported" in pinhole.reason
    assert "no perspective handling" in pinhole.reason
    assert unknown.representation == "unknown"
    assert not (tmp_path / "out" / "images").exists(), "nothing was written for them"


def test_a_blob_whose_bytes_contradict_its_label_is_refused(fake_e57, tmp_path):
    """An entry labelled jpegImage holding PNG bytes is a broken file, not a format to guess."""
    path, _ = fake_e57(
        [_scan(data=cartesian_data(3))],
        root=root_with_images(image_node(payload=b"\x89PNG\r\n\x1a\nnot a jpeg")),
    )
    m = extract(path, tmp_path / "out", compute_hash=False)
    assert m.image_outputs == []
    (skipped,) = m.skipped_images
    assert "do not start with the jpeg signature" in skipped.reason
    assert any("jpeg signature" in i for i in m.issues)


def test_an_unreadable_blob_does_not_write_a_truncated_file(fake_e57, tmp_path):
    path, _ = fake_e57(
        [_scan(data=cartesian_data(3))],
        root=root_with_images(image_node(blob=ExplodingBlob("jpegImage", b"xx"))),
    )
    with pytest.raises(Exception, match="blob payload unreadable"):
        extract(path, tmp_path / "out", compute_hash=False)
    assert not (tmp_path / "out" / "images" / "image_000.jpg").exists()


def test_images_can_be_skipped_entirely(fake_e57, tmp_path):
    path, _ = fake_e57(
        [_scan(data=cartesian_data(3))],
        root=root_with_images(image_node(associated_scan_guid=GUID_A)),
    )
    m = extract(path, tmp_path / "out", with_images=False, compute_hash=False)
    assert m.image_outputs == [] and m.skipped_images == []
    assert any("not extracted" in n for n in m.notes)
    assert not (tmp_path / "out" / "images").exists()


def test_mapping_metadata_reaches_the_extracted_images(fake_e57, tmp_path):
    """§30.38: what the mapping decided must travel with the file it decided about."""
    path, _ = fake_e57(
        [_scan(guid=GUID_A, data=cartesian_data(3)), _scan(guid=GUID_B, data=cartesian_data(3))],
        root=root_with_images(
            image_node(associated_scan_guid=GUID_B, name="pano_b"),
            image_node(associated_scan_guid=None, name="pano_x"),
        ),
    )
    m = extract(path, tmp_path / "out", compute_hash=False)
    mapped, unmapped = m.image_outputs
    assert mapped.mapping_status == "confirmed"
    assert mapped.mapped_scan_id == "scan_001" and mapped.mapped_station_id == "S001"
    assert mapped.mapping_evidence_type == "e57_associated_guid"
    assert unmapped.mapping_status == "unmapped"
    assert unmapped.mapped_scan_id is None and unmapped.mapped_station_id is None
    assert m.mapping_report is not None
    assert m.mapping_report.record_for("image_000").scan_id == "scan_001"


def test_manual_mapping_metadata_reaches_the_images(fake_e57, tmp_path):
    mapping = tmp_path / "map.csv"
    mapping.write_text("scan_id,image_id\nscan_000,image_000\n")
    path, _ = fake_e57(
        [_scan(data=cartesian_data(3))],
        root=root_with_images(image_node(associated_scan_guid=None)),
    )
    (im,) = extract(path, tmp_path / "out", mapping=mapping, compute_hash=False).image_outputs
    assert im.mapping_status == "manual" and im.mapped_scan_id == "scan_000"
    assert im.mapping_evidence_type == "explicit_mapping"


def test_external_images_are_referenced_not_copied(fake_e57, tmp_path):
    from PIL import Image

    d = tmp_path / "images"
    d.mkdir()
    Image.new("RGB", (8, 4)).save(d / "a.png")
    path, _ = fake_e57([_scan(data=cartesian_data(3))])
    (im,) = extract(path, tmp_path / "out", images_dir=d, compute_hash=False).image_outputs
    assert im.source == "external_file" and im.extracted is False
    assert im.path == str((d / "a.png").resolve()) and im.sha256
    assert not (tmp_path / "out" / "images").exists()


# ---------------------------------------------------------------- the manifest (§26)


def test_extraction_manifest_round_trips_and_records_hashes(fake_e57, tmp_path):
    path, _ = fake_e57(
        [_scan(data=cartesian_data(3))],
        root=root_with_images(image_node(associated_scan_guid=GUID_A)),
    )
    out = tmp_path / "out"
    m = extract(path, out, compute_hash=True)

    reloaded = E57ExtractionManifest.load(out / "extraction_manifest.json")
    assert reloaded.schema_version == "1.0"
    assert reloaded.model_dump() == m.model_dump()

    assert m.source_sha256 and len(m.source_sha256) == 64
    assert m.scan_outputs[0].sha256 and len(m.scan_outputs[0].sha256) == 64
    assert m.image_outputs[0].sha256 and len(m.image_outputs[0].sha256) == 64
    assert m.provenance.source_assets[0].sha256 == m.source_sha256
    assert "python" in m.provenance.tool_versions

    raw = json.loads((out / "extraction_manifest.json").read_text())
    assert set(raw) >= {
        "schema_version",
        "source_e57",
        "source_sha256",
        "scan_outputs",
        "image_outputs",
        "mapping_report",
        "issues",
        "notes",
        "provenance",
    }


def test_staging_layout_is_not_a_dataset(fake_e57, tmp_path):
    """A reader must never mistake Phase 0B staging for the Phase 0C dataset contract."""
    path, _ = fake_e57(
        [_scan(data=cartesian_data(3))],
        root=root_with_images(image_node(associated_scan_guid=GUID_A)),
    )
    out = tmp_path / "out"
    extract(path, out, compute_hash=False)
    assert sorted(p.name for p in out.iterdir()) == [
        "extraction_manifest.json",
        "images",
        "inventory.json",
        "pano_mapping.json",
        "scans",
    ]
    for forbidden in ("manifest.json", "sparse", "init_points.ply", "masks"):
        assert not (out / forbidden).exists()


def test_output_filenames_are_deterministic(fake_e57, tmp_path):
    path, _ = fake_e57([_scan(data=cartesian_data(3)) for _ in range(3)])
    extract(path, tmp_path / "out", compute_hash=False)
    assert sorted(p.name for p in (tmp_path / "out" / "scans").glob("*.ply")) == [
        "scan_000.ply",
        "scan_001.ply",
        "scan_002.ply",
    ]


def test_manifest_never_claims_a_tls_global_frame(fake_e57, tmp_path):
    path, _ = fake_e57([_scan(data=cartesian_data(3))])
    m = extract(path, tmp_path / "out", compute_hash=False)
    dumped = m.model_dump_json()
    assert "TLS_GLOBAL" not in dumped and "LOCAL_METRIC" not in dumped, (
        "not even a disclaimer may name a later frame, so grepping an artifact stays a "
        "true positive"
    )
    assert '"output_frame":"SOURCE"' in dumped
    ply_header = (tmp_path / "out" / "scans" / "scan_000.ply").read_bytes()[:200]
    assert b"frame=SOURCE" in ply_header and b"TLS_GLOBAL" not in ply_header


def test_dropped_attributes_are_named_not_silently_lost(fake_e57, tmp_path):
    data = cartesian_data(3)
    data["timeStamp"] = np.arange(3, dtype=np.float64)
    data["returnIndex"] = np.zeros(3, dtype=np.int8)
    path, _ = fake_e57([_scan(data=data)])
    s = extract(path, tmp_path / "out", compute_hash=False).scan_output("scan_000")
    assert set(s.dropped_attributes) == {"timeStamp", "returnIndex"}
    assert all("re-read it from the source E57" in why for why in s.dropped_attributes.values())


# ---------------------------------------------------------------- CLI


def test_extract_cli_prints_the_frame_and_the_mask(fake_e57, tmp_path, capsys):
    from minegs.cli.ingest import _print_extraction

    path, _ = fake_e57(
        [
            _scan(
                data=cartesian_data(6, invalid=[0, 1, 0, 0, 0, 0], rgb=True),
                extra={"colorLimits": COLOR_LIMITS_255},
            )
        ]
    )
    _print_extraction(extract(path, tmp_path / "out", compute_hash=False))
    out = capsys.readouterr().out
    assert "SOURCE" in out and "registered" in out
    assert "1 invalid removed" in out
    assert "not a dataset" in out


# ---------------------------------------------------------------- real libE57 (§31)

pye57 = pytest.importorskip("pye57", reason="real-E57 extraction needs minegs[e57]")


def _write_real_e57(path, scans):
    f = pye57.E57(str(path), mode="w")
    try:
        for spec in scans:
            n = spec.get("n", 5)
            data = {
                "cartesianX": np.arange(n, dtype=np.float64),
                "cartesianY": np.zeros(n),
                "cartesianZ": np.zeros(n),
            }
            if spec.get("rgb"):
                data.update(
                    colorRed=np.arange(n, dtype=np.float64),
                    colorGreen=np.arange(n, dtype=np.float64) + 100,
                    colorBlue=np.arange(n, dtype=np.float64) + 200,
                )
            if spec.get("intensity"):
                data["intensity"] = np.arange(n, dtype=np.float64) / 10.0
            if spec.get("invalid") is not None:
                data["cartesianInvalidState"] = np.asarray(spec["invalid"], dtype=np.int32)
            f.write_scan_raw(
                data,
                name=spec.get("name", "Scan"),
                rotation=np.asarray(spec.get("rotation", [1.0, 0.0, 0.0, 0.0])),
                translation=np.asarray(spec.get("translation", [0.0, 0.0, 0.0])),
            )
    finally:
        f.close()
    return path


def test_real_e57_extraction_round_trip(tmp_path):
    """Genuine libE57 output, written at test time: no binary fixture in the repository."""
    path = _write_real_e57(
        tmp_path / "real.e57",
        [
            {"name": "A", "n": 6, "rgb": True, "intensity": True, "translation": [4.0, 5.0, 6.0]},
            {"name": "B", "n": 3},
        ],
    )
    out = tmp_path / "out"
    m = extract(path, out, compute_hash=True)

    assert [s.scan_id for s in m.scan_outputs] == ["scan_000", "scan_001"]
    a = m.scan_output("scan_000")
    assert a.registration_status == "registered" and a.source_frame == "SOURCE"
    assert a.color_source_range == [0.0, 255.0], "pye57 writes colorLimits"
    assert a.point_count_input == 6 and a.point_count_output == 6
    cloud = read_ply(a.path)
    assert cloud.frame == "SOURCE"
    # scanner (i, 0, 0) with an identity rotation and t = (4, 5, 6)
    assert np.allclose(cloud.xyz[:, 0], np.arange(6) + 4.0)
    assert np.allclose(cloud.xyz[:, 1], 5.0) and np.allclose(cloud.xyz[:, 2], 6.0)
    assert cloud.rgb[:, 0].tolist() == list(range(6))
    assert np.allclose(cloud.extra["intensity"], np.arange(6) / 10.0)
    assert a.sha256 == m.scan_output("scan_000").sha256

    pose = json.loads((out / "scans" / "scan_000.pose.json").read_text())
    assert pose["source_frame"] == "SOURCE" and "T_tls_from_scanner" not in pose
    assert [row[3] for row in pose["T_source_from_scanner"][:3]] == [4.0, 5.0, 6.0]

    reloaded = E57ExtractionManifest.load(out / "extraction_manifest.json")
    assert reloaded.source_sha256 == m.source_sha256


def test_real_e57_invalid_state_keeps_colour_aligned(tmp_path):
    """The same defect, on a genuine file libE57 wrote: colour must not shift."""
    path = _write_real_e57(
        tmp_path / "masked.e57",
        [{"name": "A", "n": 6, "rgb": True, "intensity": True, "invalid": [0, 1, 0, 2, 0, 0]}],
    )
    s = extract(path, tmp_path / "out", compute_hash=False).scan_output("scan_000")
    assert s.invalid_state_field == "cartesianInvalidState"
    assert s.invalid_points_removed == 2 and s.point_count_output == 4
    cloud = read_ply(s.path)
    kept = [0, 2, 4, 5]
    assert cloud.xyz[:, 0].tolist() == [float(i) for i in kept]
    assert cloud.rgb[:, 0].tolist() == kept
    assert cloud.rgb[:, 1].tolist() == [i + 100 for i in kept]
    assert np.allclose(cloud.extra["intensity"], [i / 10.0 for i in kept])


def test_real_e57_raw_extraction_does_not_apply_the_pose(tmp_path):
    path = _write_real_e57(
        tmp_path / "raw.e57", [{"name": "A", "n": 4, "translation": [100.0, 0.0, 0.0]}]
    )
    s = extract(path, tmp_path / "out", registered=False, compute_hash=False).scan_output(
        "scan_000"
    )
    assert s.source_frame == "SCANNER"
    assert read_ply(s.path).xyz[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_real_e57_extraction_reads_points_while_inventory_does_not(tmp_path, monkeypatch):
    """The two phases differ exactly here: 0B.1 must not read the payload, 0B.3 must."""
    from minegs.ingest.e57.inventory import inventory

    path = _write_real_e57(tmp_path / "guard.e57", [{"name": "A", "n": 4}])
    calls: list[int] = []
    original = pye57.E57.read_scan_raw
    monkeypatch.setattr(
        pye57.E57,
        "read_scan_raw",
        lambda self, i, *a, **k: (calls.append(i), original(self, i, *a, **k))[1],
    )
    inventory(path, compute_hash=False)
    assert calls == [], "the inventory must never read point data"
    extract(path, tmp_path / "out", compute_hash=False)
    assert calls == [0], "extraction must read it exactly once per scan"
