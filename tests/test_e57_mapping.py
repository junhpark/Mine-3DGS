"""Phase 0B.2 contract tests: image discovery, and station mapping from evidence only.

The tests that matter most here are the negative ones. Mapping a panorama to the wrong
station does not crash, does not warn, and produces a reconstruction that trains and
converges — it is simply wrong. So each of the inferences a plausible implementation would
reach for (index equality, equal counts, file order, name similarity) has a test asserting it
produces *no* mapping.
"""

from __future__ import annotations

import json

import pytest
from minegs.core.errors import ContractError
from minegs.ingest.e57.images import (
    ImageAsset,
    classify_representation,
    discover_embedded_images,
    discover_external_images,
    image_id_for,
)
from minegs.ingest.e57.inventory import inventory, list_images2d
from minegs.ingest.e57.mapping import (
    PanoMappingReport,
    build_mapping_report,
    read_mapping_file,
    read_vendor_manifest,
)

from e57_fakes import ExplodingNode, FakeNode, image_node, make_scan, root_with_images

GUID_A = "{11111111-1111-1111-1111-111111111111}"
GUID_B = "{22222222-2222-2222-2222-222222222222}"
GUID_C = "{33333333-3333-3333-3333-333333333333}"


def _scans(*guids: str | None):
    return [make_scan(name=f"Setup {i}", guid=g) for i, g in enumerate(guids)]


def _report(fake_e57, scans, images, **kw) -> PanoMappingReport:
    path, _ = fake_e57(scans, root=root_with_images(*images))
    return build_mapping_report(path, compute_hash=False, **kw)


# ---------------------------------------------------------------- image discovery


def test_image_ids_are_deterministic_and_independent_of_vendor_metadata(fake_e57):
    assert [image_id_for(i) for i in (0, 1, 23)] == ["image_000", "image_001", "image_023"]
    with pytest.raises(ValueError):
        image_id_for(-1)

    path, _ = fake_e57(
        _scans(GUID_A),
        root=root_with_images(image_node(guid=None, name=None), image_node(guid=None, name=None)),
    )
    assets = discover_embedded_images(path)
    assert [a.image_id for a in assets] == ["image_000", "image_001"]
    assert [a.source_index for a in assets] == [0, 1]
    assert all(a.guid is None and a.name is None for a in assets)


def test_embedded_discovery_records_what_the_entry_declares(fake_e57):
    path, _ = fake_e57(
        _scans(GUID_A),
        root=root_with_images(
            image_node(
                guid="{img-1}",
                name="pano_a",
                associated_scan_guid=GUID_A,
                width=4096,
                height=2048,
                payload=b"x" * 1234,
                rep_extra={"pixelWidth": 0.001},
            )
        ),
    )
    (a,) = discover_embedded_images(path)
    assert a.source == "e57_embedded" and a.guid == "{img-1}" and a.name == "pano_a"
    assert a.associated_scan_guid == GUID_A
    assert (a.representation, a.representation_source) == ("spherical", "sphericalRepresentation")
    assert (a.width, a.height) == (4096, 2048)
    assert a.blob_field == "jpegImage" and a.blob_bytes == 1234
    assert a.vendor_metadata == {"pixelWidth": 0.001}
    assert a.is_extractable and not a.issues


def test_legacy_images2d_listing_is_the_same_reader(fake_e57):
    """``list_images2d`` is the Phase 0A shape of the same discovery, not a second one."""
    path, _ = fake_e57(
        _scans(GUID_A),
        root=root_with_images(image_node(name="pano_a", associated_scan_guid=GUID_A)),
    )
    assert list_images2d(path) == [
        {
            "index": 0,
            "guid": None,
            "name": "pano_a",
            "associated_scan_guid": GUID_A,
            "representation": "spherical",
            "width": 8,
            "height": 4,
        }
    ]


def test_unenumerable_images2d_is_an_error_not_an_empty_list(fake_e57):
    from minegs.ingest.e57.exceptions import E57UnsupportedStructureError

    path, _ = fake_e57(
        _scans(GUID_A), root=FakeNode("root", {"images2D": ExplodingNode("images2D")})
    )
    with pytest.raises(E57UnsupportedStructureError, match="could not be enumerated"):
        discover_embedded_images(path)


def test_absent_images2d_discovers_nothing_without_erroring(fake_e57):
    path, _ = fake_e57(_scans(GUID_A), root=FakeNode("root", {}))
    assert discover_embedded_images(path) == []


# -------------------------------------------------- representation classification (§8)


def test_spherical_is_a_panorama_candidate(fake_e57):
    (a,) = discover_embedded_images(
        _path(fake_e57, image_node(representation="sphericalRepresentation"))
    )
    assert a.representation == "spherical" and a.panorama_candidate


def test_cylindrical_is_a_panorama_candidate(fake_e57):
    (a,) = discover_embedded_images(
        _path(fake_e57, image_node(representation="cylindricalRepresentation"))
    )
    assert a.representation == "cylindrical" and a.panorama_candidate


def test_pinhole_is_not_automatically_a_panorama(fake_e57):
    """Six pinhole faces do tile a sphere. Recognising that takes evidence about the set."""
    (a,) = discover_embedded_images(
        _path(fake_e57, image_node(representation="pinholeRepresentation"))
    )
    assert a.representation == "pinhole" and not a.panorama_candidate
    assert a.is_extractable, "not a panorama is not the same as not extractable"


def test_visual_reference_is_not_a_panorama(fake_e57):
    (a,) = discover_embedded_images(
        _path(fake_e57, image_node(representation="visualReferenceRepresentation"))
    )
    assert a.representation == "visual_reference" and not a.panorama_candidate


def test_unknown_representation_makes_no_panorama_claim(fake_e57):
    """An entry we cannot interpret is recorded, never reinterpreted as something else."""
    (a,) = discover_embedded_images(
        _path(fake_e57, image_node(representation="cubeMapRepresentation"))
    )
    assert a.representation == "unknown" and a.representation_source is None
    assert not a.panorama_candidate and not a.is_extractable
    assert any("cannot be interpreted" in i for i in a.issues)


def test_classification_never_falls_back_to_another_projection():
    node = FakeNode("image2D", {"cylindricalRepresentation": FakeNode("cylindricalRepresentation")})
    assert classify_representation(node) == ("cylindrical", "cylindricalRepresentation")
    assert classify_representation(FakeNode("image2D", {})) == ("unknown", None)


def test_representation_without_pixels_is_reported(fake_e57):
    (a,) = discover_embedded_images(_path(fake_e57, image_node(blob_field=None)))
    assert a.representation == "spherical" and a.blob_field is None
    assert not a.is_extractable
    assert any("no jpegImage or pngImage blob" in i for i in a.issues)


def _path(fake_e57, *images):
    path, _ = fake_e57(_scans(GUID_A), root=root_with_images(*images))
    return path


# ---------------------------------------------------------------- Tier A: the E57 itself


def test_unique_associated_guid_is_confirmed(fake_e57):
    rep = _report(
        fake_e57,
        _scans(GUID_A, GUID_B),
        [image_node(associated_scan_guid=GUID_B, name="pano_b")],
    )
    m = rep.record_for("image_000")
    assert m.status == "confirmed" and m.evidence_type == "e57_associated_guid"
    assert m.scan_id == "scan_001" and m.station_id == "S001"
    assert m.evidence_value == GUID_B and m.is_resolved


def test_no_association_is_unmapped_never_guessed(fake_e57):
    rep = _report(fake_e57, _scans(GUID_A), [image_node(associated_scan_guid=None)])
    m = rep.record_for("image_000")
    assert m.status == "unmapped" and m.evidence_type == "none"
    assert m.scan_id is None and m.station_id is None and not m.is_resolved
    assert "nothing in the data says" in m.reason


def test_association_to_a_missing_scan_is_orphan(fake_e57):
    rep = _report(fake_e57, _scans(GUID_A), [image_node(associated_scan_guid=GUID_C)])
    m = rep.record_for("image_000")
    assert m.status == "orphan" and m.scan_id is None
    assert m.evidence_value == GUID_C and "does not contain" in m.reason


def test_duplicate_scan_guid_is_ambiguous_not_first_match(fake_e57):
    """Two scans share a GUID, so the association identifies neither (§12)."""
    rep = _report(
        fake_e57, _scans(GUID_A, GUID_A, GUID_B), [image_node(associated_scan_guid=GUID_A)]
    )
    m = rep.record_for("image_000")
    assert m.status == "ambiguous" and m.scan_id is None
    assert m.candidate_scan_ids == ["scan_000", "scan_001"]
    assert "first match is not a resolution" in m.reason
    assert any("declared by 2 scans" in i for i in rep.issues)


def test_duplicate_image_guid_is_reported(fake_e57):
    rep = _report(
        fake_e57,
        _scans(GUID_A),
        [image_node(guid="{dup}", associated_scan_guid=GUID_A) for _ in range(2)],
    )
    assert any("image GUID {dup} is declared by 2 images" in i for i in rep.issues)


def test_a_guid_differing_only_in_spelling_is_a_hint_not_a_mapping(fake_e57):
    """Case/brace/hyphen differences are a vendor quirk, not an identity this reader asserts."""
    rep = _report(
        fake_e57,
        _scans(GUID_A),
        [image_node(associated_scan_guid=GUID_A.strip("{}").upper())],
    )
    m = rep.record_for("image_000")
    assert m.status == "orphan" and m.scan_id is None
    assert any("differ only in case, braces or hyphens" in h for h in m.hints)


# ------------------------------------------------- forbidden inferences (§10)


def test_equal_image_and_scan_counts_do_not_create_a_mapping(fake_e57):
    rep = _report(
        fake_e57,
        _scans(GUID_A, GUID_B, GUID_C),
        [image_node(associated_scan_guid=None) for _ in range(3)],
    )
    assert len(rep.images) == rep.scan_count == 3
    assert [m.status for m in rep.mappings] == ["unmapped"] * 3
    assert all(m.scan_id is None for m in rep.mappings)
    assert any("Equal image and scan counts" in n for n in rep.notes)


def test_image_order_does_not_create_a_mapping(fake_e57):
    """image_000 sitting next to scan_000 is a coincidence of file layout, not evidence."""
    rep = _report(
        fake_e57,
        _scans(GUID_A, GUID_B),
        [image_node(associated_scan_guid=None), image_node(associated_scan_guid=GUID_B)],
    )
    first, second = rep.mappings
    assert first.status == "unmapped" and first.scan_id is None
    assert second.status == "confirmed" and second.scan_id == "scan_001"


def test_filename_similarity_does_not_create_a_mapping(fake_e57):
    """The image is literally called "Setup 0"; that still is not evidence."""
    scans = [make_scan(name="Setup 0", guid=GUID_A), make_scan(name="Setup 1", guid=GUID_B)]
    rep = _report(fake_e57, scans, [image_node(name="Setup 0", associated_scan_guid=None)])
    m = rep.record_for("image_000")
    assert m.status == "unmapped" and m.scan_id is None


# ---------------------------------------------------------------- Tier B: explicit mapping


def test_explicit_mapping_is_manual_not_confirmed(fake_e57, tmp_path):
    mapping = tmp_path / "mapping.csv"
    mapping.write_text("scan_id,image_id\nscan_001,image_000\n")
    rep = _report(
        fake_e57, _scans(GUID_A, GUID_B), [image_node(associated_scan_guid=None)], mapping=mapping
    )
    m = rep.record_for("image_000")
    assert m.status == "manual" and m.evidence_type == "explicit_mapping"
    assert m.scan_id == "scan_001" and m.station_id == "S001"
    assert "not machine-verifiable" in m.reason and m.is_resolved


def test_explicit_mapping_by_station_id(fake_e57, tmp_path):
    mapping = tmp_path / "mapping.csv"
    mapping.write_text("station_id,image_id\nS000,image_000\n")
    rep = _report(
        fake_e57, _scans(GUID_A, GUID_B), [image_node(associated_scan_guid=None)], mapping=mapping
    )
    m = rep.record_for("image_000")
    assert m.status == "manual" and m.scan_id == "scan_000" and m.station_id == "S000"


def test_explicit_mapping_agreeing_with_the_file_is_confirmed(fake_e57, tmp_path):
    mapping = tmp_path / "mapping.csv"
    mapping.write_text("scan_id,image_id\nscan_000,image_000\n")
    rep = _report(
        fake_e57, _scans(GUID_A), [image_node(associated_scan_guid=GUID_A)], mapping=mapping
    )
    m = rep.record_for("image_000")
    assert m.status == "confirmed" and m.evidence_type == "e57_associated_guid"
    assert "agree on scan_000" in m.reason


def test_manual_mapping_never_overwrites_the_files_own_evidence(fake_e57, tmp_path):
    """§13: the user says scan_001, the E57 says scan_000. Neither wins; the run is told."""
    mapping = tmp_path / "mapping.csv"
    mapping.write_text("scan_id,image_id\nscan_001,image_000\n")
    rep = _report(
        fake_e57,
        _scans(GUID_A, GUID_B),
        [image_node(associated_scan_guid=GUID_A)],
        mapping=mapping,
    )
    m = rep.record_for("image_000")
    assert m.status == "conflict" and m.scan_id is None and not m.is_resolved
    assert sorted(m.candidate_scan_ids) == ["scan_000", "scan_001"]
    assert "does not silently overwrite" in m.reason


def test_a_manual_mapping_cannot_override_an_orphan_e57_association(fake_e57, tmp_path):
    """The E57 names a GUID this file does not contain, and the user names a scan.

    Those are two different statements about the same image. The file's target being missing
    makes its statement unresolvable, not absent — and letting the mapping file win exactly
    when we understand the E57's claim least is the silent override this module exists to
    prevent. An explicit override policy would be a declared flag, not a side effect.
    """
    mapping = tmp_path / "mapping.csv"
    mapping.write_text("scan_id,image_id\nscan_000,image_000\n")
    rep = _report(
        fake_e57, _scans(GUID_A), [image_node(associated_scan_guid=GUID_C)], mapping=mapping
    )
    m = rep.record_for("image_000")
    assert m.status == "conflict" and m.scan_id is None and not m.is_resolved
    assert m.station_id is None
    assert GUID_C in (m.evidence_value or "")
    assert "which is not in this file" in m.reason
    assert m.candidate_scan_ids == ["scan_000"]


def test_a_vendor_manifest_cannot_override_an_orphan_e57_association(fake_e57, tmp_path):
    """Tier C is machine-generated, which makes it confirmed evidence — not a tiebreaker."""
    manifest = tmp_path / "vendor.json"
    manifest.write_text(
        json.dumps(
            {
                "vendor": "Acme",
                "generated_by": "exporter",
                "mappings": [{"image_id": "image_000", "scan_id": "scan_000"}],
            }
        )
    )
    rep = _report(
        fake_e57,
        _scans(GUID_A),
        [image_node(associated_scan_guid=GUID_C)],
        vendor_manifest=manifest,
    )
    assert rep.record_for("image_000").status == "conflict"


def test_an_orphan_association_alone_is_still_orphan_not_conflict(fake_e57):
    """Only a *disagreement* is a conflict; the E57 alone naming a missing scan is orphan."""
    rep = _report(fake_e57, _scans(GUID_A), [image_node(associated_scan_guid=GUID_C)])
    m = rep.record_for("image_000")
    assert m.status == "orphan" and m.scan_id is None


def test_a_mapping_file_only_overrides_nothing_when_the_e57_is_silent(fake_e57, tmp_path):
    """The permitted case: no embedded association at all, so nothing is being overridden."""
    mapping = tmp_path / "mapping.csv"
    mapping.write_text("scan_id,image_id\nscan_000,image_000\n")
    rep = _report(
        fake_e57, _scans(GUID_A), [image_node(associated_scan_guid=None)], mapping=mapping
    )
    m = rep.record_for("image_000")
    assert m.status == "manual" and m.scan_id == "scan_000"


def test_a_mapping_file_that_contradicts_itself_is_a_conflict(fake_e57, tmp_path):
    mapping = tmp_path / "mapping.csv"
    mapping.write_text("scan_id,image_id\nscan_000,image_000\nscan_001,image_000\n")
    rep = _report(
        fake_e57, _scans(GUID_A, GUID_B), [image_node(associated_scan_guid=None)], mapping=mapping
    )
    m = rep.record_for("image_000")
    assert m.status == "conflict" and m.scan_id is None
    assert "contradicts itself" in m.reason


def test_mapping_to_a_nonexistent_scan_is_orphan(fake_e57, tmp_path):
    mapping = tmp_path / "mapping.csv"
    mapping.write_text("scan_id,image_id\nscan_042,image_000\n")
    rep = _report(
        fake_e57, _scans(GUID_A), [image_node(associated_scan_guid=None)], mapping=mapping
    )
    m = rep.record_for("image_000")
    assert m.status == "orphan" and m.scan_id is None and "does not contain" in m.reason


def test_mapping_row_naming_a_nonexistent_image_is_kept_and_reported(fake_e57, tmp_path):
    """A typo in a mapping file must not be able to look like a clean run (§29.18)."""
    mapping = tmp_path / "mapping.csv"
    mapping.write_text("scan_id,image_id\nscan_000,image_000\nscan_000,image_099\n")
    rep = _report(
        fake_e57, _scans(GUID_A), [image_node(associated_scan_guid=None)], mapping=mapping
    )
    assert len(rep.mappings) == 1, "one record per discovered image, no phantom records"
    (ref,) = rep.unresolved_references
    assert ref.image_ref == "image_099" and ref.origin == "mapping.csv:line 3"
    assert "applies to nothing" in ref.reason
    assert any("image_099" in i for i in rep.issues) and rep.has_any_issue()


def test_ambiguous_image_reference_in_a_mapping_file_is_refused(fake_e57, tmp_path):
    mapping = tmp_path / "mapping.csv"
    mapping.write_text("scan_id,image_name\nscan_000,pano\n")
    rep = _report(
        fake_e57,
        _scans(GUID_A),
        [image_node(name="pano"), image_node(name="pano")],
        mapping=mapping,
    )
    (ref,) = rep.unresolved_references
    assert "refer to one image by image_id" in ref.reason
    assert [m.status for m in rep.mappings] == ["unmapped", "unmapped"]


def test_mapping_file_formats_and_refusals(tmp_path):
    csv_path = tmp_path / "m.csv"
    csv_path.write_text("image_id,scan_guid\nimage_000,{g}\n")
    (entry,) = read_mapping_file(csv_path)
    assert entry.image_ref == "image_000" and entry.image_ref_kind == "image_id"
    assert entry.target.kind == "scan_guid" and entry.target.value == "{g}"

    json_path = tmp_path / "m.json"
    json_path.write_text(
        json.dumps({"mappings": [{"image_id": "image_001", "station_id": "S003"}]})
    )
    (entry,) = read_mapping_file(json_path)
    assert entry.target.kind == "station_id" and entry.origin == "m.json:record 1"

    headerless = tmp_path / "bad.csv"
    headerless.write_text("S000,pano_a\n")
    with pytest.raises(ContractError, match="does not name an image column"):
        read_mapping_file(headerless)

    half = tmp_path / "half.csv"
    half.write_text("image_id,scan_id\nimage_000,\n")
    with pytest.raises(ContractError, match="no target"):
        read_mapping_file(half)

    both = tmp_path / "both.csv"
    both.write_text("image_id,scan_id,station_id\nimage_000,scan_000,S000\n")
    with pytest.raises(ContractError, match="more than one image or target column"):
        read_mapping_file(both)

    with pytest.raises(ContractError, match="mapping file not found"):
        read_mapping_file(tmp_path / "nope.csv")


# ---------------------------------------------------------------- Tier C: vendor manifest


def test_vendor_manifest_is_confirmed_evidence(fake_e57, tmp_path):
    manifest = tmp_path / "vendor.json"
    manifest.write_text(
        json.dumps(
            {
                "vendor": "Matterport",
                "generated_by": "cortex 1.2",
                "mappings": [{"image_id": "image_000", "scan_guid": GUID_B}],
            }
        )
    )
    rep = _report(
        fake_e57,
        _scans(GUID_A, GUID_B),
        [image_node(associated_scan_guid=None)],
        vendor_manifest=manifest,
    )
    m = rep.record_for("image_000")
    assert m.status == "confirmed" and m.evidence_type == "vendor_manifest"
    assert m.scan_id == "scan_001" and "machine-generated" in m.reason


def test_a_hand_written_file_cannot_pose_as_a_vendor_manifest(tmp_path):
    """Otherwise Tier C is just Tier B with a better-looking status."""
    csv_path = tmp_path / "mine.csv"
    csv_path.write_text("scan_id,image_id\nscan_000,image_000\n")
    with pytest.raises(ContractError, match="must be JSON declaring"):
        read_vendor_manifest(csv_path)

    undeclared = tmp_path / "v.json"
    undeclared.write_text(json.dumps({"mappings": [{"image_id": "image_000", "scan_id": "s"}]}))
    with pytest.raises(ContractError, match="'vendor' and 'generated_by'"):
        read_vendor_manifest(undeclared)


def test_vendor_manifest_disagreeing_with_the_file_is_a_conflict(fake_e57, tmp_path):
    manifest = tmp_path / "vendor.json"
    manifest.write_text(
        json.dumps(
            {
                "vendor": "Acme",
                "generated_by": "exporter",
                "mappings": [{"image_id": "image_000", "scan_id": "scan_001"}],
            }
        )
    )
    rep = _report(
        fake_e57,
        _scans(GUID_A, GUID_B),
        [image_node(associated_scan_guid=GUID_A)],
        vendor_manifest=manifest,
    )
    assert rep.record_for("image_000").status == "conflict"


# ---------------------------------------------------------------- external images (§15)


def _write_png(path, size=(8, 4)):
    from PIL import Image

    Image.new("RGB", size, (10, 20, 30)).save(path)
    return path


def test_external_image_discovery_declares_no_projection(tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    _write_png(d / "pano_b.png", (16, 8))
    _write_png(d / "pano_a.png", (8, 4))
    (d / "notes.txt").write_text("ignored")

    assets = discover_external_images(d)
    assert [a.image_id for a in assets] == ["image_000", "image_001"]
    assert [a.name for a in assets] == ["pano_a.png", "pano_b.png"], "sorted, for a stable index"
    assert [a.representation for a in assets] == ["unknown", "unknown"]
    assert not any(a.panorama_candidate for a in assets)
    assert [(a.width, a.height) for a in assets] == [(8, 4), (16, 8)]
    assert all(a.source == "external_file" and a.is_extractable for a in assets)
    assert all(any("not declared anywhere" in n for n in a.notes) for a in assets)


def test_external_images_are_unmapped_without_an_explicit_mapping(fake_e57, tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    _write_png(d / "S000.png")
    _write_png(d / "S001.png")
    path, _ = fake_e57(_scans(GUID_A, GUID_B))
    rep = build_mapping_report(path, images_dir=d, compute_hash=False)
    assert rep.image_root == str(d.resolve())
    assert [m.status for m in rep.mappings] == ["unmapped", "unmapped"]
    assert all(m.scan_id is None for m in rep.mappings), "a file name is not a station id"


def test_external_images_map_with_an_explicit_mapping(fake_e57, tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    _write_png(d / "left.png")
    _write_png(d / "right.png")
    mapping = tmp_path / "map.csv"
    mapping.write_text("station_id,image_name\nS001,left.png\nS000,right.png\n")
    path, _ = fake_e57(_scans(GUID_A, GUID_B))
    rep = build_mapping_report(path, images_dir=d, mapping=mapping, compute_hash=False)
    assert [(m.image_id, m.scan_id, m.status) for m in rep.mappings] == [
        ("image_000", "scan_001", "manual"),
        ("image_001", "scan_000", "manual"),
    ]


def test_external_path_that_is_not_a_directory_fails_closed(tmp_path, fake_e57):
    from minegs.ingest.e57.exceptions import E57NotAFileError

    f = tmp_path / "a.jpg"
    _write_png(f)
    path, _ = fake_e57(_scans(GUID_A))
    with pytest.raises(E57NotAFileError):
        build_mapping_report(path, images_dir=f, compute_hash=False)


def test_an_unreadable_external_image_is_reported_not_skipped(tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    (d / "broken.jpg").write_bytes(b"not an image")
    (a,) = discover_external_images(d)
    assert a.image_id == "image_000" and (a.width, a.height) == (None, None)
    assert any("could not read image header" in i for i in a.issues)


def test_vendor_pose_and_intrinsics_are_kept_verbatim(fake_e57):
    """Recorded for later phases, interpreted by none of them here."""
    node = image_node(
        rep_extra={"focalLength": 0.012, "principalPointX": 2048.0},
        extra={
            "pose": FakeNode(
                "pose",
                {
                    "rotation": FakeNode("rotation", {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}),
                    "translation": FakeNode("translation", {"x": 1.0, "y": 2.0, "z": 3.0}),
                },
            )
        },
    )
    path, _ = fake_e57(_scans(GUID_A), root=root_with_images(node))
    (a,) = discover_embedded_images(path)
    assert a.vendor_metadata == {
        "focalLength": 0.012,
        "principalPointX": 2048.0,
        "pose_rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "pose_translation": [1.0, 2.0, 3.0],
    }


def test_vendor_export_refuses_to_read_stations_from_file_names(tmp_path):
    """§10: an undocumented naming convention holds until an export renumbers, then every
    panorama is attributed to the wrong station silently."""
    from minegs.ingest.e57.pano.vendor_export import VendorExport

    root = tmp_path / "vendor"
    root.mkdir()
    for name in ("Station_01.jpg", "Station_02.jpg"):
        _write_png(root / name)

    with pytest.raises(ContractError, match="not inferred from file names"):
        VendorExport(root)
    with pytest.raises(ContractError, match="vendor index not found"):
        VendorExport(root, index=root / "nope.csv")

    index = tmp_path / "index.csv"
    index.write_text("station_id,pano_id\nS000,Station_01.jpg\nS001,Station_02.jpg\n")
    src = VendorExport(root, index=index)
    assert [(r.station_id, r.pano_id) for r in src.list_panoramas()] == [
        ("S000", "Station_01.jpg"),
        ("S001", "Station_02.jpg"),
    ]


def test_missing_external_directory_fails_closed(tmp_path, fake_e57):
    from minegs.ingest.e57.exceptions import E57FileNotFoundError

    path, _ = fake_e57(_scans(GUID_A))
    with pytest.raises(E57FileNotFoundError):
        build_mapping_report(path, images_dir=tmp_path / "nope", compute_hash=False)


# ---------------------------------------------------------------- the report artifact


def test_report_round_trips_through_json(fake_e57, tmp_path):
    rep = _report(
        fake_e57,
        _scans(GUID_A, GUID_B),
        [image_node(associated_scan_guid=GUID_A), image_node(associated_scan_guid=None)],
    )
    out = rep.save(tmp_path / "pano_mapping.json")
    reloaded = PanoMappingReport.load(out)
    assert reloaded.schema_version == "1.0"
    assert reloaded.model_dump() == rep.model_dump()
    raw = json.loads(out.read_text())
    assert raw["mappings"][0]["status"] == "confirmed"
    assert raw["mappings"][1]["scan_id"] is None
    assert set(raw) >= {
        "schema_version",
        "source_file",
        "source_sha256",
        "images",
        "mappings",
        "issues",
        "notes",
        "provenance",
    }


def test_report_records_provenance_and_the_source_hash(fake_e57):
    path, _ = fake_e57(_scans(GUID_A), root=root_with_images(image_node()))
    rep = build_mapping_report(path, compute_hash=True)
    assert rep.source_sha256 and len(rep.source_sha256) == 64
    assert rep.hash_skipped_reason is None
    asset = rep.provenance.source_assets[0]
    assert asset.sha256 == rep.source_sha256 and asset.size_bytes == path.stat().st_size
    assert rep.provenance.minegs_version and "python" in rep.provenance.tool_versions

    skipped = build_mapping_report(path, compute_hash=False)
    assert skipped.source_sha256 is None and skipped.hash_skipped_reason


def test_every_mapping_input_is_hashed_into_the_report(fake_e57, tmp_path):
    """The same E57 with two different CSVs is two different reports.

    A provenance record naming only the E57 cannot tell them apart, so it cannot reproduce
    either — and a mapping file is the one input a person edits between runs.
    """
    from minegs.core.provenance import sha256_file

    mapping = tmp_path / "mapping.csv"
    mapping.write_text("scan_id,image_id\nscan_000,image_000\n")
    manifest = tmp_path / "vendor.json"
    manifest.write_text(
        json.dumps(
            {
                "vendor": "Acme",
                "generated_by": "exporter 2.1",
                "mappings": [{"image_id": "image_001", "scan_id": "scan_000"}],
            }
        )
    )
    path, _ = fake_e57(_scans(GUID_A), root=root_with_images(image_node(), image_node()))
    rep = build_mapping_report(path, mapping=mapping, vendor_manifest=manifest)

    # vendor first: the report reads in evidence-tier order
    assert [i.role for i in rep.mapping_inputs] == ["vendor_manifest", "explicit_mapping"]
    by_role = {i.role: i for i in rep.mapping_inputs}
    assert by_role["explicit_mapping"].path == str(mapping.resolve())
    assert by_role["explicit_mapping"].sha256 == sha256_file(mapping)
    assert by_role["vendor_manifest"].sha256 == sha256_file(manifest)
    assert by_role["vendor_manifest"].row_count == 1
    assert all(i.size_bytes > 0 for i in rep.mapping_inputs)

    recorded = {a.path: a.sha256 for a in rep.provenance.source_assets}
    assert recorded[str(path.resolve())] == rep.source_sha256
    assert recorded[str(mapping.resolve())] == sha256_file(mapping)
    assert recorded[str(manifest.resolve())] == sha256_file(manifest)


def test_a_report_with_no_mapping_file_records_none(fake_e57):
    rep = _report(fake_e57, _scans(GUID_A), [image_node()])
    assert rep.mapping_inputs == []
    assert len(rep.provenance.source_assets) == 1


def test_external_images_are_hashed_into_the_report(tmp_path, fake_e57):
    from minegs.core.provenance import sha256_file

    d = tmp_path / "images"
    d.mkdir()
    _write_png(d / "a.png")
    _write_png(d / "b.png", (16, 8))
    path, _ = fake_e57(_scans(GUID_A))
    rep = build_mapping_report(path, images_dir=d)

    assert [a.sha256 for a in rep.images] == [sha256_file(d / "a.png"), sha256_file(d / "b.png")]
    assert all(a.hash_skipped_reason is None for a in rep.images)
    assert rep.image_root_sha256 and len(rep.image_root_sha256) == 64
    root = next(a for a in rep.provenance.source_assets if a.path == str(d.resolve()))
    assert root.sha256 == rep.image_root_sha256
    assert root.size_bytes == sum(f.stat().st_size for f in (d / "a.png", d / "b.png"))


def test_the_image_set_digest_moves_when_the_set_does(tmp_path, fake_e57):
    d = tmp_path / "images"
    d.mkdir()
    _write_png(d / "a.png")
    path, _ = fake_e57(_scans(GUID_A))
    one = build_mapping_report(path, images_dir=d).image_root_sha256
    _write_png(d / "b.png", (16, 8))
    two = build_mapping_report(path, images_dir=d).image_root_sha256
    assert one and two and one != two
    assert build_mapping_report(path, images_dir=d).image_root_sha256 == two, "and is stable"


def test_no_hash_leaves_no_partial_digest_anywhere(tmp_path, fake_e57):
    """A digest covering only the inputs that happened to be hashed is worse than none."""
    d = tmp_path / "images"
    d.mkdir()
    _write_png(d / "a.png")
    mapping = tmp_path / "mapping.csv"
    mapping.write_text("scan_id,image_name\nscan_000,a.png\n")
    path, _ = fake_e57(_scans(GUID_A))
    rep = build_mapping_report(path, images_dir=d, mapping=mapping, compute_hash=False)

    assert rep.source_sha256 is None and rep.hash_skipped_reason
    assert rep.image_root_sha256 is None
    assert rep.images[0].sha256 is None and rep.images[0].hash_skipped_reason
    assert rep.mapping_inputs[0].sha256 is None
    assert rep.mapping_inputs[0].hash_skipped_reason
    assert rep.record_for("image_000").status == "manual", "mapping still works without hashes"


def test_a_precomputed_source_hash_is_used_instead_of_rereading(tmp_path, fake_e57, monkeypatch):
    """Extraction hashes the E57 once for three artifacts; this is the seam that lets it."""
    import minegs.ingest.e57.mapping as mapping_mod

    def forbidden(*a, **k):
        raise AssertionError("the caller already supplied the digest")

    path, _ = fake_e57(_scans(GUID_A), root=root_with_images(image_node()))
    monkeypatch.setattr(mapping_mod, "sha256_file", forbidden)
    rep = build_mapping_report(path, compute_hash=False, source_sha256="a" * 64)
    assert rep.source_sha256 == "a" * 64 and rep.hash_skipped_reason is None
    assert rep.provenance.source_assets[0].sha256 == "a" * 64


def test_one_file_cannot_be_both_evidence_tiers(tmp_path, fake_e57):
    """Passing the same JSON twice would make it agree with itself and report confirmed."""
    both = tmp_path / "v.json"
    both.write_text(
        json.dumps(
            {
                "vendor": "Acme",
                "generated_by": "exporter",
                "mappings": [{"image_id": "image_000", "scan_id": "scan_000"}],
            }
        )
    )
    path, _ = fake_e57(_scans(GUID_A), root=root_with_images(image_node()))
    with pytest.raises(ContractError, match="both --mapping and --vendor-manifest"):
        build_mapping_report(path, mapping=both, vendor_manifest=both, compute_hash=False)


def test_an_empty_image_directory_still_gets_a_digest(tmp_path, fake_e57):
    """ "I looked here and found nothing" is a fact about the run, not a missing measurement."""
    d = tmp_path / "images"
    d.mkdir()
    path, _ = fake_e57(_scans(GUID_A))
    rep = build_mapping_report(path, images_dir=d)
    assert rep.images == []
    assert rep.image_root_sha256 and len(rep.image_root_sha256) == 64
    root = next(a for a in rep.provenance.source_assets if a.path == str(d.resolve()))
    assert root.sha256 == rep.image_root_sha256, "never an empty hash claiming to be one"


def test_an_external_directory_says_it_has_no_embedded_evidence(tmp_path, fake_e57):
    """The laundering path, named out loud: extract images, then re-map them without the E57.

    Nothing on that path can contradict a mapping file, so a conflict the E57 would have
    raised quietly becomes a clean manual mapping. The report says so rather than looking the
    same as a mapping made against the file itself.
    """
    d = tmp_path / "images"
    d.mkdir()
    _write_png(d / "a.png")
    path, _ = fake_e57(_scans(GUID_A))
    rep = build_mapping_report(path, images_dir=d, compute_hash=False)
    note = " ".join(rep.notes)
    assert "nothing to contradict" in note
    assert "name order, not the first entry in /images2D" in note


def test_hints_from_every_mapping_row_survive(fake_e57, tmp_path):
    """A near-miss on row 7 is exactly the one the user needs to see."""
    mapping = tmp_path / "mapping.csv"
    near_a = GUID_A.strip("{}").upper()
    near_b = GUID_B.strip("{}").upper()
    mapping.write_text(f"scan_guid,image_id\n{near_a},image_000\n{near_b},image_000\n")
    rep = _report(
        fake_e57, _scans(GUID_A, GUID_B), [image_node(associated_scan_guid=None)], mapping=mapping
    )
    m = rep.record_for("image_000")
    # neither GUID matches exactly, so both rows resolve to nothing: an orphan, with a
    # near-miss hint from each row rather than only from the first
    assert m.status == "orphan" and m.scan_id is None
    assert len(m.hints) == 2, m.hints
    assert all("differ only in case, braces or hyphens" in h for h in m.hints)


def test_report_never_claims_a_tls_global_frame(fake_e57):
    rep = _report(fake_e57, _scans(GUID_A), [image_node(associated_scan_guid=GUID_A)])
    dumped = rep.model_dump_json()
    assert "TLS_GLOBAL" not in dumped and "LOCAL_METRIC" not in dumped


def test_report_helpers_partition_by_status(fake_e57):
    rep = _report(
        fake_e57,
        _scans(GUID_A, GUID_A),
        [image_node(associated_scan_guid=GUID_A), image_node(associated_scan_guid=None)],
    )
    assert [m.image_id for m in rep.by_status("ambiguous")] == ["image_000"]
    assert [m.image_id for m in rep.by_status("unmapped")] == ["image_001"]
    assert rep.resolved() == []
    with pytest.raises(KeyError):
        rep.record_for("image_404")


def test_mapping_does_not_read_point_data(fake_e57):
    """Mapping is metadata only; the fake fails any read (§17 applies here too)."""
    path, opened = fake_e57(
        [make_scan(guid=GUID_A, point_count=900_000_000)],
        root=root_with_images(image_node(associated_scan_guid=GUID_A)),
    )
    rep = build_mapping_report(path, compute_hash=False)
    assert rep.record_for("image_000").status == "confirmed"
    assert all(h.reads == [] for h in opened) and all(h.closed for h in opened)


def test_an_image_model_cannot_carry_an_unknown_field():
    """extra="forbid": a mapping must not be smuggled onto the asset that discovery produced."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ImageAsset(image_id="image_000", source="e57_embedded", source_index=0, station_id="S000")


# ---------------------------------------------------------------- CLI


def test_pano_map_cli_reports_without_guessing(fake_e57, tmp_path, capsys):
    from minegs.cli.ingest import _print_mapping

    path, _ = fake_e57(
        _scans(GUID_A, GUID_B),
        root=root_with_images(
            image_node(associated_scan_guid=GUID_A, name="pano_a"),
            image_node(associated_scan_guid=None, name="pano_b"),
        ),
    )
    rep = build_mapping_report(path, compute_hash=False)
    _print_mapping(rep)
    out = capsys.readouterr().out
    assert "image_000" in out and "scan_000" in out and "confirmed" in out
    assert "unmapped" in out
    assert "S000" in out


def test_inventory_and_mapping_agree_about_image_count(fake_e57):
    path, _ = fake_e57(
        _scans(GUID_A), root=root_with_images(image_node(), image_node(), image_node())
    )
    assert inventory(path, compute_hash=False).images.image_count == 3
    assert len(build_mapping_report(path, compute_hash=False).images) == 3
