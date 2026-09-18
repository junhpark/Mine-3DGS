"""A fake libE57 node tree, shared by every E57 contract test.

Real E57 files cannot cover the cases that matter most here: a scan with no pose, a pose node
that raises when touched, two scans sharing a GUID, an image declaring a representation this
reader does not model. Vendors produce those; test fixtures cannot be asked to. So the reader
is also run against a hand-built tree with libE57's accessor shapes (``childCount``, ``get``,
``isDefined``, ``elementName``, ``value``), which no pye57 install is needed to exercise.

The complementary layer is ``_write_real_e57`` in ``test_e57_inventory.py``: genuine libE57
output, written at test time so no binary fixture is committed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


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


class FakeBlob:
    """A blob node: libE57 exposes a byte count and a read-into-buffer call."""

    def __init__(self, name: str, payload: bytes) -> None:
        self._name, self.payload = name, payload

    def elementName(self) -> str:
        return self._name

    def byteCount(self) -> int:
        return len(self.payload)

    def read(self, buf: Any, start: int, count: int) -> int:
        data = np.frombuffer(self.payload[start : start + count], dtype=np.uint8)
        buf[: len(data)] = data
        return len(data)

    def childCount(self) -> int:
        return 0

    def isDefined(self, key: str) -> bool:
        return False

    def value(self) -> Any:
        raise FakeE57Error("a blob has no scalar value")


class ExplodingBlob(FakeBlob):
    """A blob whose bytes cannot be read, as a truncated file's would be."""

    def read(self, buf: Any, start: int, count: int) -> int:
        raise FakeE57Error("blob payload unreadable")


class FakeNode:
    """A structure node: ordered children, libE57-style accessors."""

    def __init__(self, name: str = "", children: dict[str, Any] | None = None) -> None:
        self._name = name
        self._children: dict[str, Any] = {}
        for k, v in (children or {}).items():
            self._children[k] = v if hasattr(v, "elementName") else FakeLeaf(k, v)

    def elementName(self) -> str:
        return self._name

    def childCount(self) -> int:
        return len(self._children)

    def get(self, index_or_name: Any) -> Any:
        """``VectorNode.get``, including pye57's missing downcast.

        pye57 binds ``get(int)`` straight to the C++ signature, which returns a base ``Node``;
        only ``__getitem__`` runs the result through ``cast_node``. So ``vec.get(i)`` hands back
        something with none of a structure node's accessors, and code that reads fields off it
        sees an entry that declares nothing. Modelling that here is the point: a fake that
        answered ``get(i)`` fully is what let the real reader ship broken.
        """
        if isinstance(index_or_name, int):
            return FakeBareNode(list(self._children.values())[index_or_name])
        return self[index_or_name]

    def isDefined(self, key: str) -> bool:
        return key in self._children

    def __getitem__(self, key: Any) -> Any:
        """``__getitem__``, which pye57 *does* downcast — by name or by index."""
        if isinstance(key, int):
            children = list(self._children.values())
            if not -len(children) <= key < len(children):
                raise IndexError(key)
            return children[key]
        if key not in self._children:
            raise FakeE57Error(f"node {key!r} is not defined")
        return self._children[key]

    def __len__(self) -> int:
        return len(self._children)


class FakeBareNode:
    """What ``VectorNode.get(i)`` returns in pye57: a base ``Node``, not a structure node.

    The real object carries only ``elementName``/``type``/``pathName``/``parent``-style
    accessors (checked against pye57 0.4.19). Reading a child off it raises, which is what
    ``minegs.ingest.e57._nodes`` turns into "this entry declares nothing".
    """

    def __init__(self, node: Any) -> None:
        self._node = node

    def elementName(self) -> str:
        return self._node.elementName()

    def type(self) -> str:
        return "E57_STRUCTURE"

    def pathName(self) -> str:
        return "/" + self._node.elementName()

    def isRoot(self) -> bool:
        return False


class ExplodingNode(FakeNode):
    """A node whose enumeration fails, as a corrupt or exotic file's would."""

    def childCount(self) -> int:
        raise FakeE57Error("cannot enumerate")


class FakeHeader:
    def __init__(
        self,
        node: FakeNode,
        point_fields: list[str],
        point_count: int | None,
        point_data: dict[str, Any] | BaseException | None = None,
    ) -> None:
        self.node = node
        self.point_fields = point_fields
        self._point_count = point_count
        #: Column arrays returned by ``read_scan_raw``. ``None`` means reading is a test
        #: failure, which is how the inventory's "never reads points" guarantee is enforced.
        #: An exception instance is raised instead, standing in for a truncated payload.
        self.point_data = point_data

    @property
    def point_count(self) -> int:
        if self._point_count is None:
            raise FakeE57Error("point count unavailable")
        return self._point_count

    def __getitem__(self, key: str) -> Any:
        return self.node[key]


class FakeE57:
    """Stands in for pye57.E57.

    Reading point data raises unless the header supplies ``point_data``: a module that is not
    supposed to touch the payload fails loudly rather than quietly working.
    """

    def __init__(self, headers: list[FakeHeader], root: FakeNode) -> None:
        self._headers = headers
        self.root = root
        self.closed = False
        self.reads: list[int] = []

    @property
    def scan_count(self) -> int:
        return len(self._headers)

    def get_header(self, index: int) -> FakeHeader:
        return self._headers[index]

    def read_scan(self, *a: Any, **k: Any) -> Any:
        raise AssertionError(
            "read_scan applies pye57's silently-falling-back pose; use read_scan_raw (§10)"
        )

    def read_scan_raw(self, index: int, *a: Any, **k: Any) -> Any:
        data = self._headers[index].point_data
        if data is None:
            raise AssertionError("this reader must never read point data (§17)")
        self.reads.append(index)
        if isinstance(data, BaseException):
            raise data
        return {k2: np.asarray(v) for k2, v in data.items()}

    def close(self) -> None:
        self.closed = True


def rotmat_from_quat(q: list[float]) -> np.ndarray:
    """Rotation from a quaternion WITHOUT normalising, matching what the reader builds."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def pose_node(quat_wxyz: tuple[float, ...], xyz: tuple[float, ...]) -> FakeNode:
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
    point_data: dict[str, Any] | BaseException | None = None,
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
    if isinstance(point_data, dict) and point_fields is None:
        fields = list(point_data)
    return FakeHeader(FakeNode("data3D", children), fields, point_count, point_data)


CARTESIAN_BOUNDS_OK = {
    "xMinimum": -5.0,
    "xMaximum": 5.0,
    "yMinimum": -4.0,
    "yMaximum": 4.0,
    "zMinimum": -1.0,
    "zMaximum": 2.0,
}


# ---------------------------------------------------------------- images2D


def image_node(
    representation: str | None = "sphericalRepresentation",
    guid: str | None = None,
    name: str | None = None,
    associated_scan_guid: str | None = None,
    width: int | None = 8,
    height: int | None = 4,
    blob_field: str | None = "jpegImage",
    payload: bytes = b"\xff\xd8\xff\xdbFAKEJPEG",
    blob: Any = None,
    rep_extra: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> FakeNode:
    """One ``/images2D`` entry, as a vendor would write it."""
    children: dict[str, Any] = {}
    if guid is not None:
        children["guid"] = guid
    if name is not None:
        children["name"] = name
    if associated_scan_guid is not None:
        children["associatedData3DGuid"] = associated_scan_guid
    if representation is not None:
        rep: dict[str, Any] = {}
        if width is not None:
            rep["imageWidth"] = width
        if height is not None:
            rep["imageHeight"] = height
        if blob_field is not None:
            rep[blob_field] = blob if blob is not None else FakeBlob(blob_field, payload)
        rep.update(rep_extra or {})
        children[representation] = FakeNode(representation, rep)
    children.update(extra or {})
    return FakeNode("image2D", children)


def root_with_images(*images: FakeNode, **extra: Any) -> FakeNode:
    """A root node carrying ``/images2D`` with the given entries, in order."""
    return FakeNode(
        "root",
        {
            "images2D": FakeNode("images2D", {f"image2D_{i}": im for i, im in enumerate(images)}),
            **extra,
        },
    )


def install_fake_pye57(
    monkeypatch: Any,
    tmp_path: Path,
    headers: list[FakeHeader],
    root: FakeNode | None = None,
    name: str = "f.e57",
) -> tuple[Path, list[FakeE57]]:
    """Point every E57 reader at a fake file. Returns (path, handles opened so far).

    One install covers all of them because they share ``_nodes.open_e57``; see
    ``test_every_e57_reader_opens_the_file_through_one_seam``.
    """
    from minegs.ingest.e57 import _nodes

    path = tmp_path / name
    path.write_bytes(b"not really an e57, the reader is faked")
    opened: list[FakeE57] = []

    class _Module:
        @staticmethod
        def E57(p: str, mode: str = "r") -> FakeE57:
            obj = FakeE57(headers, root if root is not None else FakeNode("root", {}))
            opened.append(obj)
            return obj

    monkeypatch.setattr(_nodes, "pye57_module", lambda: _Module)
    return path, opened
