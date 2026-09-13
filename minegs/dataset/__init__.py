"""Phase 0C — materialise a Phase 0B staging tree into the ``dataset/`` contract (§4).

The point of this package is not to *produce* a dataset but to *prove* that the frame chain
holds on real data::

    E57 SOURCE  --explicit SE(3)-->  TLS_GLOBAL  --deterministic origin-->  LOCAL_METRIC
                                                                              |
                                              COLMAP cameras + images + leak-free TLS init

so that TLS points, camera projections and ``init_points.ply`` land in one physical space —
checked by a reprojection overlay (``golden_gate``) and in Viser, not assumed.
"""
