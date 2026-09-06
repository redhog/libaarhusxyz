import pandas as pd
import numpy as np


def normalize_layer_depths(data):
    """Normalizes all layer depths, so that layer X is at the same depth
    in all soundings. This is done by upsampling all soundings to a
    set of layers with boundaries (top and bottom) being the union of
    all boundaries from all the soundings. Note that this is just pure
    upsampling, not interpolation.
    """
    d_bot = data["layer_data"]["dep_bot"]
    groups = data["layer_data"]["dep_top"].groupby(by=list(data["layer_data"]["dep_top"].columns)).groups

    unique_depths = np.unique(d_bot.values.flatten())
    u_d = pd.DataFrame({"unique": pd.Series(unique_depths)})

    res_layer_data = {param: pd.DataFrame(index=data["layer_data"][param].index,
                                          columns=np.arange(len(u_d.unique)))
                      for param in data["layer_data"].keys()}

    for g1 in groups.values():
        g1_boundaries = pd.DataFrame({"top": data["layer_data"]["dep_top"].loc[g1].iloc[0],
                                      "bot": data["layer_data"]["dep_bot"].loc[g1].iloc[0]}).fillna(np.inf)
        for dest_layer, (top, bot) in enumerate(zip([0] + list(u_d.unique), list(u_d.unique))):
            source_layer = g1_boundaries.index[(g1_boundaries.top <= top) & (g1_boundaries.bot >= bot)][0]
            
            for param in res_layer_data.keys():
                if param in ("dep_top", "dep_bot"):
                    continue
                if source_layer not in data["layer_data"][param].columns:
                    param_data = np.nan
                else:
                    param_data = data["layer_data"][param].loc[g1, source_layer]
                res_layer_data[param].loc[g1, dest_layer] = param_data
            res_layer_data["dep_top"].loc[g1, dest_layer] = top
            res_layer_data["dep_bot"].loc[g1, dest_layer] = bot

    data = dict(data)
    data["layer_data"] = res_layer_data
    return data


# ---------------------------------------------------------------------------
# Workbench dat/syn exports: two rows per sounding -> one row, per-moment gates
# ---------------------------------------------------------------------------

# Workbench writes dB/dt in V/(A m^4); a delivered SkyTEM XYZ - what importers
# are built for, with a scale factor of 1e-12 - carries the same quantity in
# pV/(A m^4). Both are positive for a normal decay (checked on the SkyTEM 304:
# 95.5% of exported values positive, the rest noisy late gates; per-gate medians
# of the converted export agree with the delivered file to within 6%). The sign
# stays a parameter because a source writing negative decays exists (SimPEG's
# convention), and a silent sign flip inverts to a plausible wrong model.
WORKBENCH_TO_DELIVERED_SCALE = 1e12
WORKBENCH_TO_DELIVERED_SIGN = 1.0

_SEGMENT_COL = "segment"
_LINE_COL = "line"
_FID_COL = "fid"
_DATA_KEY = "data"
_STD_KEY = "datastd"
_MASTER_TIMES_KEY = "gate times"
_DUMMY_KEY = "dummy"
_DEFAULT_DUMMY = 9999.0

# Workbench sounding column -> the canonical (ALC) name importers understand.
# Everything else stays under its Workbench name. resdata/restotal are the
# contractor's own misfit: kept, since they are the target to match, but under
# a name that cannot be mistaken for a residual computed later.
WORKBENCH_FLIGHTLINE_RENAMES = {
    "line": "Line", "fid": "Fid", "x": "UTMX", "y": "UTMY",
    "topo": "Topography", "invalt": "TxAltitude",
    "resdata": "wb_resdata", "restotal": "wb_restotal",
}
# Per-segment bookkeeping that means nothing once the pair is one row.
_FLIGHTLINE_DROPS = ("segment", "numdata")


class WorkbenchSplitError(ValueError):
    """The export and the GEX do not describe the same thing."""


def is_workbench_export(xyz):
    """True for an Aarhus Workbench ``_MOD_dat`` / ``_MOD_syn`` export.

    The signature is a ``segment`` column in the flightlines together with a
    ``data`` gate block in ``layer_data``. Accepts an ``XYZ`` or its dict form.
    """
    fl, ld = _flightlines_and_layer_data(xyz)
    return _SEGMENT_COL in fl.columns and _DATA_KEY in ld


def workbench_gate_map(master_times, gex, channels=(1, 2), tol=5e-3):
    """Map the export's master gate-time columns onto GEX gate indices, per channel.

    Returns ``({channel: {master_col: gate_idx}}, report)``. The export header
    carries one gate-time array, the time-sorted union of both moments' gate
    times. For each channel both time conventions - GEX gate centre minus that
    channel's ``GateTimeShift``, and centre as written - are tried and the one
    matching more columns wins; the report says which. (Workbench applied the
    shift to one moment and not the other in the files this was built on, so
    it is tested, not assumed.)

    The tolerance is relative; 0.5% is far inside the 10-30% spacing between
    neighbouring gates, and wide enough for a shared gate whose two centres
    differ slightly (277.7 us on the 304: LM 276.7 us, HM 277.7 us). A master
    column matching two gates of one channel, or two columns one gate, is an
    error: the map must be one-to-one where it exists.
    """
    T = np.asarray(master_times, dtype=float)
    maps, report = {}, {}
    for ch in channels:
        block = gex.gex_dict["Channel%d" % ch]
        centres = np.asarray(gex.gate_times(ch), dtype=float)
        centres = centres[:, 0] if centres.ndim == 2 else centres
        shift = float(block.get("GateTimeShift", 0.0))
        best = None
        for convention, cand in (("centre - GateTimeShift", centres - shift), ("centre", centres)):
            idx = np.argmin(np.abs(T[:, None] - cand[None, :]), axis=1)
            ok = np.abs(cand[idx] - T) <= tol * np.abs(T)
            m = {int(j): int(idx[j]) for j in np.flatnonzero(ok)}
            if best is None or len(m) > len(best[1]):
                best = (convention, m)
        convention, m = best
        gates = list(m.values())
        if len(set(gates)) != len(gates):
            dup = sorted({g for g in gates if gates.count(g) > 1})
            raise WorkbenchSplitError(
                "channel %d: GEX gate(s) %s matched by more than one master column" % (ch, dup))
        maps[ch] = m
        report[ch] = {"convention": convention, "n": len(m), "n_gates": len(centres),
                      "gates": sorted(set(gates))}
    return maps, report


def split_workbench_moments(xyz, gex, tol=5e-3,
                            scale=WORKBENCH_TO_DELIVERED_SCALE,
                            sign=WORKBENCH_TO_DELIVERED_SIGN):
    """Turn a Workbench dat/syn export into one row per sounding with per-moment gates.

    Aarhus Workbench exports the data that went into an inversion (``_MOD_dat``)
    and a model's forward response (``_MOD_syn``) as **two rows per sounding**,
    one per transmitter moment (``segment`` 1 = LM, 2 = HM), over one set of
    gate columns that is the union of both moments' gate times sorted by time.
    A segment-1 row holds values only in the LM-time columns and the dummy
    elsewhere; a segment-2 row the converse; a gate time both moments share is
    filled in both. Gates culled for a sounding are dummy too.

    An importer wants the layout of a delivered SkyTEM file: one row per
    sounding, ``Gate_Ch01`` (LM) and ``Gate_Ch02`` (HM) with one column per GEX
    gate, ``STD_Ch0n`` where the export carries ``datastd``, ``InUse_Ch0n``
    flags, positive pV/(A m^4). That is what this returns.

    The column -> moment map comes from the GEX (:func:`workbench_gate_map`),
    not from the data, so it also assigns columns that are dummy in every row.
    Because the map lands on GEX gate *indices*, the output carries every GEX
    gate; gates absent from the export or culled for a sounding are NaN with
    in-use 0, so the file imports against the standard GEX for the system - no
    gate-matched GEX is needed. The data-derived map (which segment carries a
    real value in each column) is kept as a cross-check and the split refuses
    if the two disagree on any column that carries data.

    Workbench writes no row for a moment it culled entirely; such a sounding is
    kept with that moment all NaN rather than dropped, so the output has every
    sounding the published model has.

    ``xyz`` is the parsed export (``XYZ`` or its dict form); ``gex`` the
    system's ``GEX``. Returns a new ``XYZ``.
    """
    from .xyz import XYZ

    fl, layer_data = _flightlines_and_layer_data(xyz)
    model_info = dict(xyz.model_info if hasattr(xyz, "model_info") else xyz.get("model_info", {}))
    for col in (_SEGMENT_COL, _LINE_COL, _FID_COL):
        if col not in fl.columns:
            raise WorkbenchSplitError("not a Workbench dat/syn export: no %r column" % col)
    if _DATA_KEY not in layer_data:
        raise WorkbenchSplitError("not a Workbench dat/syn export: no %r gate block" % _DATA_KEY)
    master = model_info.get(_MASTER_TIMES_KEY)
    if master is None:
        raise WorkbenchSplitError("export header carries no %r line" % _MASTER_TIMES_KEY)
    master = np.asarray(master, dtype=float)
    data = layer_data[_DATA_KEY].to_numpy(dtype=float)
    if data.shape[1] != len(master):
        raise WorkbenchSplitError("%d gate columns but %d master gate times" % (data.shape[1], len(master)))
    std = layer_data[_STD_KEY].to_numpy(dtype=float) if _STD_KEY in layer_data else None
    dummy = float(model_info.get(_DUMMY_KEY, _DEFAULT_DUMMY))

    maps, report = workbench_gate_map(master, gex, tol=tol)
    lm, hm = maps[1], maps[2]
    unmatched = [j for j in range(len(master)) if j not in lm and j not in hm]
    if unmatched:
        raise WorkbenchSplitError("master gate times %s match no gate of either channel: %s"
                                  % (unmatched, master[unmatched]))

    p1, p2, counts = _pair_rows(fl)
    if len(p1) == 0:
        raise WorkbenchSplitError("no soundings")
    real = ~(np.isclose(data, dummy) | np.isnan(data))

    seg1_has = real[p1[p1 >= 0]].any(axis=0)
    seg2_has = real[p2[p2 >= 0]].any(axis=0)
    bad = [j for j in range(len(master)) if (seg1_has[j] and j not in lm) or (seg2_has[j] and j not in hm)]
    if bad:
        raise WorkbenchSplitError("GEX map and data disagree on column(s) %s (master times %s)"
                                  % (bad, master[bad]))

    n = len(p1)
    out_layers = {}
    for ch, m, rows in ((1, lm, p1), (2, hm, p2)):
        n_gates = report[ch]["n_gates"]
        gate = np.full((n, n_gates), np.nan)
        sd = np.full((n, n_gates), np.nan) if std is not None else None
        present = np.flatnonzero(rows >= 0)
        src = rows[present]
        for j, k in m.items():
            keep = real[src, j]
            dst = present[keep]
            gate[dst, k] = sign * scale * data[src[keep], j]
            if sd is not None:
                sd[dst, k] = std[src[keep], j]
        finite = gate[np.isfinite(gate)]
        if finite.size and np.median(finite) < 0:
            raise WorkbenchSplitError(
                "channel %d: converted dB/dt is negative on the whole (median %.3g); delivered "
                "SkyTEM data is positive - check this export's sign convention" % (ch, np.median(finite)))
        chan = "%02d" % ch
        out_layers["Gate_Ch" + chan] = pd.DataFrame(gate)
        if sd is not None:
            out_layers["STD_Ch" + chan] = pd.DataFrame(sd)
        out_layers["InUse_Ch" + chan] = pd.DataFrame((~np.isnan(gate)).astype(np.int8))

    keep_cols = [c for c in fl.columns if c not in _FLIGHTLINE_DROPS]
    aux_rows = np.where(p1 >= 0, p1, p2)
    out_fl = fl.iloc[aux_rows][keep_cols].reset_index(drop=True)
    out_fl = out_fl.rename(columns={c: WORKBENCH_FLIGHTLINE_RENAMES[c]
                                    for c in keep_cols if c in WORKBENCH_FLIGHTLINE_RENAMES})

    for k in ("number of gates", _MASTER_TIMES_KEY, _DUMMY_KEY):
        model_info.pop(k, None)
    model_info["gate layout"] = (
        "one row per sounding; Gate_Ch01 = LM on GEX Channel1 gates (%d), Gate_Ch02 = HM on GEX "
        "Channel2 gates (%d); gates absent from the Workbench export or culled for a sounding are "
        "dummy with InUse 0" % (report[1]["n_gates"], report[2]["n_gates"]))
    model_info["units"] = ("dB/dt in pV/(A m^4), positive, as delivered by SkyTEM "
                           "(Workbench V/(A m^4) x %g)" % (sign * scale))
    model_info["moment split"] = (
        "Workbench dat/syn two-rows-per-sounding export merged by GEX gate times (LM: %s, HM: %s); "
        "libaarhusxyz.transforms.split_workbench_moments"
        % (report[1]["convention"], report[2]["convention"]))
    model_info["moment split counts"] = dict(
        lm_columns=report[1]["n"], hm_columns=report[2]["n"], shared=len(set(lm) & set(hm)),
        soundings=n, hm_only=counts["hm_only"], lm_only=counts["lm_only"])

    return XYZ({"flightlines": out_fl, "layer_data": out_layers, "model_info": model_info})


def _flightlines_and_layer_data(xyz):
    if hasattr(xyz, "flightlines"):
        return xyz.flightlines, xyz.layer_data
    return xyz["flightlines"], xyz["layer_data"]


def _pair_rows(fl):
    """Row positions of each sounding's segment-1 and segment-2 rows, in file order.

    Returns ``(p1, p2, counts)``; a position is -1 where the sounding has no row
    for that moment. Soundings are keyed by (line, fid); a duplicate row for one
    moment is an error, since nothing here could choose between them.
    """
    seg = fl[_SEGMENT_COL].to_numpy()
    keys = list(zip(fl[_LINE_COL].to_numpy(), fl[_FID_COL].to_numpy()))
    pos = {1: {}, 2: {}}
    order, seen = [], set()
    for i, (k, s) in enumerate(zip(keys, seg)):
        if s not in pos:
            raise WorkbenchSplitError("row %d: segment %r is neither 1 (LM) nor 2 (HM)" % (i, s))
        if k in pos[s]:
            raise WorkbenchSplitError("sounding %r has more than one segment-%d row" % (k, s))
        pos[s][k] = i
        if k not in seen:
            seen.add(k)
            order.append(k)
    p1 = np.asarray([pos[1].get(k, -1) for k in order], dtype=int)
    p2 = np.asarray([pos[2].get(k, -1) for k in order], dtype=int)
    counts = {"lm_only": int((p2 < 0).sum()), "hm_only": int((p1 < 0).sum())}
    return p1, p2, counts
