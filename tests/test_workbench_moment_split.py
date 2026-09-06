"""Splitting a Workbench dat/syn export into per-moment gate groups, one row per sounding.

Aarhus Workbench exports inversion data and forward responses as two rows per
sounding (``segment`` 1 = LM, 2 = HM) over one set of gate columns - the union
of both moments' gate times, sorted by time and interleaved. Nothing imports
that. ``split_workbench_moments`` maps every column onto its GEX gate by gate
time and writes the delivered-SkyTEM layout instead.

The fixture is 40 soundings of a real SkyTEM 304 export (LPNNRD 2018, line
300901, header scrubbed of the operator's details) with the system's xyz GEX.
The numbers asserted below were established on the full line and verified
cell for cell against the Workbench original; the fixture reproduces them.

Regression tests for YmerFlow/libaarhusxyz#10.
"""

import os.path

import numpy as np
import pandas as pd
import pytest

import libaarhusxyz
from libaarhusxyz import transforms
from libaarhusxyz.transforms import (
    WorkbenchSplitError, is_workbench_export, split_workbench_moments, workbench_gate_map)


DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "workbench_dat_split")
DAT = os.path.join(DATA_DIR, "LPNNRD2018_SkyTEM304_line300901_MOD_dat_40soundings.xyz")
GEX = os.path.join(DATA_DIR, "SkyTEM304_xyz.gex")

N_LM, N_HM = 28, 37


@pytest.fixture(scope="module")
def gex():
    return libaarhusxyz.GEX(GEX)


@pytest.fixture
def export():
    return libaarhusxyz.XYZ(DAT, normalize=False)


@pytest.fixture
def split(export, gex):
    return split_workbench_moments(export, gex)


# ── Detecting the layout ─────────────────────────────────────────────────────

def test_detects_a_workbench_export(export):
    assert is_workbench_export(export)
    assert is_workbench_export(export.model_dict)


def test_a_delivered_file_is_not_one(export):
    x = libaarhusxyz.XYZ()
    x.flightlines = pd.DataFrame({"x": [0.0], "y": [0.0]})
    x.layer_data["Gate_Ch01"] = pd.DataFrame(np.ones((1, 3)))
    assert not is_workbench_export(x)


# ── The gate map, from the GEX alone ─────────────────────────────────────────

def test_every_master_column_lands_on_a_gate(export, gex):
    maps, report = workbench_gate_map(export.model_info["gate times"], gex)
    lm, hm = maps[1], maps[2]
    assert report[1]["n"] == 23 and report[2]["n"] == 30
    assert len(set(lm) & set(hm)) == 2, "two shared gate times on the 304"
    assert set(lm) | set(hm) == set(range(51)), "no column left unassigned"


def test_conventions_are_tested_per_channel_not_assumed(export, gex):
    """Workbench wrote LM times minus the GateTimeShift and HM times as-is."""
    _, report = workbench_gate_map(export.model_info["gate times"], gex)
    assert report[1]["convention"] == "centre - GateTimeShift"
    assert report[2]["convention"] == "centre"


def test_map_lands_on_gex_gate_indices(export, gex):
    maps, _ = workbench_gate_map(export.model_info["gate times"], gex)
    assert sorted(set(maps[1].values())) == [2, 3] + list(range(5, 26))
    assert sorted(set(maps[2].values())) == list(range(6, 36))


def test_two_columns_on_one_gate_is_refused(export, gex):
    times = list(export.model_info["gate times"])
    times[10] = times[9] * 1.001          # a second column at (nearly) gate 9's time
    with pytest.raises(WorkbenchSplitError, match="more than one master column"):
        workbench_gate_map(times, gex)


# ── The split ────────────────────────────────────────────────────────────────

def test_one_row_per_sounding_every_gex_gate(split):
    assert len(split.flightlines) == 40
    assert split.layer_data["Gate_Ch01"].shape == (40, N_LM)
    assert split.layer_data["Gate_Ch02"].shape == (40, N_HM)
    assert set(split.layer_data) == {"Gate_Ch01", "Gate_Ch02", "STD_Ch01", "STD_Ch02",
                                     "InUse_Ch01", "InUse_Ch02"}


def test_gates_workbench_never_exported_are_nan_and_not_in_use(split):
    g1 = split.layer_data["Gate_Ch01"]
    assert g1.iloc[:, [0, 1, 4]].isna().all().all()          # GEX gates 0, 1, 4: not in the export
    assert g1.iloc[:, 26:].isna().all().all()                 # 26, 27 likewise
    assert (split.layer_data["InUse_Ch01"].to_numpy().astype(bool) == ~g1.isna().to_numpy()).all()
    assert (split.layer_data["InUse_Ch02"].to_numpy().astype(bool)
            == ~split.layer_data["Gate_Ch02"].isna().to_numpy()).all()


def test_values_are_the_workbench_values_scaled_to_delivered_units(export, split, gex):
    """Every exported gate lands on its GEX gate, x1e12, sign unchanged."""
    maps, _ = workbench_gate_map(export.model_info["gate times"], gex)
    data = export.layer_data["data"].to_numpy(float)
    std = export.layer_data["datastd"].to_numpy(float)
    seg = export.flightlines["segment"].to_numpy()
    fl = export.flightlines
    checked = 0
    for si in range(40):
        line, fid = split.flightlines.loc[si, "Line"], split.flightlines.loc[si, "Fid"]
        for segno, m, G, S in ((1, maps[1], split.layer_data["Gate_Ch01"], split.layer_data["STD_Ch01"]),
                               (2, maps[2], split.layer_data["Gate_Ch02"], split.layer_data["STD_Ch02"])):
            rows = np.flatnonzero((fl["line"] == line) & (fl["fid"] == fid) & (seg == segno))
            assert len(rows) == 1
            r = rows[0]
            for j, k in m.items():
                v = data[r, j]
                if v == 9999.0:
                    assert np.isnan(G.iat[si, k])
                else:
                    assert G.iat[si, k] == pytest.approx(1e12 * v)
                    assert S.iat[si, k] == pytest.approx(std[r, j])
                    checked += 1
    assert checked > 1500


def test_uncertainties_are_relative_and_untouched(split):
    sd = split.layer_data["STD_Ch01"].to_numpy()
    sd = sd[np.isfinite(sd)]
    assert 0.02 < sd.min() and sd.max() < 0.5


def test_sounding_columns_are_renamed_for_import_and_kept_once(split):
    fl = split.flightlines
    for col in ("Line", "Fid", "UTMX", "UTMY", "Topography", "TxAltitude", "wb_resdata", "wb_restotal"):
        assert col in fl.columns, col
    for col in ("segment", "numdata", "x", "y", "resdata"):
        assert col not in fl.columns, col


def test_header_describes_the_new_layout(split):
    info = split.model_info
    assert "gate times" not in info and "number of gates" not in info
    assert "Gate_Ch01 = LM" in info["gate layout"]
    assert info["moment split counts"] == dict(lm_columns=23, hm_columns=30, shared=2,
                                               soundings=40, hm_only=0, lm_only=0)


# ── Soundings with one moment only ───────────────────────────────────────────

def test_a_sounding_with_only_an_hm_row_is_kept_with_lm_all_nan(export, gex):
    """Workbench writes no row for a moment it culled entirely.

    Dropping such a sounding would leave the input with fewer soundings than
    the published model; keeping it with the missing moment NaN is the honest
    representation.
    """
    fl = export.flightlines
    seg = fl["segment"].to_numpy()
    first_lm = np.flatnonzero(seg == 1)[0]
    keep = np.ones(len(fl), dtype=bool)
    keep[first_lm] = False
    trimmed = libaarhusxyz.XYZ({"flightlines": fl[keep].reset_index(drop=True),
                               "layer_data": {k: v[keep].reset_index(drop=True)
                                              for k, v in export.layer_data.items()},
                               "model_info": dict(export.model_info)})
    out = split_workbench_moments(trimmed, gex)
    assert len(out.flightlines) == 40
    assert out.model_info["moment split counts"]["hm_only"] == 1
    lm_rows_all_nan = out.layer_data["Gate_Ch01"].isna().all(axis=1)
    assert lm_rows_all_nan.sum() == 1
    assert (out.layer_data["InUse_Ch01"].to_numpy()[lm_rows_all_nan.to_numpy()] == 0).all()
    assert out.layer_data["Gate_Ch02"].notna().any(axis=1).all()


# ── Refusing what does not add up ────────────────────────────────────────────

def test_data_in_a_column_the_gex_gives_to_the_other_moment_is_refused(export, gex):
    """The data-derived map is a cross-check on the GEX map, not decoration."""
    maps, _ = workbench_gate_map(export.model_info["gate times"], gex)
    hm_only_col = next(j for j in maps[2] if j not in maps[1])
    seg1 = np.flatnonzero(export.flightlines["segment"].to_numpy() == 1)[0]
    export.layer_data["data"].iat[seg1, hm_only_col] = 1e-7      # LM data where only HM belongs
    with pytest.raises(WorkbenchSplitError, match="disagree"):
        split_workbench_moments(export, gex)


def test_negative_decays_are_refused(export, gex):
    with pytest.raises(WorkbenchSplitError, match="negative"):
        split_workbench_moments(export, gex, sign=-1.0)


def test_not_an_export_is_refused(gex):
    x = libaarhusxyz.XYZ()
    x.flightlines = pd.DataFrame({"x": [0.0], "y": [0.0], "line": [1], "fid": [1]})
    with pytest.raises(WorkbenchSplitError, match="not a Workbench"):
        split_workbench_moments(x, gex)


# ── Round trip through a file, the way an importer sees it ───────────────────

def test_dump_and_import_with_the_written_alc(split, tmp_path):
    out = tmp_path / "split.xyz"
    alc = tmp_path / "split.alc"
    split.dump(str(out), alcfile=str(alc))
    back = libaarhusxyz.XYZ(str(out), alcfile=str(alc))
    back.normalize(naming_standard="alc")
    assert back.layer_data["Gate_Ch01"].shape == (40, N_LM)
    assert back.layer_data["Gate_Ch02"].shape == (40, N_HM)
    np.testing.assert_allclose(back.layer_data["Gate_Ch02"].to_numpy(),
                               split.layer_data["Gate_Ch02"].to_numpy(), rtol=1e-6, equal_nan=True)
    assert "TxAltitude" in back.flightlines.columns
