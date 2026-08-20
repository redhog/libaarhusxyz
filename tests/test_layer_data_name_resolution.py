"""layer_data_*_name() must return the key it matched, not the stub it matched with.

The three methods find the real key by matching a naming-convention stub against
layer_data, then throw the match away and return the stub. That only works where
the key happens to equal the stub, which is true of the alc naming standard
('Gate_Ch01') and false of the libaarhusxyz one ('dbdt_ch1gt'). The return value
is dereferenced as a key in YmerFlow's AEM importer, so on libaarhusxyz-named
data it raises KeyError.

Matching was also unanchored, so channel 1's stub 'dbdt_ch1' matched channel
10's 'dbdt_ch10gt'.

Regression tests for YmerFlow/libaarhusxyz#5.
"""

import os.path

import pytest

from libaarhusxyz import normalizer, xyzparser
from libaarhusxyz.xyz import XYZ


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# One Aarhus Workbench export carrying gate, std and inuse data for two
# channels, used to check the stub set against real delivered names.
RAW_EXPORT = os.path.join(
    DATA_DIR, "aarhus_workbench.6.7.0.0", "RAW_export_example_raw_data_export.xyz")


def make_xyz(*keys):
    """An XYZ whose layer_data holds `keys` and nothing else.

    The values are irrelevant here -- every method under test looks only at the
    keys -- so they are left as None to keep the intent obvious.
    """
    return XYZ({"flightlines": None,
                "model_info": {},
                "layer_data": {key: None for key in keys}})


# ── The reported failure: the stub is returned instead of the key ────────────

def test_data_name_returns_suffixed_key():
    xyz = make_xyz("dbdt_ch1gt")
    assert xyz.layer_data_data_name(1) == "dbdt_ch1gt"


def test_std_name_returns_suffixed_key():
    xyz = make_xyz("dbdt_std_ch1gt")
    assert xyz.layer_data_std_name(1) == "dbdt_std_ch1gt"


def test_inuse_name_returns_suffixed_key():
    xyz = make_xyz("dbdt_inuse_ch1gt")
    assert xyz.layer_data_inuse_name(1) == "dbdt_inuse_ch1gt"


def test_returned_name_indexes_layer_data():
    """The importer does `xyz.layer_data[xyz.layer_data_data_name(ch)]`.

    This is the assertion that actually fails before the fix, and the reason
    the defect is not cosmetic.
    """
    xyz = make_xyz("dbdt_ch1gt", "dbdt_std_ch1gt", "dbdt_inuse_ch1gt")
    for name in (xyz.layer_data_data_name(1),
                 xyz.layer_data_std_name(1),
                 xyz.layer_data_inuse_name(1)):
        assert name in xyz.layer_data


# ── The alc convention, where key and stub coincide ──────────────────────────

def test_key_equal_to_stub_still_resolves():
    xyz = make_xyz("Gate_Ch01", "STD_Ch01", "InUse_Ch01")
    assert xyz.layer_data_data_name(1) == "Gate_Ch01"
    assert xyz.layer_data_std_name(1) == "STD_Ch01"
    assert xyz.layer_data_inuse_name(1) == "InUse_Ch01"


def test_second_channel_resolves_under_both_conventions():
    """Channel 2 is zero-padded in the alc stub and bare in the libaarhusxyz one."""
    assert make_xyz("dbdt_ch2gt").layer_data_data_name(2) == "dbdt_ch2gt"
    assert make_xyz("Gate_Ch02").layer_data_data_name(2) == "Gate_Ch02"


# ── The ch1 / ch10 collision ─────────────────────────────────────────────────

def test_channel_1_does_not_match_channel_10():
    """'dbdt_ch1' is a prefix of 'dbdt_ch10gt', so unanchored matching collides."""
    xyz = make_xyz("dbdt_ch10gt")
    assert xyz.layer_data_data_name(1) is None
    assert xyz.layer_data_data_name(10) == "dbdt_ch10gt"


def test_adjacent_channels_resolve_independently():
    xyz = make_xyz("dbdt_ch1gt", "dbdt_ch10gt", "dbdt_ch11gt")
    assert xyz.layer_data_data_name(1) == "dbdt_ch1gt"
    assert xyz.layer_data_data_name(10) == "dbdt_ch10gt"
    assert xyz.layer_data_data_name(11) == "dbdt_ch11gt"


def test_collision_applies_to_std_and_inuse_too():
    assert make_xyz("dbdt_std_ch10gt").layer_data_std_name(1) is None
    assert make_xyz("dbdt_inuse_ch10gt").layer_data_inuse_name(1) is None


# ── Absence, which callers use as a presence test ────────────────────────────

def test_absence_returns_none():
    """The importer branches on `if xyz.layer_data_inuse_name(channel) is None`."""
    xyz = make_xyz("dbdt_ch1gt", "dbdt_std_ch1gt")
    assert xyz.layer_data_inuse_name(1) is None


def test_empty_layer_data_returns_none():
    xyz = make_xyz()
    assert xyz.layer_data_data_name(1) is None
    assert xyz.layer_data_std_name(1) is None
    assert xyz.layer_data_inuse_name(1) is None


def test_absent_channel_returns_none():
    xyz = make_xyz("dbdt_ch1gt")
    assert xyz.layer_data_data_name(3) is None


def test_data_stub_does_not_match_std_or_inuse_keys():
    xyz = make_xyz("dbdt_std_ch1gt", "dbdt_inuse_ch1gt")
    assert xyz.layer_data_data_name(1) is None


# ── Ambiguous input ──────────────────────────────────────────────────────────

def test_multiple_matches_raise():
    """Two keys matching one stub have no single right answer.

    Returning either one silently would hand the caller data from a channel
    variant it never asked for.
    """
    xyz = make_xyz("Gate_Ch01", "Gate_Ch01_lm")
    with pytest.raises(ValueError):
        xyz.layer_data_data_name(1)


def test_ambiguity_message_names_the_candidates():
    xyz = make_xyz("Gate_Ch01", "Gate_Ch01_lm")
    with pytest.raises(ValueError, match="Gate_Ch01_lm"):
        xyz.layer_data_data_name(1)


def test_ambiguity_in_a_later_stub_raises():
    """The alc stub is tried second, so its ambiguity must not be skipped over."""
    xyz = make_xyz("STD_Ch01", "STD_Ch01_hm")
    with pytest.raises(ValueError):
        xyz.layer_data_std_name(1)


def test_matching_stub_wins_before_a_later_ambiguous_one():
    """Stubs are alternatives in preference order, so the first match ends the search."""
    xyz = make_xyz("dbdt_ch1gt", "Gate_Ch01", "Gate_Ch01_lm")
    assert xyz.layer_data_data_name(1) == "dbdt_ch1gt"


# ── Against real delivered names ─────────────────────────────────────────────

@pytest.mark.parametrize("naming_standard", ["libaarhusxyz", "alc"])
def test_real_export_names_resolve_to_real_keys(naming_standard):
    """The stubs have to cover both naming standards on a real export.

    normalize_naming is used rather than the full normalize, which needs
    projnames and a projection this file does not carry.
    """
    xyz = XYZ(xyzparser.parse(RAW_EXPORT))
    normalizer.normalize_naming(xyz, naming_standard)

    for channel in (1, 2):
        for name in (xyz.layer_data_data_name(channel),
                     xyz.layer_data_std_name(channel),
                     xyz.layer_data_inuse_name(channel)):
            assert name is not None
            assert name in xyz.layer_data


def test_real_export_libaarhusxyz_keys_are_suffixed():
    """Guards the premise of this whole test module.

    If the default naming standard ever stopped suffixing these keys, the
    suffix tests above would still pass while testing nothing.
    """
    xyz = XYZ(xyzparser.parse(RAW_EXPORT))
    normalizer.normalize_naming(xyz, "libaarhusxyz")

    assert xyz.layer_data_data_name(1) == "dbdt_ch1gt"
    assert xyz.layer_data_std_name(1) == "dbdt_std_ch1gt"
    assert xyz.layer_data_inuse_name(1) == "dbdt_inuse_ch1gt"
