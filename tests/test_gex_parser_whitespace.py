"""An indented key must be reachable by its own name.

The parser strips trailing digits so that ``GateTimeLM01``, ``GateTimeLM02``, …
collapse into one ``GateTimeLM`` array. It did not strip leading whitespace, so
an indented line produced a key with a leading space that no lookup would match:
the value was parsed, stored, and unreachable.

GEX files in hand are not indented, so this is latent rather than observed — but
the format does not forbid it, and a silently unreachable calibration parameter
is the kind of failure that surfaces as wrong physics rather than an error.
"""

import io
import libaarhusxyz.gex


FLUSH = """[General]
NumberOfChannels=2
"""

INDENTED = """[General]
    NumberOfChannels=2
"""

MIXED = """[General]
NumberOfChannels=2
\tTxLoopArea=340.0
   GateTimeLM01=1.0 2.0 3.0
"""


def _parse(text):
    return libaarhusxyz.gex.parse_parameters(io.StringIO(text).readlines())


def test_flush_key_is_reachable():
    assert _parse(FLUSH)["NumberOfChannels"] == 2


def test_indented_key_is_reachable_by_its_own_name():
    assert _parse(INDENTED)["NumberOfChannels"] == 2


def test_indented_key_has_no_leading_whitespace():
    assert all(k == k.strip() for k in _parse(INDENTED))


def test_tab_indent_and_trailing_digit_stripping_coexist():
    parsed = _parse(MIXED)
    assert parsed["NumberOfChannels"] == 2
    assert parsed["TxLoopArea"] == 340.0
    assert "GateTimeLM" in parsed, "trailing digits should still be stripped"
    assert all(k == k.strip() for k in parsed)
