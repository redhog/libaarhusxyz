"""GEX must follow the attribute-lookup protocol.

A missing attribute has to raise AttributeError, not KeyError. Everything in the
standard library that feature-detects by attribute — copy, deepcopy, pickle —
probes for a dunder and reads AttributeError as "not provided". A KeyError
escapes the probe and aborts the operation.

Regression tests for the four failures reported in YmerFlow/libaarhusxyz#3, and
for the recursion that a naive fix introduces.
"""

import copy
import pickle

import pytest

from libaarhusxyz.gex import GEX


@pytest.fixture
def minimal():
    """A GEX with nothing optional in it.

    Deliberately sparse: the failures only appear when a probed name is absent,
    which is exactly the case a partially-populated GEX presents.
    """
    return GEX({"General": {}})


# ── The reported failures ────────────────────────────────────────────────────

def test_deepcopy(minimal):
    assert copy.deepcopy(minimal).gex_dict == minimal.gex_dict


def test_copy(minimal):
    assert copy.copy(minimal).gex_dict == minimal.gex_dict


def test_pickle_roundtrip(minimal):
    assert pickle.loads(pickle.dumps(minimal)).gex_dict == minimal.gex_dict


def test_repr_does_not_raise(minimal):
    """A __repr__ is called by debuggers and loggers, so it must never raise.

    Raising here turns an unrelated error into a confusing one — pytest reports
    it as "raised in repr()" and hides the original failure.
    """
    assert isinstance(repr(minimal), str)


# ── The guards that make those pass ──────────────────────────────────────────

def test_missing_attribute_raises_attributeerror(minimal):
    with pytest.raises(AttributeError):
        minimal.no_such_field


def test_dunder_probe_raises_attributeerror(minimal):
    """Dunders are feature detection and never real GEX fields."""
    for name in ("__deepcopy__", "__getstate__", "__setstate__", "__reduce_ex__"):
        with pytest.raises(AttributeError):
            GEX.__getattr__(minimal, name)


def test_gex_dict_probe_does_not_recurse():
    """The backing attribute must be guarded by name.

    copy and pickle build the instance through __new__ with an empty __dict__
    and populate it afterwards, so until then every attribute access lands in
    __getattr__ — including ``gex_dict`` itself. Without a guard, looking it up
    evaluates ``self.gex_dict``, which lands back in __getattr__, and recurses
    until the stack is exhausted.

    This is the failure a naive "convert KeyError to AttributeError" fix
    introduces: it removes the KeyError that was accidentally aborting the
    lookup before the recursion could start.
    """
    bare = GEX.__new__(GEX)          # no __init__, so __dict__ is empty
    with pytest.raises(AttributeError):
        bare.gex_dict


# ── What must keep working ───────────────────────────────────────────────────

def test_real_fields_still_resolve():
    gex = GEX({"General": {"TxLoopArea": 342.0}, "header": "test system"})
    assert gex.General["TxLoopArea"] == 342.0
    assert gex.header == "test system"


def test_deepcopy_is_independent():
    gex = GEX({"General": {"TxLoopArea": 342.0}})
    clone = copy.deepcopy(gex)
    clone.gex_dict["General"]["TxLoopArea"] = 1.0
    assert gex.gex_dict["General"]["TxLoopArea"] == 342.0
