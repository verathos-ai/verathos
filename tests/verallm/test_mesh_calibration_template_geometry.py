from types import SimpleNamespace

import pytest

from verallm.mesh.worker import slot_template_model_layers


def test_calibration_uses_local_model_geometry():
    context = {"layer_start": 33, "layer_end": 64}
    assert slot_template_model_layers(context, SimpleNamespace(total_layers=64)) == 64
    assert "model_total_layers" not in context


@pytest.mark.parametrize("supplied", [None, 0, 63])
def test_signed_geometry_never_falls_back(supplied):
    context = {"proof_receipt_format": "opaque_stage_v2"}
    if supplied is not None:
        context["model_total_layers"] = supplied
    with pytest.raises(RuntimeError):
        slot_template_model_layers(context, SimpleNamespace(total_layers=64))


def test_signed_geometry_matches_local_spec():
    assert slot_template_model_layers(
        {"proof_receipt_format": "opaque_stage_v2", "model_total_layers": 64},
        SimpleNamespace(total_layers=64),
    ) == 64


def test_calibration_does_not_infer_total_from_stage_end():
    with pytest.raises(RuntimeError):
        slot_template_model_layers({"layer_end": 33}, None)


def test_conflicting_calibration_geometry_rejected():
    with pytest.raises(RuntimeError):
        slot_template_model_layers({"model_total_layers": 33}, SimpleNamespace(total_layers=64))
