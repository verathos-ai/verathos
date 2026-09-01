"""Deploy must measure against the live mesh, never a lingering corpse.

Manager mesh dicts iterate in insertion order, and a corpse of an earlier
generation (stopping/error, e.g. after a dead worker that can never ack
its teardown) always precedes the live mesh of the same model. Selecting
the first match made the measurement gate stop the corpse and then wait
forever for workers that were busy serving the healthy mesh nobody
stopped.
"""

from __future__ import annotations

from verallm.mesh.deploy import _find_model_mesh

MODEL = "qwen3.8-27b-uncensored-q4-k-m"


def _status(*meshes: tuple[str, str, str]) -> dict:
    return {
        "meshes": {
            key: {"model_id": model, "status": status}
            for key, model, status in meshes
        }
    }


def test_corpse_before_serving_mesh_is_shadowed_no_more():
    status = _status(
        ("m-corpse", MODEL, "stopping"),
        ("m-live", MODEL, "serving"),
    )
    key, mesh = _find_model_mesh(status, MODEL)
    assert key == "m-live"
    assert mesh["status"] == "serving"


def test_serving_before_corpse_still_picks_serving():
    status = _status(
        ("m-live", MODEL, "serving"),
        ("m-corpse", MODEL, "error"),
    )
    key, mesh = _find_model_mesh(status, MODEL)
    assert key == "m-live"


def test_mid_launch_beats_corpse():
    status = _status(
        ("m-corpse", MODEL, "stopping"),
        ("m-driving", MODEL, "driving"),
    )
    key, mesh = _find_model_mesh(status, MODEL)
    assert key == "m-driving"
    assert mesh["status"] == "driving"


def test_serving_beats_mid_launch():
    status = _status(
        ("m-driving", MODEL, "driving"),
        ("m-live", MODEL, "serving"),
    )
    key, _ = _find_model_mesh(status, MODEL)
    assert key == "m-live"


def test_corpse_only_is_still_returned_for_cleanup():
    status = _status(("m-corpse", MODEL, "stopping"))
    key, mesh = _find_model_mesh(status, MODEL)
    assert key == "m-corpse"
    assert mesh["status"] == "stopping"


def test_no_mesh_returns_empty():
    status = _status(("m-other", "another-model", "serving"))
    key, mesh = _find_model_mesh(status, MODEL)
    assert key == ""
    assert mesh == {}


def test_other_models_never_match():
    status = _status(
        ("m-other", "another-model", "serving"),
        ("m-live", MODEL, "joining"),
    )
    key, _ = _find_model_mesh(status, MODEL)
    assert key == "m-live"


def test_corpse_never_shadows_fetching_mesh():
    status = _status(
        ("m-corpse", MODEL, "stopping"),
        ("m-fetching", MODEL, "fetching"),
    )
    key, mesh = _find_model_mesh(status, MODEL)
    assert key == "m-fetching"
    assert mesh["status"] == "fetching"


def test_serving_beats_fetching():
    status = _status(
        ("m-fetching", MODEL, "fetching"),
        ("m-live", MODEL, "serving"),
    )
    key, _ = _find_model_mesh(status, MODEL)
    assert key == "m-live"
