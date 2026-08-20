"""Compatibility tests for the inactive deferred mesh-audit helper.

The authenticated inline canary path is covered by
``test_mesh_canary_client.py``.  Deferred proof is retained only as an
explicitly inactive inspection surface.
"""

from __future__ import annotations

import json

import pytest

import neurons.mesh_verify as mv


class TestResolveDeferredAudit:
    def test_fetch_failure_is_hard_fail(self):
        def post(url, body, timeout):
            raise ConnectionError("gone")

        ok, reason = mv.resolve_mesh_deferred_audit(
            "https://coord:9338", {"receipt": {}}, {"model": "m"}, "ab" * 32,
            post_fn=post, verify_bundle_fn=lambda *a, **k: True,
        )
        assert ok is False and "fetch failed" in reason

    def test_bundle_mismatch_is_hard_fail(self):
        def post(url, body, timeout):
            assert url.endswith("/v1/mesh/proof/deferred-audit")
            assert body["deferred_randomness"] == "ab" * 32
            return {"version": 1}

        def verify(bundle, artifact, openai_request, **kw):
            raise RuntimeError("audit receipt hash mismatch")

        ok, reason = mv.resolve_mesh_deferred_audit(
            "https://coord:9338", {"receipt": {}}, {"model": "m"}, "ab" * 32,
            post_fn=post, verify_bundle_fn=verify,
        )
        assert ok is False and "verification error" in reason

    def test_verified_bundle_passes(self):
        ok, reason = mv.resolve_mesh_deferred_audit(
            "https://coord:9338", {"receipt": {}}, {"model": "m"}, "ab" * 32,
            post_fn=lambda u, b, timeout: {"version": 1},
            verify_bundle_fn=lambda *a, **k: True,
        )
        assert ok is True and reason == ""


class TestSnapshotRefusalClassification:
    """A pinned-snapshot refusal must never be retried as an outage."""

    @staticmethod
    def _http_error(status: int, body: dict) -> "mv.urllib.error.HTTPError":
        import io

        return mv.urllib.error.HTTPError(
            "https://coord:9338/v1/mesh/inference",
            status,
            "refused",
            {},
            io.BytesIO(json.dumps(body).encode("utf-8")),
        )

    def _transport_error(self, monkeypatch, status: int, body: dict):
        error = self._http_error(status, body)

        def fake_urlopen(request, timeout=None):
            raise error

        monkeypatch.setattr(mv.urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(mv.MeshCanaryTransportError) as excinfo:
            mv._default_mesh_canary_transport(
                "https://coord:9338/v1/mesh/inference", b"{}", {}, 5.0,
            )
        return excinfo.value

    def test_snapshot_mismatch_is_not_retryable(self, monkeypatch):
        error = self._transport_error(
            monkeypatch,
            409,
            {
                "error": "request verification snapshot is not served here",
                "error_code": mv.MESH_SNAPSHOT_MISMATCH_ERROR_CODE,
                "retryable": False,
            },
        )

        assert error.error_code == mv.MESH_SNAPSHOT_MISMATCH_ERROR_CODE
        assert error.retryable is False

    def test_snapshot_mismatch_dressed_up_as_busy_is_still_not_retryable(
        self,
        monkeypatch,
    ):
        """503 normally means busy, and busy traffic is forgiven at epoch close.

        A coordinator that wants its rotation forgiven would send the refusal
        with that status, so the error code has to win over the status.
        """

        error = self._transport_error(
            monkeypatch,
            503,
            {
                "error": "please retry",
                "error_code": mv.MESH_SNAPSHOT_MISMATCH_ERROR_CODE,
                "retryable": True,
            },
        )

        assert error.retryable is False

    def test_a_genuine_503_stays_retryable(self, monkeypatch):
        error = self._transport_error(monkeypatch, 503, {"error": "busy"})

        assert error.error_code == ""
        assert error.retryable is True



class TestCanariesAreShapedLikeOrganicTraffic:
    """A canary must not be separable from real traffic by transport shape.

    Before this, canaries posted to /v1/mesh/inference with stream=false while
    users streamed to /v1/chat/completions. A coordinator told them apart on
    the URL alone, before parsing a body, and could serve the canary-shaped
    endpoint honestly while cheating the endpoint carrying real traffic.
    """

    def test_the_canary_uses_the_streaming_chat_route(self):
        assert mv.MESH_CHAT_COMPLETIONS_PATH == "/v1/chat/completions"
        assert mv._coordinator_chat_url("https://coord:9338").endswith(
            "/v1/chat/completions"
        )


class TestMeshSseArtifactReader:
    """The terminal event is the only part of the stream the validator trusts."""

    @staticmethod
    def _stream(*lines: bytes):
        import io

        return io.BytesIO(b"".join(lines))

    @staticmethod
    def _done_event(**overrides) -> bytes:
        payload = {
            "event": "done",
            "response": {"choices": [{"message": {"content": "hi"}}]},
            "verathos_mesh": {"receipt": {"receipt_hash": "ab" * 32}},
        }
        payload.update(overrides)
        return b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n"

    def test_the_terminal_artifact_is_rebuilt_from_the_done_event(self):
        raw = mv._read_mesh_sse_artifact(
            self._stream(
                b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
                self._done_event(),
                b"data: [DONE]\n\n",
            )
        )

        artifact = json.loads(raw)
        assert artifact["receipt"]["receipt_hash"] == "ab" * 32
        assert artifact["response"]["choices"][0]["message"]["content"] == "hi"

    def test_a_stream_with_no_terminal_event_fails(self):
        with pytest.raises(mv.MeshCanaryTransportError, match="without a terminal"):
            mv._read_mesh_sse_artifact(
                self._stream(b'data: {"choices":[]}\n\n', b"data: [DONE]\n\n")
            )

    def test_two_terminal_events_are_rejected(self):
        """Otherwise a coordinator picks whichever artifact verifies."""

        with pytest.raises(mv.MeshCanaryTransportError, match="multiple terminal"):
            mv._read_mesh_sse_artifact(
                self._stream(self._done_event(), self._done_event())
            )

    def test_a_terminal_event_without_its_artifact_is_rejected(self):
        with pytest.raises(mv.MeshCanaryTransportError, match="missing its artifact"):
            mv._read_mesh_sse_artifact(
                self._stream(self._done_event(verathos_mesh="nope"))
            )

    def test_a_malformed_event_is_a_protocol_failure(self):
        with pytest.raises(mv.MeshCanaryTransportError, match="malformed event"):
            mv._read_mesh_sse_artifact(self._stream(b"data: {not json}\n\n"))

    def test_none_of_these_are_retryable(self):
        """A coordinator must not earn a retry by streaming garbage."""

        with pytest.raises(mv.MeshCanaryTransportError) as excinfo:
            mv._read_mesh_sse_artifact(self._stream(b"data: [DONE]\n\n"))

        assert excinfo.value.retryable is False
        assert excinfo.value.streamed_content is True

    def test_an_in_stream_slots_busy_verdict_is_a_retryable_503(self):
        """The mid-stream busy verdict (audit window did not drain within
        the validator lane's wait budget) must land in the same busy-skip
        machinery as the pre-stream slots_busy 503 — not count as an
        inference failure or probation signal."""
        busy = {
            "event": "error",
            "error": "all 4 generation slots are busy; retry or fail over",
            "type": "slots_busy",
            "retryable": True,
        }
        with pytest.raises(mv.MeshCanaryTransportError) as excinfo:
            mv._read_mesh_sse_artifact(
                self._stream(
                    b"data: " + json.dumps(busy).encode() + b"\n\n",
                    b"data: [DONE]\n\n",
                )
            )

        assert excinfo.value.status_code == 503
        assert excinfo.value.retryable is True
        assert excinfo.value.streamed_content is False
        assert excinfo.value.error_code == "slots_busy"

    def test_a_generic_in_stream_error_is_surfaced_not_swallowed(self):
        event = {"event": "error", "error": "mesh inference failed"}
        with pytest.raises(
            mv.MeshCanaryTransportError, match="mesh inference failed"
        ) as excinfo:
            mv._read_mesh_sse_artifact(
                self._stream(b"data: " + json.dumps(event).encode() + b"\n\n")
            )

        assert excinfo.value.retryable is False
        assert excinfo.value.streamed_content is True

    def test_snapshot_mismatch_error_code_survives_the_stream(self):
        """The in-stream snapshot-mismatch refusal must keep its error code
        so consumers can keep it away from retry and busy forgiveness."""
        event = {
            "event": "error",
            "error": "request verification snapshot is not served here",
            "error_code": "verification_snapshot_mismatch",
            "retryable": False,
        }
        with pytest.raises(mv.MeshCanaryTransportError) as excinfo:
            mv._read_mesh_sse_artifact(
                self._stream(b"data: " + json.dumps(event).encode() + b"\n\n")
            )

        assert excinfo.value.retryable is False
        assert excinfo.value.error_code == "verification_snapshot_mismatch"


class TestCanaryTimeToFirstToken:
    """Streaming canaries can measure TTFT; the old ones could not."""

    def test_the_first_content_chunk_is_timed(self):
        import io

        slot: list[float] = []
        stream = io.BytesIO(
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":" there"}}]}\n\n'
            b'data: {"event":"done","response":{"choices":[]},'
            b'"verathos_mesh":{"receipt":{}}}\n\n'
        )

        mv._read_mesh_sse_artifact(stream, first_delta=slot)

        # Only the first chunk carrying visible content is timed, and a
        # role-only opening chunk is not visible content.
        assert len(slot) == 1

    def test_a_stream_with_no_visible_content_reports_nothing(self):
        import io

        slot: list[float] = []
        stream = io.BytesIO(
            b'data: {"event":"done","response":{"choices":[]},'
            b'"verathos_mesh":{"receipt":{}}}\n\n'
        )

        mv._read_mesh_sse_artifact(stream, first_delta=slot)

        assert slot == []

    def test_the_timing_slot_is_per_thread(self):
        """Canaries run on a thread pool; timings must not cross miners."""

        import threading

        seen: list[int] = []

        def worker():
            slot = mv._canary_first_delta()
            slot.append(1.0)
            seen.append(len(slot))

        mv._canary_first_delta().clear()
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert seen == [1, 1, 1, 1]
        assert mv._canary_first_delta() == []
