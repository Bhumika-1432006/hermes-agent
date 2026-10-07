"""pre-API gate must not keep _provider_overflow_recovery_pending armed forever when the
context engine reports threshold_tokens=0 (e.g. a plugin that drops update_model).

Previously `_preflight_threshold <= 0` was treated as "always over the threshold", so
`_provider_overflow_preflight` stayed True on every request and the turn failed closed
well below the real context window.

The fix falls back to `compressor.context_length` when `threshold_tokens` is 0.

Fixes: https://github.com/NousResearch/hermes-agent/issues/134321
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call


def _make_compressor(threshold_tokens: int, context_length: int) -> SimpleNamespace:
    return SimpleNamespace(
        threshold_tokens=threshold_tokens,
        context_length=context_length,
        get_active_compression_failure_cooldown=lambda: None,
        should_defer_preflight_to_real_usage=lambda _t: False,
        should_compress=lambda _t: False,
    )


def _run_gate(threshold_tokens: int, context_length: int, request_pressure_tokens: int,
              overflow_pending: bool) -> bool:
    """Return the provider_overflow_preflight value that was passed to run_preflight_compression."""
    from agent.turn_preflight_gate import run_preflight_gate

    agent = MagicMock()
    agent.context_compressor = _make_compressor(threshold_tokens, context_length)
    agent.compression_enabled = True
    agent.iteration_budget = MagicMock()

    captured = {}

    def _fake_run_preflight_compression(ag, v, *, provider_overflow_preflight, **kw):
        captured["pof"] = provider_overflow_preflight
        v.action = "fallthrough"
        v.result = None
        return v

    common_kw = dict(
        request_pressure_tokens=request_pressure_tokens,
        _moa_prepared_request=None,
        pending_moa_prepared_request=None,
        messages=[MagicMock(), MagicMock()],
        system_message=None,
        user_message=None,
        active_system_prompt=None,
        conversation_history=[],
        api_call_count=0,
        compression_attempts=0,
        max_compression_attempts=3,
        effective_task_id="t1",
        final_response=None,
        failed=False,
        _turn_exit_reason=None,
        _compression_timeout_exhausted=False,
        _preflight_compression_blocked=False,
        _provider_overflow_recovery_pending=overflow_pending,
        _last_preflight_pressure=None,
    )

    with patch("agent.turn_preflight_gate.run_preflight_compression",
               side_effect=_fake_run_preflight_compression), \
         patch("agent.conversation_loop._ollama_context_limit_error", return_value=False):
        run_preflight_gate(agent, **common_kw)

    return captured["pof"]


# ---------------------------------------------------------------------------
# Zero threshold → fall back to context_length
# ---------------------------------------------------------------------------

def test_zero_threshold_fits_window_clears_pending_flag():
    """Request well within context_length: overflow flag must not stay armed."""
    pof = _run_gate(threshold_tokens=0, context_length=128_000,
                    request_pressure_tokens=10_000, overflow_pending=True)
    assert pof is False


def test_zero_threshold_exceeds_window_keeps_pending_flag():
    """Request over context_length: overflow flag stays armed (real overflow)."""
    pof = _run_gate(threshold_tokens=0, context_length=10_000,
                    request_pressure_tokens=50_000, overflow_pending=True)
    assert pof is True


# ---------------------------------------------------------------------------
# Normal threshold must not regress
# ---------------------------------------------------------------------------

def test_normal_threshold_fits_clears_pending_flag():
    pof = _run_gate(threshold_tokens=80_000, context_length=128_000,
                    request_pressure_tokens=10_000, overflow_pending=True)
    assert pof is False


def test_normal_threshold_exceeds_keeps_pending_flag():
    pof = _run_gate(threshold_tokens=80_000, context_length=128_000,
                    request_pressure_tokens=90_000, overflow_pending=True)
    assert pof is True


# ---------------------------------------------------------------------------
# Both zero — safe fail-closed
# ---------------------------------------------------------------------------

def test_both_zero_keeps_pending_flag_as_safe_fallback():
    """When neither threshold nor context_length is known, fail closed."""
    pof = _run_gate(threshold_tokens=0, context_length=0,
                    request_pressure_tokens=10_000, overflow_pending=True)
    assert pof is True
