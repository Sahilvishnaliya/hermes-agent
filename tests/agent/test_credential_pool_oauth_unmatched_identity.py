"""#139226: a 401 whose identity matches no entry must not blind-rotate an OAuth pool.

An OAuth access token is refreshed in place, so the 401's ``api_key_hint`` is the
*previous* token and can never equal the entry's current one. The unmatched branch
therefore marked nothing and rotated blindly: the revoked entry stayed available, the
rotation came back to it, and the caller oscillated between ``401 revoked`` and ``429``
from the healthy-but-rate-limited entry until the lap bound tripped.

The entry the pool last handed out IS the one that failed, so an OAuth pool now
attributes the failure to it. A revoked 401 is a terminal reason, so the entry goes
DEAD and the rotation converges instead of coming back. An api-key pool keeps the old
behaviour - its hint is a real key, and a hint matching nothing means the failure came
from outside the pool (see ``test_credential_pool_unmatched_rotation_bound``).
"""
import json

import pytest


def _seed_oauth_pool(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "credential_pool": {"anthropic": [
            {"id": "cred-a", "label": "a", "auth_type": "oauth", "priority": 0,
             "source": "manual:hermes_pkce", "access_token": "sk-ant-oat-token-a"},
            {"id": "cred-b", "label": "b", "auth_type": "oauth", "priority": 1,
             "source": "manual:hermes_pkce", "access_token": "sk-ant-oat-token-b"},
        ]},
    }))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    from agent.credential_pool import load_pool

    return load_pool("anthropic")


class TestUnmatchedOauthIdentity:

    def test_a_revoked_401_is_attributed_to_the_entry_that_was_handed_out(
        self, tmp_path, monkeypatch
    ):
        pool = _seed_oauth_pool(tmp_path, monkeypatch)
        used = pool.select()
        assert used is not None

        # The hint is the PREVIOUS token (the pool refreshed it away); it matches no entry.
        nxt = pool.mark_exhausted_and_rotate(
            status_code=401,
            error_context={"reason": "OAuth access token has been revoked"},
            api_key_hint="sk-ant-oat-the-previous-token",
        )

        # The failed entry is benched, so the rotation converges to the other one.
        assert next(e for e in pool._entries if e.id == used.id).last_status in ("dead", "exhausted")
        assert nxt is not None and nxt.id != used.id
        assert pool._unmatched_rotation_streak == 0

    def test_the_pool_converges_instead_of_oscillating(self, tmp_path, monkeypatch):
        pool = _seed_oauth_pool(tmp_path, monkeypatch)
        assert pool.select() is not None

        seen = []
        for _ in range(4):
            nxt = pool.mark_exhausted_and_rotate(
                status_code=401,
                error_context={"reason": "OAuth access token has been revoked"},
                api_key_hint="sk-ant-oat-the-previous-token",
            )
            if nxt is None:
                break
            seen.append(nxt.id)

        # Each entry was benched once and the failure then surfaced: no entry stays available.
        assert seen == ["cred-b"], seen
        assert [e.id for e in pool._entries if e.last_status is None] == []

    def test_an_api_key_pool_still_marks_nothing_on_an_unmatched_hint(
        self, tmp_path, monkeypatch
    ):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "credential_pool": {"openrouter": [
                {"id": "cred-0", "label": "k0", "auth_type": "api_key", "priority": 0,
                 "source": "manual", "access_token": "key-a"},
                {"id": "cred-1", "label": "k1", "auth_type": "api_key", "priority": 1,
                 "source": "manual", "access_token": "key-b"},
            ]},
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        from agent.credential_pool import load_pool

        pool = load_pool("openrouter")
        assert pool.select() is not None
        assert pool.mark_exhausted_and_rotate(
            status_code=401, api_key_hint="a-key-that-is-not-in-the-pool") is not None
        assert all(e.last_status != "exhausted" for e in pool._entries)
