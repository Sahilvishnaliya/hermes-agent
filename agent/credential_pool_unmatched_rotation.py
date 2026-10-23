"""Rotation when a failure's identity matched no pool entry (#70401, #139226)."""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:  # the pool imports this module, so these would be circular at runtime
    from agent.credential_pool import PooledCredential

# The pool logs under its own module name; a mixin on a sibling module keeps its lines in the same
# place for anyone grepping the log.
logger = logging.getLogger("agent.credential_pool")


class CredentialPoolUnmatchedRotationMixin:
    """The unmatched-identity branch of ``mark_exhausted_and_rotate``."""

    def _unmatched_attribution(self) -> Optional[PooledCredential]:
        """The entry to blame when the failure's identity matched no entry, or None to rotate blind.

        An OAuth access token is refreshed IN PLACE, so the 401's ``api_key_hint`` is the previous
        token and can never equal the entry's current one: every retry lands in the unmatched
        branch, and rotating without marking anything oscillates between the revoked credential
        and a rate-limited one (401 -> 429 -> 401, #139226, #122601). The entry the pool itself
        last handed out IS the one that failed, so blame it: a revoked 401 is a terminal reason,
        the entry goes DEAD, and the rotation converges.

        An api-key pool keeps the no-mark behaviour: its hint is a real key, so a hint matching
        nothing means the failure came from outside the pool and benching the current key would
        quarantine an innocent one for the full TTL.
        """
        from agent.credential_pool import AUTH_TYPE_OAUTH  # lazy: the pool imports this module

        current = self._current_unlocked()
        if current is not None and current.auth_type == AUTH_TYPE_OAUTH:
            return current
        return None
    def _rotate_unmatched(self) -> Optional[PooledCredential]:
        """Rotate without marking anything when the failed identity matches no entry.

        Falling through to current()/_select_unlocked() would bench an innocent healthy key for
        the full TTL. But this must be BOUNDED (#70401): with no attribution, nothing is marked,
        and the caller would retry the same dead token forever (~6/sec, starving the event loop).
        Cap consecutive no-mark rotations at one lap of the available entries, then surface the
        error; no cooldown is written.
        """
        self._unmatched_rotation_streak += 1
        available_count = len(self._available_entries()[0])
        if self._unmatched_rotation_streak > max(available_count, 1):
            logger.warning(
                "credential pool: failed credential identity matched no "
                "%s entry for %d consecutive rotations (pool size %d) - "
                "surfacing the error instead of rotating again",
                self.provider, self._unmatched_rotation_streak, available_count,
            )
            self._unmatched_rotation_streak = 0
            self._current_id = None
            return None
        logger.info(
            "credential pool: failed credential identity matched no %s "
            "entry; rotating without marking any credential exhausted",
            self.provider,
        )
        self._current_id = None
        next_entry, _pending = self._select_unlocked(refresh=False)
        if next_entry is not None and len(self._available_entries()[0]) == 1:
            # A single-entry pool cannot rotate: returning its only entry would
            # report a recovery without changing the credential, and the
            # caller retries the same 401 indefinitely.
            self._unmatched_rotation_streak = 0
            self._current_id = None
            return None
        return next_entry
