# SPDX-License-Identifier: Apache-2.0
"""
Session and SessionManager for tracking per-request state
in the multiprocess cache server.
"""

# Standard
from dataclasses import dataclass, field
from typing import Any
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.token_hasher import TokenHasher

logger = init_logger(__name__)


@dataclass
class Session:
    """Tracks accumulated token IDs and computed chunk hashes for a request.

    Thread-safe: all public methods are protected by an internal lock
    to allow concurrent access from multiple TP worker threads.
    """

    request_id: str
    hasher: TokenHasher
    token_ids: list[int] = field(default_factory=list)
    chunk_hashes: list = field(default_factory=list)
    last_prefix_hash: Any = None
    num_chunks_processed: int = 0
    created_at: float = field(default_factory=time.time)
    total_tokens: int = 0
    retrieved_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set_tokens(self, full_token_ids: list[int]) -> None:
        """Update the token sequence (idempotent, replaces not extends).

        Args:
            full_token_ids: Complete token sequence.
        """
        with self._lock:
            self.token_ids = full_token_ids

    def get_hashes(self, start: int, end: int) -> list:
        """Compute and return chunk hashes for the [start, end) token range.

        Internally computes rolling hashes up to end_chunk, skipping
        already-computed chunks.

        Args:
            start: Start token index.
            end: End token index.

        Returns:
            List of hash values for chunks in [start_chunk, end_chunk).
        """
        chunk_size = self.hasher.chunk_size
        assert start % chunk_size == 0, (
            f"start ({start}) must be a multiple of chunk_size ({chunk_size})"
        )
        assert end % chunk_size == 0, (
            f"end ({end}) must be a multiple of chunk_size ({chunk_size})"
        )
        start_chunk = start // chunk_size
        end_chunk = end // chunk_size

        with self._lock:
            self._compute_hash(end_chunk)
            return self.chunk_hashes[start_chunk:end_chunk]

    def _compute_hash(self, end_chunk: int) -> None:
        """Compute rolling hashes up to end_chunk.

        Uses cached state to skip already-computed chunks.

        Args:
            end_chunk: Compute hashes up to (but not including) this chunk.
        """
        chunk_size = self.hasher.chunk_size

        while self.num_chunks_processed < end_chunk:
            cs = self.num_chunks_processed * chunk_size
            ce = cs + chunk_size
            chunk = self.token_ids[cs:ce]

            prefix = (
                self.last_prefix_hash
                if self.last_prefix_hash is not None
                else self.hasher.none_hash
            )
            h = self.hasher.hash_tokens(chunk, prefix)
            self.last_prefix_hash = h
            self.chunk_hashes.append(h)
            self.num_chunks_processed += 1


class SessionManager:
    """Thread-safe manager for per-request sessions."""

    DEFAULT_SESSION_TTL = 600  # 10 minutes

    def __init__(self, hasher: TokenHasher, ttl: float = DEFAULT_SESSION_TTL):
        self._hasher = hasher
        self._ttl = ttl
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

        # Cumulative stats accumulated when sessions end
        self._stats_lock = threading.Lock()
        self._total_requests: int = 0
        self._total_tokens: int = 0
        self._total_retrieved_tokens: int = 0

    def get_or_create(self, request_id: str) -> Session:
        """Get existing session or create a new one.

        Args:
            request_id: Unique request identifier.

        Returns:
            The Session for this request_id.
        """
        with self._lock:
            if request_id not in self._sessions:
                self._sessions[request_id] = Session(
                    request_id=request_id, hasher=self._hasher
                )
                logger.debug("Created session for request_id=%s", request_id)
            return self._sessions[request_id]

    def remove(self, request_id: str) -> None:
        """Remove a session and accumulate its stats.

        Args:
            request_id: Unique request identifier.
        """
        with self._lock:
            session = self._sessions.pop(request_id, None)
        if session is not None:
            with self._stats_lock:
                self._total_requests += 1
                self._total_tokens += session.total_tokens
                self._total_retrieved_tokens += session.retrieved_tokens
            logger.debug(
                "Removed session for request_id=%s",
                request_id,
            )

    def cleanup_expired(self) -> int:
        """Remove sessions that have exceeded their TTL.

        Returns:
            Number of sessions removed.
        """
        now = time.time()
        with self._lock:
            expired = [
                rid
                for rid, s in self._sessions.items()
                if now - s.created_at > self._ttl
            ]

        for rid in expired:
            self.remove(rid)

        if expired:
            logger.info("Cleaned up %d expired sessions", len(expired))
        return len(expired)

    def active_count(self) -> int:
        """Return the number of active sessions.

        Returns:
            Number of currently tracked sessions.
        """
        with self._lock:
            return len(self._sessions)

    def report_hit_stats(self) -> dict[str, int | float]:
        """Return cumulative hit statistics.

        Returns:
            Dict with total_requests, total_tokens,
            total_retrieved_tokens, and hit_rate.
        """
        with self._stats_lock:
            total_req = self._total_requests
            total_tok = self._total_tokens
            retrieved_tok = self._total_retrieved_tokens
        hit_rate = round(retrieved_tok / total_tok, 4) if total_tok > 0 else 0.0
        return {
            "total_requests": total_req,
            "total_tokens": total_tok,
            "total_retrieved_tokens": retrieved_tok,
            "hit_rate": hit_rate,
        }
