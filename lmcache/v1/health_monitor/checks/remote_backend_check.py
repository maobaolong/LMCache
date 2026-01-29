# SPDX-License-Identifier: Apache-2.0
"""
Health check for RemoteBackend.
"""

# Standard
from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Optional
import asyncio
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey
from lmcache.v1.health_monitor.base import HealthCheck
from lmcache.v1.health_monitor.constants import (
    CHECK_PUT_TIMEOUT_CONFIG_KEY,
    DEFAULT_CHECK_PUT_TIMEOUT,
    DEFAULT_FALLBACK_POLICY,
    DEFAULT_PING_TIMEOUT,
    DEFAULT_REMOTE_FAILED_THRESHOLD,
    DEFAULT_WAITING_TIME_FOR_RECOVERY,
    FALLBACK_POLICY_CONFIG_KEY,
    PING_GENERIC_ERROR_CODE,
    PING_TIMEOUT_CONFIG_KEY,
    PING_TIMEOUT_ERROR_CODE,
    REMOTE_FAILED_THRESHOLD_CONFIG_KEY,
    WAITING_TIME_FOR_RECOVERY_CONFIG_KEY,
    FallbackPolicy,
)
from lmcache.v1.storage_backend.connector import InstrumentedRemoteConnector
from lmcache.v1.storage_backend.connector.audit_connector import AuditConnector

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.manager import LMCacheManager
    from lmcache.v1.storage_backend.remote_backend import RemoteBackend

logger = init_logger(__name__)


class RemoteBackendHealthCheck(HealthCheck):
    """
    Health check for RemoteBackend by pinging the remote connector.

    This check verifies that the remote backend is reachable and responsive
    by sending periodic ping requests.

    Fallback Policies:
        - RECOMPUTE (default): Skip all cache operations when
          remote backend is unhealthy
        - LOCAL_CPU: Bypass remote backend and use local CPU with hot_cache enabled
    """

    def __init__(
        self,
        backend: "RemoteBackend",
    ):
        self.backend = backend
        # Get fallback policy from config
        fallback_policy_str = backend.config.get_extra_config_value(
            FALLBACK_POLICY_CONFIG_KEY, DEFAULT_FALLBACK_POLICY.value
        )
        # Convert string to FallbackPolicy enum
        if isinstance(fallback_policy_str, str):
            try:
                self._fallback_policy = FallbackPolicy(fallback_policy_str)
            except ValueError:
                logger.warning(
                    f"Invalid fallback_policy '{fallback_policy_str}' "
                    f"for {backend}, using default: "
                    f"{DEFAULT_FALLBACK_POLICY}"
                )
                self._fallback_policy = DEFAULT_FALLBACK_POLICY
        elif isinstance(fallback_policy_str, FallbackPolicy):
            self._fallback_policy = fallback_policy_str
        self.failure_time: Optional[float] = None
        self._stats_monitor = LMCStatsMonitor.GetOrCreate()
        self._backend_name: Optional[str] = None

    @classmethod
    def create(cls, manager: "LMCacheManager") -> List[HealthCheck]:
        """
        Create RemoteBackendHealthCheck instances from a LMCacheManager.

        This method finds all RemoteBackend instances in the storage manager
        and creates a health check for each one.

        Args:
            manager: The LMCacheManager instance

        Returns:
            List of RemoteBackendHealthCheck instances
        """
        # Import here to avoid circular imports
        # First Party
        from lmcache.v1.storage_backend.remote_backend import RemoteBackend

        instances: List[HealthCheck] = []

        # Get engine from manager
        engine = manager.lmcache_engine
        if engine is None or engine.storage_manager is None:
            return instances

        for backend_name, backend in engine.storage_manager.storage_backends.items():
            if isinstance(backend, RemoteBackend):
                check = cls(backend)
                check._backend_name = backend_name
                instances.append(check)
                logger.info(f"Created {check} for {backend_name}")

        return instances

    def name(self) -> str:
        return f"RemoteBackendHealthCheck({self.backend.remote_url})"

    @property
    def fallback_policy(self) -> FallbackPolicy:
        """Return the fallback policy for this health check."""
        return self._fallback_policy

    def get_bypass_backend_name(self) -> Optional[str]:
        """
        Return the backend name to bypass when this health check fails.

        Returns:
            Optional[str]: The backend name (e.g., "RemoteBackend")
        """
        return self._backend_name

    def _try_reinitialize_connection(self) -> bool:
        """
        Try to reinitialize the connection if connector is None.

        Returns:
            bool: True if connection was successfully initialized, False otherwise
        """
        if self.backend.connection is not None:
            return True

        logger.warning("Connector is None, re-initializing connection.")
        self.backend.init_connection()

        return self.backend.connection is not None

    def check(self) -> bool:
        """
        Perform a health check on remote backend, which includes the following checks:

        - get_blocking/batched_get_blocking, if failed count >= threshold,
        which means check failed, update failure_time and return False.
        If failure_time is not None, wait for more than`waiting_time_for_recovery`
        seconds before resuming the check.

        - ping, if connector supports ping, send a ping request to remote connector.

        Returns:
            bool: True if all checks succeeds, False otherwise
        """
        # Try to reinitialize connection if needed
        if not self._try_reinitialize_connection():
            return False

        # At this point, connector is guaranteed to be not None
        connector = self.backend.connection
        assert connector is not None

        if self.failure_time is not None:
            # Get waiting time for recovery from backend config
            waiting_time_for_recovery = self.backend.config.get_extra_config_value(
                WAITING_TIME_FOR_RECOVERY_CONFIG_KEY,
                DEFAULT_WAITING_TIME_FOR_RECOVERY,
            )
            if (
                time.time() - self.failure_time > waiting_time_for_recovery
                and self._put_and_get_check()
            ):
                # recover from remote failed
                logger.info(
                    "Failure time: %s, current time: %s, recover from remote failed",
                    self.failure_time,
                    time.time(),
                )
                # clear remote failed count
                self.backend.get_and_clear_interval_remote_failed_count()
                # clear failure time
                self.failure_time = None
            else:
                logger.info(
                    "Failure time: %s, current time: %s, "
                    "waiting time for recovery: %s, "
                    "still in remote failed recovery window",
                    self.failure_time,
                    time.time(),
                    waiting_time_for_recovery,
                )
                return False

        # Get remote failed threshold from backend config
        remote_failed_threshold = self.backend.config.get_extra_config_value(
            REMOTE_FAILED_THRESHOLD_CONFIG_KEY,
            DEFAULT_REMOTE_FAILED_THRESHOLD,
        )
        # Check remote failed
        remote_failed_count = self.backend.get_and_clear_interval_remote_failed_count()
        if remote_failed_count >= remote_failed_threshold:
            logger.warning(
                "Detected %s remote failed in interval, threshold: %s",
                remote_failed_count,
                remote_failed_threshold,
            )
            self.failure_time = time.time()
            return False

        # If connector doesn't support ping, assume it's healthy
        if not connector.support_ping():
            return True

        # Check ping
        # Get ping timeout from backend config
        ping_timeout = self.backend.config.get_extra_config_value(
            PING_TIMEOUT_CONFIG_KEY, DEFAULT_PING_TIMEOUT
        )
        try:
            start_time = time.perf_counter()
            future = asyncio.run_coroutine_threadsafe(
                connector.ping(), self.backend.loop
            )
            error_code = future.result(timeout=ping_timeout)
            latency = (time.perf_counter() - start_time) * 1000

            # Record ping latency
            self._stats_monitor.update_remote_ping_latency(latency)
            # Record error code (0 means success)
            self._stats_monitor.update_remote_ping_error_code(error_code)

            if error_code != 0:
                logger.warning(f"Ping failed with error code: {error_code}")
                return False

            return True

        except asyncio.TimeoutError:
            logger.warning("Ping timeout")
            self._stats_monitor.update_remote_ping_error_code(PING_TIMEOUT_ERROR_CODE)
            return False
        except Exception as e:
            logger.error(f"Ping error: {e}")
            self._stats_monitor.update_remote_ping_error_code(PING_GENERIC_ERROR_CODE)
            return False

    def _put_and_get_check(self) -> bool:
        if self.backend.local_cpu_backend is None or self.backend.connection is None:
            return False

        with self._resource_manager() as (put_obj, get_obj):
            if put_obj is None:
                return False
            if get_obj is None:
                logger.warning("Get failed, the return value is None, check failed.")
                return False
            return torch.equal(put_obj.raw_tensor, get_obj.raw_tensor)

    @contextmanager
    def _resource_manager(self):
        key = CacheEngineKey(
            fmt="vllm",
            model_name="test",
            world_size=1,
            worker_id=0,
            chunk_hash=0,
            dtype=torch.bfloat16,
        )
        connector = self.backend.connection
        if isinstance(connector, InstrumentedRemoteConnector):
            connector = connector.getWrappedConnector()
            if isinstance(connector, AuditConnector):
                connector = connector.real_connector
        shapes = connector.meta_shapes
        dtypes = connector.meta_dtypes
        fmt = connector.meta_fmt
        put_obj, get_obj = None, None
        try:
            # check exists in put task
            if self.backend.exists_in_put_tasks(key):
                logger.warning("Key still exists in put tasks, check failed.")
                yield None, None
            else:
                check_put_timeout = self.backend.config.get_extra_config_value(
                    CHECK_PUT_TIMEOUT_CONFIG_KEY, DEFAULT_CHECK_PUT_TIMEOUT
                )
                # put
                put_obj = self.backend.local_cpu_backend.allocate(shapes, dtypes, fmt)
                future = self.backend.submit_put_task(key, put_obj)
                future.result(timeout=check_put_timeout)
                # exists
                exists = self.backend.contains(key)
                if not exists:
                    raise RuntimeError("Key not exists after put")
                # get
                get_obj = self.backend.get_blocking(key)
                yield put_obj, get_obj
        except asyncio.TimeoutError:
            logger.warning("Check timeout, check failed.")
            yield None, None
        except Exception as e:
            logger.warning("Check error, check failed: %s", e)
            yield None, None
        finally:
            if put_obj is not None:
                put_obj.ref_count_down()
            if get_obj is not None:
                get_obj.ref_count_down()
