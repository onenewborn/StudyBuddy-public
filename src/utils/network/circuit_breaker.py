"""
Circuit Breaker - Simple circuit breaker for providers.
"""

import logging
import threading
import time
from typing import Dict

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Simple circuit breaker that opens when error rate is high.
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count: Dict[str, int] = {}
        self.last_failure_time: Dict[str, float] = {}
        self.state: Dict[str, str] = {}  # 'closed', 'open', 'half-open'
        self.lock = threading.Lock()

    def call(self, provider: str) -> bool:
        """Check if call is allowed."""
        with self.lock:
            state = self.state.get(provider, "closed")
            if state == "closed":
                return True
            elif state == "open":
                if time.time() - self.last_failure_time.get(provider, 0) > self.recovery_timeout:
                    self.state[provider] = "half-open"
                    logger.info(f"Circuit breaker for {provider} entering half-open state")
                    return True
                return False
            elif state == "half-open":
                return True

    def record_success(self, provider: str):
        """Record successful call."""
        with self.lock:
            if self.state.get(provider) == "half-open":
                self.state[provider] = "closed"
                self.failure_count[provider] = 0
                logger.info(f"Circuit breaker for {provider} closed")
            elif self.state.get(provider) == "closed":
                self.failure_count[provider] = 0

    def record_failure(self, provider: str):
        """Record failed call."""
        with self.lock:
            self.failure_count[provider] = self.failure_count.get(provider, 0) + 1
            self.last_failure_time[provider] = time.time()
            if self.failure_count[provider] >= self.failure_threshold:
                self.state[provider] = "open"
                logger.warning(
                    f"Circuit breaker for {provider} opened due to {self.failure_count[provider]} failures"
                )


# Global instance
circuit_breaker = CircuitBreaker()


def alert_callback(provider: str, rate: float):
    """Alert callback to trigger circuit breaker."""
    circuit_breaker.record_failure(provider)


def is_call_allowed(provider: str) -> bool:
    """Check if call is allowed by circuit breaker."""
    return circuit_breaker.call(provider)


def record_call_success(provider: str):
    """Record successful call."""
    circuit_breaker.record_success(provider)
