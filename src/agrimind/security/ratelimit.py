"""Giới hạn tần suất theo từng bên (thùng token, trong bộ nhớ một tiến trình).

Chạy nhiều bản sao thì đặt thêm giới hạn ở cổng phía trước (nginx/API gateway);
lớp này là hàng rào cuối, không phải hàng rào duy nhất.
"""

from __future__ import annotations

import threading
import time
from typing import Callable


class RateLimiter:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self._b: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, per_minute: int) -> float:
        """Trả 0 nếu được phép; ngược lại số giây phải chờ."""
        now = self.clock()
        rate = per_minute / 60.0
        with self._lock:
            tokens, last = self._b.get(key, (float(per_minute), now))
            tokens = min(float(per_minute), tokens + (now - last) * rate)
            if tokens >= 1:
                self._b[key] = (tokens - 1, now)
                return 0.0
            self._b[key] = (tokens, now)
            return (1 - tokens) / rate
