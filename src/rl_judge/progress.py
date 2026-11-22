"""Прошло / осталось строкой лога. Не tqdm, чтобы нормально читалось через tee."""

from __future__ import annotations

import time


def fmt_duration(seconds: float) -> str:
    """45 с, 3 мин 05 с, 2 ч 10 мин."""
    s = int(round(max(seconds, 0)))
    if s < 60:
        return f"{s} с"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m} мин {s:02d} с"
    h, m = divmod(m, 60)
    return f"{h} ч {m:02d} мин"


class Eta:
    """Остаток оценивается линейно по сделанному."""

    def __init__(self, total: int, *, min_interval: float = 10.0) -> None:
        self.total = max(int(total), 1)
        self._start = time.perf_counter()
        self._last_print = float("-inf")
        self._min_interval = min_interval

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def line(self, done: int) -> str:
        """пример:40/250 (16%), прошло 3 мин 10 с, осталось ≈ 16 мин 40 с, конец ≈ 18:42"""
        text = f"{done}/{self.total} ({done / self.total:.0%}), прошло {fmt_duration(self.elapsed)}"
        if done > 0:
            remaining = self.elapsed * (self.total - done) / done
            finish = time.strftime("%H:%M", time.localtime(time.time() + remaining))
            text += f", осталось ≈ {fmt_duration(remaining)}, конец ≈ {finish}"
        return text

    def due(self) -> bool:
        """Пора печатать: с прошлой строки прошло min_interval секунд."""
        now = time.perf_counter()
        if now - self._last_print >= self._min_interval:
            self._last_print = now
            return True
        return False
