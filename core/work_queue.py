"""Process-local, non-preemptive priority admission for engine work."""
import asyncio
from collections import deque
from contextvars import ContextVar


ANONYMOUS_WORK = ContextVar("anonymous_work", default=False)


class PrioritySemaphore(asyncio.Semaphore):
    """FIFO within each class; normal work always precedes waiting anonymous work."""

    def __init__(self, value=1):
        super().__init__(value)
        self._normal = deque()
        self._anonymous = deque()

    async def acquire(self):
        queue = self._anonymous if ANONYMOUS_WORK.get() else self._normal
        future = asyncio.get_running_loop().create_future()
        queue.append(future)
        self._dispatch()
        try:
            await future
            return True
        except BaseException:
            # A slot may already have been reserved when the waiter is cancelled.
            if future.done() and not future.cancelled():
                self._value += 1
            raise
        finally:
            if future in queue:
                queue.remove(future)
            self._dispatch()

    def _dispatch(self):
        while self._value > 0 and (self._normal or self._anonymous):
            queue = self._normal if self._normal else self._anonymous
            future = queue.popleft()
            if future.cancelled():
                continue
            self._value -= 1
            future.set_result(True)

    def release(self):
        self._value += 1
        self._dispatch()
