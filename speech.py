"""TTS worker using macOS `say` in a background thread.

Drain-on-new behavior: enqueueing a new phrase drops any pending queued
phrases, but the currently-speaking utterance is allowed to finish naturally.
Interrupting mid-word felt jarring during testing, and with phrases capped to
~2 seconds the worst-case lag is small enough to live with.
"""

from __future__ import annotations

import queue
import subprocess
import threading


class TTSWorker:
    def __init__(self) -> None:
        self._queue: "queue.Queue[str | None]" = queue.Queue()
        self._enabled = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def say(self, phrase: str) -> bool:
        """Drain any pending queue and enqueue this phrase."""
        if not phrase:
            return False
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(phrase)
        return True

    def set_enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def enabled(self) -> bool:
        return self._enabled

    def stop(self) -> None:
        self._queue.put(None)

    def _run(self) -> None:
        while True:
            phrase = self._queue.get()
            if phrase is None:
                return
            if not self._queue.empty():
                continue
            if not self._enabled:
                continue
            try:
                subprocess.run(
                    ["say", phrase],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass
