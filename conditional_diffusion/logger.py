from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class StepLogger:
    max_step: Optional[int] = None
    skip: int = 1
    logger: Callable[[str], None] = print

    def update(self, step: int, message: str = "") -> None:
        if step % self.skip == 0:
            self.logger(self._full_message(step, message))

    def _full_message(self, step: int, message: str) -> str:
        if self.max_step is not None:
            return f"[{step}/{self.max_step}] {message}"
        else:
            return f"[{step}] {message}"
