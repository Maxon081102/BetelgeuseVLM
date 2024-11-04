import re
from typing import Optional

from aits.logger import init_logger
from aits import Answer


logger = init_logger(__name__)


class MathVistaAnswer(Answer):
    def __init__(self, answer: str):
        super().__init__()
        self.answer = answer

    def _get_answer(self) -> float:
        return self.answer