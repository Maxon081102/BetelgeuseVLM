from typing import Tuple, List
from PIL.PngImagePlugin import PngImageFile
from transformers import AutoProcessor

from aits import Question


class ImageQuestion(Question):
    image: PngImageFile

    def __init__(self, question):
        self.image = question["decoded_image"]
        self.answer = question["answer"]
        self.question = question["query"]

    def prepare(self) -> str:
        return self.answer

    def prepare_question(self) -> Tuple[str, PngImageFile]:
        return self.question, self.image
