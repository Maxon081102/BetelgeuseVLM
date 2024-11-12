from typing import List


from aits import BenchmarkRunner
from aits import PreparedItem
from aits import Batch

from .config import Config
from .question import ImageQuestion
from .dataset import MathVistaDataset
from .answer_checker import MathVistaChecker
from .ListBatch import ListBatch


class ImageBenchRunner(BenchmarkRunner):
    def __init__(self, config: Config):
        super().__init__(config)
        self.dataset = MathVistaDataset(config)
        self.answer_checker = MathVistaChecker(config)

    def prepare_data(self, question: ImageQuestion) -> PreparedItem:
        return {
            "image": question.image,
            "answer": question.answer,
            "question": question.question,
        }

    def collect_batch(self, prepared_data: List[PreparedItem]) -> Batch:
        return ListBatch.collect_batch(prepared_data)
