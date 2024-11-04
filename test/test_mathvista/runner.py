from typing import List


from aits import BenchmarkRunner
from aits import PreparedItem
from aits import Batch

from .config import Config
from .question import ImageQuestion
from .dataset import MathVistaDataset
from .answer_checker import MathVistaChecker


class HumanEvalRunner(BenchmarkRunner):
    def __init__(self, config: Config):
        super().__init__(config)
        self.dataset = MathVistaDataset(config)
        self.answer_checker = MathVistaChecker()

    def prepare_data(self, question: ImageQuestion) -> PreparedItem:
        return BasePreparedItem.prepare_item(question)

    def collect_batch(self, prepared_data: List[PreparedItem]) -> Batch:
        return BaseBatch.collect_batch(
            batch_size=self.config.batch_size, data=prepared_data
        )
