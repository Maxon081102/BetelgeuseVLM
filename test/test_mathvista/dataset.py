import random
from typing import List
from datasets import load_dataset

from aits import Dataset


from .config import Config
from .question import ImageQuestion


class MathVistaDataset(Dataset):
    def __init__(
        self,
        config: Config,
    ) -> None:
        self.config = config
        self.data = load_dataset(
            "AI4Math/MathVista",
            cache_dir=config.data_path,
        )

        self.train_data = None
        self.test_data = self.data[config.test_size]
        if config.size is not None:
            self.test_data = self.test_data.shuffle(seed=config.random_seed).select(
                range(config.size)
            )

    def prepare_shots_examples(self) -> List[ImageQuestion]:
        assert self.train_data is not None, "No train data"
        assert self.config.shot_count < len(self.data["train"])
        random.seed(self.config.random_seed)
        shots_from_train = [
            ImageQuestion(self.train_data[i])
            for i in random.sample(range(len(self.train_data)), self.config.shot_count)
        ]
        return shots_from_train

    def get_question(self, i: int) -> ImageQuestion:
        return ImageQuestion(self.test_data[i])