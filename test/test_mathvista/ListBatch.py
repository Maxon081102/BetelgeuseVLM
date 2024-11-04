from typing import Tuple, List

from aits import Batch

class ListBatch(Batch):
    def __init__()
    
    @classmethod
    def collect_batch(batch_size: int, data: List, **kwargs) -> "ListBatch":
        self.answers = []
        self.tasks = []
        self.images = []
        for item in data:
            self.answers.append(item['answer'])
            self.tasks.append(item['task'])
            self.images.append(item['image'])

    @abstractmethod
    def get_task_batch(self):
        raise NotImplementedError

    @abstractmethod
    def get_answer_batch(self):
        raise NotImplementedError