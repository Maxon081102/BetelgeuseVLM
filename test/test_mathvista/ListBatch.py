from typing import Tuple, List

from aits import Batch

class ListBatch(Batch):
    def __init__(self, answers, question, images):
        super().__init__()
        self.answers = answers
        self.question = question
        self.images = images
        
    
    @classmethod
    def collect_batch(cls, data: List, **kwargs) -> "ListBatch":
        answers = []
        questions = []
        images = []
        for item in data:
            # answers.append(item.answer)
            # question.append(item.question)
            # images.append(item.image)
            answers.append(item['answer'])
            questions.append(item['question'])
            images.append(item['image'])
        
        return cls(answers, questions, images)

    def get_task_batch(self):
        return [self.question, self.images]

    def get_answer_batch(self):
        return self.answers