import numpy as np
from typing import List, Union

from aits import AnswerChecker
from aits import Answer
from .answer import MathVistaAnswer
from .config import Config


class MathVistaChecker(AnswerChecker):
    def __init__(self, config: Config):
        self.config = config
        self.agent_answers = []
        self.real_answer: MathVistaAnswer = []

    def add_answers(
        self, agent_answers: List[List[Answer]], real_answer: List[str]
    ):
        self.agent_answers.extend(agent_answers)
        for answer in real_answer:
            self.real_answer.append(MathVistaAnswer(answer))

    def calculate(self) -> float:
        results = []
        for i in range(len(self.real_answer)):
            right_count = 0
            real_answer = self.real_answer[i]
            for agent_answer in self.agent_answers[i]:
                answer = agent_answer._get_answer()
                if answer is not None:
                    if answer == real_answer:
                        right_count += 1
            results.append(right_count)
        return np.mean(results)
