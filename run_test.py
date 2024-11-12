from test.test_mathvista.runner import ImageBenchRunner
from test.test_mathvista.config import Config
from test.agent import Agent


if __name__ == "__main__":
    config = Config(name="test", batch_size=1, random_seed=777)
    runner = ImageBenchRunner(config)
    agent = Agent()
    runner.run_benchmark(agent)
    print(runner.answer_checker.calculate())