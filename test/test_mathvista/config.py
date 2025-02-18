from typing import Optional

from aits import BenchmarkConfig

class Config(BenchmarkConfig):
    data_path: str = "/data/datasets"
    test_size: str = "testmini"
    size: Optional[int] = None