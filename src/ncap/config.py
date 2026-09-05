"""Serializable settings for seed-only training."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TrainConfig:
    channels: int = 16
    hidden_size: int = 128
    fire_rate: float = 0.5
    size: int = 64
    padding: int = 8
    batch_size: int = 8
    iterations: int = 8000
    min_steps: int = 64
    max_steps: int = 96
    learning_rate: float = 0.002
    seed: int = 0
    eval_seed: int = 10000
    eval_steps: int = 96
    threads: int = 1

    def __post_init__(self):
        integer_fields = ('channels', 'hidden_size', 'size', 'padding', 'batch_size',
                          'iterations', 'min_steps', 'max_steps', 'seed', 'eval_seed',
                          'eval_steps', 'threads')
        for name in integer_fields:
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if min(self.hidden_size, self.batch_size, self.iterations, self.min_steps,
               self.eval_steps, self.threads) < 1 or self.channels < 4:
            raise ValueError('training dimensions and step counts must be positive; channels >= 4')
        if self.size <= 2 * self.padding or self.max_steps < self.min_steps:
            raise ValueError('invalid target padding or rollout interval')
        if not 0 <= self.fire_rate <= 1 or not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('invalid fire rate or learning rate')
