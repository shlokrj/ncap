"""Serializable settings for seed and state-pool training."""

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
    pool_size: int = 0
    damage_probability: float = 0.0
    damage_fraction: float = 0.25

    def __post_init__(self):
        integer_fields = ('channels', 'hidden_size', 'size', 'padding', 'batch_size',
                          'iterations', 'min_steps', 'max_steps', 'seed', 'eval_seed',
                          'eval_steps', 'threads', 'pool_size')
        for name in integer_fields:
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if min(self.hidden_size, self.batch_size, self.iterations, self.min_steps,
               self.eval_steps, self.threads) < 1 or self.channels < 4:
            raise ValueError('training dimensions and step counts must be positive; channels >= 4')
        if not 0 <= self.damage_probability <= 1 or not 0 <= self.damage_fraction <= 1:
            raise ValueError('damage probability and fraction must be in [0, 1]')
        if self.damage_probability and not self.pool_size:
            raise ValueError('damage training requires a state pool')
        if self.pool_size and self.pool_size < self.batch_size:
            raise ValueError('pool_size must be zero or at least batch_size')
        if self.size <= 2 * self.padding or self.max_steps < self.min_steps:
            raise ValueError('invalid target padding or rollout interval')
        if not 0 <= self.fire_rate <= 1 or not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('invalid fire rate or learning rate')
