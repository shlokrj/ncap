"""Detached state storage with fresh-seed injection on each sampled batch."""

import torch


class StatePool:
    def __init__(self, seed, size):
        if size < 1 or seed.shape[0] != 1:
            raise ValueError('pool requires a single seed and positive size')
        self.seed = seed.detach().clone()
        self.states = self.seed.repeat(size, 1, 1, 1)

    def sample(self, batch_size, target, *, generator):
        if not 1 <= batch_size <= len(self.states):
            raise ValueError('batch size must fit in pool')
        indices = torch.randperm(len(self.states), generator=generator)[:batch_size]
        batch = self.states[indices].clone()
        losses = (batch[:, :4] - target).square().mean(dim=(1, 2, 3))
        batch[losses.argmax()] = self.seed[0]
        return indices, batch

    def commit(self, indices, states):
        if states.shape != self.states[indices].shape or not torch.isfinite(states).all():
            raise ValueError('pool updates must have matching shapes and finite values')
        self.states[indices] = states.detach()
