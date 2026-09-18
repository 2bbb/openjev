"""Bounded, process-local prefix snapshots for resident MLX scorers."""
from __future__ import annotations

import copy


def common_prefix(sequences):
    """Exact common token prefix, leaving at least one scoring token per row."""
    if not sequences or any(not ids for ids in sequences):
        raise ValueError("Nonempty encoded prompts required")
    first = sequences[0]
    end = min(len(ids) - 1 for ids in sequences)
    for index in range(end):
        if any(ids[index] != first[index] for ids in sequences[1:]):
            return first[:index]
    return first[:end]


def suffix_groups(lengths, max_batch_size=None):
    """Group similar lengths while retaining indices for original output order."""
    if max_batch_size is None:
        return [list(range(len(lengths)))]
    if type(max_batch_size) is not int or max_batch_size < 1:
        raise ValueError("max_batch_size must be a positive integer")
    order = sorted(range(len(lengths)), key=lengths.__getitem__)
    return [order[i:i+max_batch_size] for i in range(0, len(order), max_batch_size)]


class ResidentPrefixCache:
    """One retained exact-token snapshot, including KV and recurrent state.

    Single-owner use only. Invalidate after modifying weights, dtype, tokenizer
    configuration or adapters in place; distinct model/tokenizer objects miss
    automatically. Retained tokens need not be static: mismatches safely rebuild.
    """
    def __init__(self, max_prefix_tokens=160):
        if type(max_prefix_tokens) is not int or max_prefix_tokens < 1:
            raise ValueError("max_prefix_tokens must be a positive integer")
        self.max_prefix_tokens = max_prefix_tokens
        self.clear()

    def clear(self):
        self.model = self.tokenizer = self.cache = None
        self.tokens = []

    def prepare(self, model, tokenizer, prefix):
        """Return independent state at prefix end and number of reused tokens.

        All work, including miss/rebuild and copy, belongs in request timing.
        A snapshot is saved before processing the changing part, never trimmed.
        """
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        if not prefix:
            return make_prompt_cache(model), 0
        hit = (self.cache is not None and model is self.model and tokenizer is self.tokenizer
               and prefix[:len(self.tokens)] == self.tokens)
        if not hit:
            self.clear()
            tokens = list(prefix[:self.max_prefix_tokens])
            cache = make_prompt_cache(model)
            hidden = model.model(mx.array([tokens]), cache=cache)
            mx.eval(hidden, [entry.state for entry in cache])
            self.model, self.tokenizer = model, tokenizer
            self.tokens, self.cache = tokens, cache
        branch = copy.deepcopy(self.cache)
        rest = prefix[len(self.tokens):]
        if rest:
            hidden = model.model(mx.array([rest]), cache=branch)
            mx.eval(hidden, [entry.state for entry in branch])
        else:
            mx.eval([entry.state for entry in branch])
        mx.synchronize()
        return branch, len(self.tokens) if hit else 0
