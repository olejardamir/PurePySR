from __future__ import annotations

import numpy as np

from python_backend.mutation_weights import (
    MutationWeights,
    sample_mutation,
)


def test_sample_weighted_operator_deterministic():
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    weights = MutationWeights()

    seq1 = [sample_mutation(rng1, weights) for _ in range(100)]
    seq2 = [sample_mutation(rng2, weights) for _ in range(100)]
    assert seq1 == seq2


def test_zero_weight_disables_kind():
    rng = np.random.default_rng(42)
    # Set all weights to 0 except mutate_constant
    weights = MutationWeights(
        mutate_constant=1.0,
        mutate_operator=0.0,
        mutate_feature=0.0,
        swap_operands=0.0,
        rotate_tree=0.0,
        add_node=0.0,
        insert_node=0.0,
        delete_node=0.0,
        simplify=0.0,
        randomize=0.0,
        do_nothing=0.0,
        optimize=0.0,
        backsolve=0.0,
        form_connection=0.0,
        break_connection=0.0,
    )

    for _ in range(200):
        chosen = sample_mutation(rng, weights)
        assert chosen == "mutate_constant"
