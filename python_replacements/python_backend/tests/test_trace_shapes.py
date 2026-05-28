from python_backend.trace import (
    run_start_record,
    search_step_record,
    run_end_record,
    REQUIRED_RUN_START_KEYS,
    REQUIRED_SEARCH_STEP_KEYS,
    REQUIRED_RUN_END_KEYS,
)


def test_run_start_has_all_required_keys():
    rec = run_start_record(run_id="test-run")
    for k in REQUIRED_RUN_START_KEYS:
        assert k in rec, f"missing required run_start key: {k}"


def test_run_start_no_extra_required_keys():
    rec = run_start_record(run_id="test-run")
    for k in rec:
        assert k in REQUIRED_RUN_START_KEYS | {"record_type"}, (
            f"unexpected key in run_start: {k}"
        )


def test_search_step_has_all_required_keys():
    rec = search_step_record(
        t=0,
        rng_fingerprint="u64:12345",
        population_id=0,
        selected_parent_hashes=["sha256:abc"],
        proposal_operator="sr.mutation.replace_subtree_v1",
        mutation_or_crossover_type="mutation",
        candidate_hash_after="sha256:def",
        validity_status="valid",
    )
    for k in REQUIRED_SEARCH_STEP_KEYS:
        assert k in rec, f"missing required search_step key: {k}"


def test_search_step_with_invalid():
    rec = search_step_record(
        t=1,
        rng_fingerprint="u64:67890",
        population_id=0,
        selected_parent_hashes=[],
        proposal_operator="sr.mutation.point_v1",
        mutation_or_crossover_type="mutation",
        candidate_hash_after="sha256:ghi",
        validity_status="invalid",
        invalid_reason_code="SR-INV-EVAL-001",
    )
    assert rec["invalid_reason_code"] == "SR-INV-EVAL-001"


def test_run_end_has_all_required_keys():
    rec = run_end_record(run_id="test-run")
    for k in REQUIRED_RUN_END_KEYS:
        assert k in rec, f"missing required run_end key: {k}"


def test_run_end_values():
    rec = run_end_record(
        run_id="r1",
        completion_status="success",
        termination_reason="max_iterations",
    )
    assert rec["record_type"] == "run_end"
    assert rec["completion_status"] == "success"
    assert rec["termination_reason"] == "max_iterations"
