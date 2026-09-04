from scripts.rollout_collection_common import deterministic_job_id, default_collection_config, make_jobs


def test_job_manifest_has_expected_full_size():
    jobs = make_jobs(default_collection_config())
    assert len(jobs) == 804
    assert sum(1 for j in jobs if j["phase"] == "preflight") == 4
    assert sum(1 for j in jobs if j["phase"] == "full") == 800
    assert len({j["job_id"] for j in jobs}) == len(jobs)


def test_job_id_is_deterministic():
    assert deterministic_job_id("p", 1, 2, 3) == deterministic_job_id("p", 1, 2, 3)
