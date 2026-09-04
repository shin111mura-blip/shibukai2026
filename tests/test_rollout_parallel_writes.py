from scripts.rollout_collection_common import default_collection_config, make_jobs


def test_workers_have_separate_output_paths():
    jobs = make_jobs(default_collection_config())
    preflight_paths = [j["output_path"] for j in jobs if j["phase"] == "preflight"]
    assert len(preflight_paths) == len(set(preflight_paths))
    assert all("worker_" in p for p in preflight_paths)
