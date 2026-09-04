from pathlib import Path

from scripts.rollout_collection_common import validate_episode_dir


def test_incomplete_episode_is_not_valid(tmp_path: Path):
    episode_dir = tmp_path / "episode"
    episode_dir.mkdir()
    ok, errors, _meta = validate_episode_dir(episode_dir)
    assert not ok
    assert errors
