def test_action_interface_dimension_constant():
    from pathlib import Path

    text = Path("openvla/experiments/robot/robot_utils.py").read_text(encoding="utf-8")
    assert "ACTION_DIM = 7" in text
