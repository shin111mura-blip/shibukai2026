def test_temporal_order_contract():
    order = [
        "observation",
        "rgb",
        "instruction",
        "sim_state",
        "oracle_graph",
        "policy_action",
        "save_pre_action_tuple",
        "env_step",
    ]
    assert order.index("oracle_graph") < order.index("env_step")
    assert order.index("rgb") < order.index("policy_action")
