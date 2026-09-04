from scene_graph_generator.graph_generator.token_selection import infer_prismatic_token_layout


class Tok:
    bos_token_id = 1


def test_prismatic_layout_inserts_image_tokens_after_bos():
    layout = infer_prismatic_token_layout([[1, 10, 11, 0]], [[1, 1, 1, 0]], image_token_count=4, tokenizer=Tok())
    assert layout.bos_positions == [0]
    assert (layout.image_start, layout.image_end) == (1, 5)
    assert layout.instruction_positions == [5, 6]
    assert layout.padding_positions == [3]
