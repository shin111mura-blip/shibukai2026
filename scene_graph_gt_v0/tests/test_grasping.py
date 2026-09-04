from scene_graph.grasp_detector import FingerGeomSets, grasping_edges


FINGERS = FingerGeomSets(left=("left_pad",), right=("right_pad",))
OBJECTS = {"obj_a": ["obj_a_geom"], "obj_b": ["obj_b_geom"]}


def test_left_only_contact_false():
    edges, diag = grasping_edges([{"geom1": "left_pad", "geom2": "obj_a_geom"}], OBJECTS, FINGERS)
    assert edges == []
    assert diag["obj_a"]["left_finger_contact"] is True
    assert diag["obj_a"]["right_finger_contact"] is False


def test_right_only_contact_false():
    edges, diag = grasping_edges([{"geom1": "right_pad", "geom2": "obj_a_geom"}], OBJECTS, FINGERS)
    assert edges == []
    assert diag["obj_a"]["right_finger_contact"] is True


def test_both_fingers_same_object_true():
    edges, _diag = grasping_edges(
        [{"geom1": "left_pad", "geom2": "obj_a_geom"}, {"geom1": "right_pad", "geom2": "obj_a_geom"}],
        OBJECTS,
        FINGERS,
    )
    assert [(e.subject, e.predicate, e.object) for e in edges] == [("gripper", "grasping", "obj_a")]


def test_libero_official_result_overrides_contact_fallback():
    edges, diag = grasping_edges(
        [{"geom1": "left_pad", "geom2": "obj_a_geom"}, {"geom1": "right_pad", "geom2": "obj_a_geom"}],
        OBJECTS,
        FINGERS,
        official_results={"obj_a": False},
    )
    assert edges == []
    assert diag["obj_a"]["contact_grasp_result"] is True
    assert diag["obj_a"]["official_grasp_result"] is False
    assert diag["obj_a"]["rule"] == "libero_official_check_grasp"


def test_libero_official_true_emits_grasp_edge():
    edges, diag = grasping_edges([], OBJECTS, FINGERS, official_results={"obj_a": True})
    assert [(e.subject, e.predicate, e.object) for e in edges] == [("gripper", "grasping", "obj_a")]
    assert diag["obj_a"]["contact_grasp_result"] is False
    assert diag["obj_a"]["official_grasp_result"] is True
    assert diag["obj_a"]["rule"] == "libero_official_check_grasp"


def test_fingers_on_different_objects_false_for_both():
    edges, _diag = grasping_edges(
        [{"geom1": "left_pad", "geom2": "obj_a_geom"}, {"geom1": "right_pad", "geom2": "obj_b_geom"}],
        OBJECTS,
        FINGERS,
    )
    assert edges == []


def test_grasping_not_reversed_and_no_touching_edge():
    edges, _diag = grasping_edges(
        [{"geom1": "left_pad", "geom2": "obj_a_geom"}, {"geom1": "right_pad", "geom2": "obj_a_geom"}],
        OBJECTS,
        FINGERS,
    )
    assert all(e.subject == "gripper" for e in edges)
    assert all(e.predicate != "touching" for e in edges)
