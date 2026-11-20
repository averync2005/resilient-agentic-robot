"""
Tests for the named-pose layer: poses.yaml loading, the POSE/PRESS interlock, the edge agent's arm graph, and `arm move`

- Every test drives the FakeController from test_arm_driver.py, never the real arm
- The arm graph tests use a canned chat model, so no llama-server is needed
"""

# ---------------------------------
# Imports
# ---------------------------------

# Standard library
import asyncio

# Test framework and test files
import pytest
import yaml

# Canned model replies
from langchain_core.messages import AIMessage

# Code under test
import capstone.edge_agent.edge_agent as edge
from capstone.arm_controller import arm_cli
from capstone.arm_controller.arm_actions import ArmActions, parseAction
from capstone.arm_controller.arm_driver import ArmDriver
from capstone.arm_controller.poses import PoseError, loadPoses

# Fake arm shared with the driver tests
from test_arm_driver import FakeController




# ---------------------------------
# Test helpers
# ---------------------------------

def allServos(degrees):
    """
    A pose with every servo at the same angle
    """

    return {servoId: degrees for servoId in range(1, 7)}


def writePoses(tmp_path, data):
    """
    Write a poses file and return its path
    """

    path = tmp_path / "poses.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    return path


@pytest.fixture
def posesPath(tmp_path):
    """
    A poses file with home, two button poses, and one button
    """

    return writePoses(tmp_path, {
        "poses": {"home": allServos(0.0), "button_1_above": allServos(10.0), "button_1_press": allServos(20.0)},
        "buttons": {"button_1": ["button_1_above", "button_1_press", "button_1_above"]}
    })


@pytest.fixture
def fake():
    """
    A fresh FakeController per test
    """

    return FakeController()


@pytest.fixture
def actions(fake, posesPath):
    """
    ArmActions over the fake controller and the test poses file
    """

    return ArmActions(ArmDriver(fake), loadPoses(posesPath), 1500)


class FakeChatModel:
    """
    Canned chat model that records the messages it was sent
    """

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)

        return AIMessage(content=self.reply)




# ---------------------------------
# Poses file tests
# ---------------------------------

def test_load_poses_reads_poses_and_buttons(posesPath):
    """
    A valid file should load every pose (int servo IDs, float angles) and every button sequence
    """

    library = loadPoses(posesPath)

    assert (library.pose("home") == {servoId: 0.0 for servoId in range(1, 7)})
    assert (library.buttonSequence("button_1") == ["button_1_above", "button_1_press", "button_1_above"])


def test_load_poses_rejects_an_angle_past_the_limits(tmp_path):
    """
    A pose outside SERVO_LIMITS should fail at load time, not mid-move
    """

    path = writePoses(tmp_path, {"poses": {"bad": {**allServos(0.0), 6: 115.0}}})

    with pytest.raises(PoseError, match="servo 6"):
        loadPoses(path)


def test_load_poses_rejects_a_pose_missing_a_servo(tmp_path):
    """
    Every pose has to name all six servos
    """

    pose = allServos(0.0)
    del pose[3]

    with pytest.raises(PoseError, match="missing"):
        loadPoses(writePoses(tmp_path, {"poses": {"partial": pose}}))


def test_load_poses_rejects_a_button_using_an_unknown_pose(tmp_path):
    """
    A button can only move through poses that exist in the file
    """

    path = writePoses(tmp_path, {"poses": {"home": allServos(0.0)}, "buttons": {"button_1": ["home", "nowhere"]}})

    with pytest.raises(PoseError, match="nowhere"):
        loadPoses(path)


def test_load_poses_explains_a_missing_file(tmp_path):
    """
    No file should point at `arm teach`
    """

    with pytest.raises(PoseError, match="arm teach"):
        loadPoses(tmp_path / "missing.yaml")


def test_save_round_trips(posesPath):
    """
    Saving and reloading should give back the same poses and buttons
    """

    library = loadPoses(posesPath)
    library.poses["new"] = allServos(-5.25)
    library.save()

    reloaded = loadPoses(posesPath)

    assert (reloaded.pose("new")[1] == -5.2)
    assert (reloaded.buttons == library.buttons)


def test_unknown_names_raise_unknown_pose_error(posesPath):
    """
    Looking up a name that isn't approved should raise the interlock error
    """

    library = loadPoses(posesPath)

    with pytest.raises(PoseError):
        library.pose("dance")

    with pytest.raises(PoseError):
        library.buttonSequence("button_9")




# ---------------------------------
# Parsing tests
# ---------------------------------

def test_parse_action_finds_pose_and_press():
    """
    The first POSE or PRESS line should be pulled out, even with text or backticks around it
    """

    assert (parseAction("POSE home") == ("POSE", "home"))
    assert (parseAction("Sure.\n`PRESS button_1`") == ("PRESS", "button_1"))


def test_parse_action_ignores_ordinary_prose():
    """
    Lowercase "press the button" in a normal answer shouldn't count as an action
    """

    assert (parseAction("I can press the button if you ask.") is None)
    assert (parseAction("") is None)




# ---------------------------------
# Action tests
# ---------------------------------

def test_pose_moves_every_servo_with_the_pose_duration(actions, fake):
    """
    An approved pose should go out as one six-servo move at the configured duration
    """

    accepted, message = actions.run("POSE", "button_1_above")

    assert (accepted)
    assert (fake.moves == [([(servoId, 542) for servoId in range(1, 7)], 1500, True)])


def test_press_moves_through_the_button_sequence(actions, fake):
    """
    A button press should move through each pose in its sequence, in order
    """

    accepted, message = actions.run("PRESS", "button_1")

    assert (accepted)
    assert ([move[0][0][1] for move in fake.moves] == [542, 583, 542])


def test_names_are_matched_case_insensitively(actions, fake):
    """
    "HOME" from the model should match the approved "home"
    """

    assert (actions.run("POSE", "HOME")[0])


def test_unapproved_pose_is_rejected_and_nothing_is_sent(actions, fake):
    """
    The interlock: a name not in poses.yaml is refused before the driver sees it
    """

    accepted, message = actions.run("POSE", "dance")

    assert (not accepted)
    assert (message.startswith("REJECTED"))
    assert (fake.moves == [])


def test_unapproved_button_is_rejected_and_nothing_is_sent(actions, fake):
    """
    Same interlock for buttons
    """

    accepted, message = actions.run("PRESS", "button_9")

    assert (not accepted)
    assert (fake.moves == [])


def test_arm_failure_is_reported_not_raised(actions, fake, monkeypatch):
    """
    A lost USB link mid-move should come back as an ARM ERROR result
    """

    def unplugged(*args, **kwargs):
        raise OSError("write error")

    monkeypatch.setattr(fake, "setPosition", unplugged)

    accepted, message = actions.run("POSE", "home")

    assert (not accepted)
    assert (message.startswith("ARM ERROR"))




# ---------------------------------
# Edge agent arm graph tests
# ---------------------------------

def test_arm_graph_runs_the_action_gemma_names(actions, fake):
    """
    "POSE home" from the model should move the arm and come back as the reply
    """

    graph = edge.buildArmGraph(FakeChatModel("POSE home"), actions)

    reply = asyncio.run(edge.generateReply(graph, edge.EdgeSettings(_env_file=None), "go home"))

    assert (reply == "Gemma chose: POSE home\nMoved to pose 'home'")
    assert (len(fake.moves) == 1)


def test_arm_graph_rejects_an_unapproved_pose_from_gemma(actions, fake):
    """
    Even if the model names an unapproved pose, the interlock should refuse it
    """

    graph = edge.buildArmGraph(FakeChatModel("POSE backflip"), actions)

    reply = asyncio.run(edge.generateReply(graph, edge.EdgeSettings(_env_file=None), "do a backflip"))

    assert ("REJECTED" in reply)
    assert (fake.moves == [])


def test_arm_graph_passes_plain_answers_through(actions, fake):
    """
    A reply with no action (ex. a status question) should come back unchanged, with nothing sent
    """

    graph = edge.buildArmGraph(FakeChatModel("I'm online."), actions)

    reply = asyncio.run(edge.generateReply(graph, edge.EdgeSettings(_env_file=None), "are you there?"))

    assert (reply == "I'm online.")
    assert (fake.moves == [])


def test_arm_graph_merges_instructions_into_one_system_message(actions):
    """
    Gemma's template expects one system message first, so the arm instructions join the existing one
    """

    chatModel = FakeChatModel("ok")
    graph = edge.buildArmGraph(chatModel, actions)

    asyncio.run(edge.generateReply(graph, edge.EdgeSettings(_env_file=None, system_prompt="base prompt"), "hi"))
    sent = chatModel.calls[0]

    assert ([message.type for message in sent] == ["system", "human"])
    assert (sent[0].content.startswith("base prompt"))
    assert ("Approved poses: button_1_above, button_1_press, home" in sent[0].content)


def test_direct_command_skips_the_model(actions, fake):
    """
    "/pose home" should run without a model, and anything else should show usage
    """

    assert (edge.runDirectCommand(actions, "/pose home") == "Moved to pose 'home'")
    assert (edge.runDirectCommand(actions, "/dance").startswith("Direct commands"))




# ---------------------------------
# arm move tests
# ---------------------------------

def test_arm_move_accepts_an_in_range_command(fake, capsys):
    """
    An in-range move should print ACCEPTED and reach the controller
    """

    assert (arm_cli.moveOne(ArmDriver(fake), 6, 20.0, 1000))
    assert ("ACCEPTED" in capsys.readouterr().out)
    assert (len(fake.moves) == 1)


def test_arm_move_rejects_an_out_of_range_command(fake, capsys):
    """
    An out-of-range move should print REJECTED and send nothing
    """

    assert (not arm_cli.moveOne(ArmDriver(fake), 6, 150.0, 1000))
    assert ("REJECTED" in capsys.readouterr().out)
    assert (fake.moves == [])