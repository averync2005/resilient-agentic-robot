"""
The edge agent's arm actions: move to a named pose, or press a named button

Interlock (S-1):
- The local model only ever names an action and a pose or button (ex. "POSE home", "PRESS button_a")
- `ArmActions.run()` looks the name up in the poses file, and anything not in it is rejected before the driver sees it
- The driver then checks every angle against SERVO_LIMITS, so a bad pose can't slip through either

Model output:
- `parseAction()` pulls the first "POSE <name>" or "PRESS <name>" out of the model's reply
- `ArmActions.readReply()` also accepts a reply that is only an approved name (Gemma sometimes answers just "home")
- Parsed here instead of through llama-server's tool calling (Gemma 4's native tool-call format isn't reliably parsed there)
"""

# ---------------------------------
# Imports
# ---------------------------------

# Standard library
import re
import threading

# The driver underneath, and the approved poses it's allowed to reach
from capstone.arm_controller.arm_driver import ArmDriver, ArmError
from capstone.arm_controller.poses import PoseError, PoseLibrary




# ---------------------------------
# Move timing
# ---------------------------------

# Default time for a full-arm move between poses (slower moves draw less peak current from the 1 A supply)
DEFAULT_POSE_MS = 2500




# ---------------------------------
# Reading the model's reply
# ---------------------------------

# Uppercase keyword only, so ordinary prose like "press the button" doesn't count as an action
ACTION_PATTERN = re.compile(r"\b(POSE|PRESS)\s+[`'\"]?([A-Za-z0-9_\-]+)")


def parseAction(text: str) -> tuple[str, str] | None:
    """
    Find the first POSE or PRESS action in a model reply

    Args:
        text: The model's reply

    Returns:
        (action, name) such as ("POSE", "home"), or None if the reply doesn't name an action
    """

    match = ACTION_PATTERN.search(text or "")

    if (match is None):
        return None

    return (match.group(1), match.group(2))




# ---------------------------------
# Running actions on the arm
# ---------------------------------

class ArmActions:
    """
    Runs named-pose actions on the arm, one at a time

    - `run()` is the entry point (interlock, then the move), and never raises for a rejected or failed action
    - A lock keeps two requests (ex. two A2A messages) from driving the arm at once
    """

    def __init__(self, arm: ArmDriver, library: PoseLibrary, moveMs: int) -> None:
        """
        Store the arm, the approved poses, and how long each pose move takes

        Args:
            arm: An open ArmDriver (or one wrapping a fake controller in tests)
            library: The approved poses and buttons
            moveMs: Duration of each move between poses, in milliseconds
        """

        self.arm = arm
        self.library = library
        self.moveMs = moveMs
        self.lock = threading.Lock()

    def run(self, action: str, name: str) -> tuple[bool, str]:
        """
        Run one action, turning interlock rejections and arm failures into a result instead of an exception

        Args:
            action: "POSE" or "PRESS"
            name: The pose or button name (matched case-insensitively)

        Returns:
            (accepted, message), where accepted is False for rejections and arm failures
        """

        name = name.lower()

        with self.lock:
            try:
                if (action == "POSE"):
                    return (True, self.moveToPose(name))

                if (action == "PRESS"):
                    return (True, self.pressButton(name))

                return (False, f"REJECTED: unknown action {action!r}. Nothing was sent to the arm.")
            except PoseError as error:
                return (False, f"REJECTED: {error}. Nothing was sent to the arm.")
            except ArmError as error:
                return (False, f"ARM ERROR: {error}")

    def readReply(self, text: str) -> tuple[str, str] | None:
        """
        Find the action in a model reply, including a reply that is only an approved pose or button name

        Args:
            text: The model's reply

        Returns:
            (action, name), or None if the reply names no action and isn't a bare approved name
        """

        request = parseAction(text)

        if (request is not None):
            return request

        # Stripping quotes, backticks, and a trailing period (ex. "`home`." -> "home")
        bareName = (text or "").strip().strip("`'\".").strip().lower()

        if (bareName in self.library.poses):
            return ("POSE", bareName)

        if (bareName in self.library.buttons):
            return ("PRESS", bareName)

        return None

    def moveToPose(self, name: str) -> str:
        """
        Move every servo to one approved pose and wait for the move to finish

        Args:
            name: The pose's name

        Returns:
            A one-line summary for the reply

        Raises:
            PoseError: If the name isn't an approved pose (nothing is sent)
        """

        self.arm.moveServos(self.library.pose(name), durationMs=self.moveMs, wait=True)

        return f"Moved to pose '{name}'"

    def pressButton(self, name: str) -> str:
        """
        Press one approved button by moving through its pose sequence

        Args:
            name: The button's name

        Returns:
            A one-line summary for the reply

        Raises:
            PoseError: If the button isn't approved (nothing is sent)
        """

        sequence = self.library.buttonSequence(name)

        for poseName in sequence:
            self.arm.moveServos(self.library.pose(poseName), durationMs=self.moveMs, wait=True)

        return f"Pressed '{name}' ({' -> '.join(sequence)})"

    def describe(self) -> str:
        """
        List the approved poses and buttons, for the model's system prompt

        Returns:
            Two lines: the pose names and the button names
        """

        poseNames = ", ".join(sorted(self.library.poses)) or "none"
        buttonNames = ", ".join(sorted(self.library.buttons)) or "none"

        return f"Approved poses: {poseNames}\nApproved buttons: {buttonNames}"