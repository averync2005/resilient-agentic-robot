"""
Named poses and button sequences for the xArm 1S, stored in one YAML file

File layout (poses.yaml at the repo root by default):
- `poses`: pose name -> degrees from center per servo ID (all six servos, IDs 1-6)
- `buttons`: button name -> the poses to move through, in order, to press it (ex. above, press, above)

Rules:
- Only names in this file can be sent to the arm (everything else is rejected before it reaches the driver)
- Every angle is checked against SERVO_LIMITS when the file loads, so a bad edit fails at startup, not mid-move
- `uv run arm teach` writes poses here, so hand edits and taught poses live in the same place
"""

# ---------------------------------
# Imports
# ---------------------------------

# Standard library
import math
from dataclasses import dataclass, field
from pathlib import Path

# Reading and writing poses.yaml
import yaml

# Servo limits every saved angle is checked against
from capstone.arm_controller.arm_driver import SERVO_LIMITS, SERVO_NAMES




# ---------------------------------
# Errors
# ---------------------------------

class PoseError(Exception):
    """
    A problem with a pose: a missing or malformed poses file, an angle outside SERVO_LIMITS, or a name that isn't approved (the named-pose interlock)
    """




# ---------------------------------
# Checking file contents
# ---------------------------------

def checkPoseAngles(name: str, angles: dict) -> dict[int, float]:
    """
    Make sure one pose names all six servos with angles inside their limits

    Args:
        name: The pose's name (for error messages)
        angles: Servo ID -> degrees, as loaded from YAML

    Returns:
        The same angles with int keys and float values

    Raises:
        PoseError: If a servo is missing, unknown, non-numeric, or outside its limits
    """

    if (not isinstance(angles, dict)):
        raise PoseError(f"Pose '{name}' must map servo IDs to degrees")

    checked = {}

    for servoId, degrees in angles.items():
        if (servoId not in SERVO_NAMES):
            raise PoseError(f"Pose '{name}' names servo {servoId!r} (the xArm 1S has servos 1-6)")

        if (isinstance(degrees, bool) or not isinstance(degrees, (int, float)) or math.isnan(degrees)):
            raise PoseError(f"Pose '{name}' has a non-numeric angle for servo {servoId}: {degrees!r}")

        low, high = SERVO_LIMITS[servoId]

        if (not (low <= degrees <= high)):
            raise PoseError(f"Pose '{name}' puts servo {servoId} ({SERVO_NAMES[servoId]}) at {degrees} degrees (allowed: {low} to {high})")

        checked[servoId] = float(degrees)

    missing = sorted(set(SERVO_NAMES) - set(checked))

    if (missing):
        raise PoseError(f"Pose '{name}' is missing servo(s) {missing} (every pose names all six)")

    return checked




# ---------------------------------
# Pose library
# ---------------------------------

@dataclass
class PoseLibrary:
    """
    The approved poses and button sequences, plus the file they came from

    - `poses`: pose name -> degrees per servo ID
    - `buttons`: button name -> list of pose names to move through
    """

    path: Path
    poses: dict[str, dict[int, float]] = field(default_factory=dict)
    buttons: dict[str, list[str]] = field(default_factory=dict)

    def pose(self, name: str) -> dict[int, float]:
        """
        Look up one approved pose

        Args:
            name: The pose's name

        Returns:
            Degrees per servo ID

        Raises:
            PoseError: If the name isn't an approved pose
        """

        if (name not in self.poses):
            raise PoseError(f"'{name}' isn't an approved pose (approved: {', '.join(sorted(self.poses)) or 'none'})")

        return self.poses[name]

    def buttonSequence(self, name: str) -> list[str]:
        """
        Look up the poses that press one approved button

        Args:
            name: The button's name

        Returns:
            Pose names, in the order to move through them

        Raises:
            PoseError: If the name isn't an approved button
        """

        if (name not in self.buttons):
            raise PoseError(f"'{name}' isn't an approved button (approved: {', '.join(sorted(self.buttons)) or 'none'})")

        return self.buttons[name]

    def save(self) -> None:
        """
        Write the poses and buttons back to the file (angles rounded to 0.1 degree)
        """

        data = {
            "poses": {name: {servoId: round(degrees, 1) for servoId, degrees in sorted(angles.items())} for name, angles in self.poses.items()},
            "buttons": self.buttons
        }

        self.path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def loadPoses(path: str | Path) -> PoseLibrary:
    """
    Load and check the poses file

    Args:
        path: Where the YAML file is

    Returns:
        A PoseLibrary with every pose checked against SERVO_LIMITS

    Raises:
        PoseError: If the file is missing or anything in it fails a check
    """

    path = Path(path)

    if (not path.exists()):
        raise PoseError(f"No poses file at {path} (teach poses with: uv run arm teach)")

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as error:
        raise PoseError(f"{path} isn't valid YAML ({error})") from error

    poses = {str(name): checkPoseAngles(str(name), angles) for name, angles in (data.get("poses") or {}).items()}

    buttons = {}

    for name, sequence in (data.get("buttons") or {}).items():
        if ((not isinstance(sequence, list)) or (not sequence)):
            raise PoseError(f"Button '{name}' must list the poses to move through")

        unknown = [poseName for poseName in sequence if (poseName not in poses)]

        if (unknown):
            raise PoseError(f"Button '{name}' uses pose(s) not in the file: {unknown}")

        buttons[str(name)] = [str(poseName) for poseName in sequence]

    return PoseLibrary(path, poses, buttons)