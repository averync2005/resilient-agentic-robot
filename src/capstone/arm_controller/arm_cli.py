"""
Command-line tools for the xArm 1S (the `arm` command)

Commands:
- `uv run arm status` -- Supply voltage and every servo's angle
- `uv run arm move <servo> <degrees>` -- One servo move, printed as ACCEPTED or REJECTED (rejections send nothing)
- `uv run arm pose <name>` / `uv run arm press <name>` -- Run a named pose or button from poses.yaml, without the model
- `uv run arm teach` -- Interactive pose teaching: free the servos, position the arm by hand, save the angles by name

Notes:
- Run from the repo root, so poses.yaml is found (or pass --poses)
- On the Jetson, the board needs the udev rule from README.md, or opening it fails with "open failed"
"""

# ---------------------------------
# Imports
# ---------------------------------

# Standard library
import argparse
import sys
from pathlib import Path

# The arm, its approved poses, and the actions that run them
from capstone.arm_controller.arm_actions import ArmActions
from capstone.arm_controller.arm_driver import SERVO_NAMES, ArmDriver, ArmError, openArm
from capstone.arm_controller.poses import PoseError, PoseLibrary, checkPoseAngles, loadPoses




# ---------------------------------
# Constants
# ---------------------------------

DEFAULT_POSES_FILE = "poses.yaml"
DEFAULT_POSE_MS = 1500  # Gentle enough for a full-arm move between poses

TEACH_HELP = """Commands:
  show              print every servo's angle
  free [ids...]     turn servos off so you can move them by hand (all six if no IDs - support the arm first)
  hold              lock every servo where it is now
  save <name>       save the current angles as a named pose
  go <name>         move to a saved pose
  button <name> <pose> [<pose>...]   save a button as the poses to move through
  list              list saved poses and buttons
  help              show this list
  quit              hold every servo and exit"""




# ---------------------------------
# Printing
# ---------------------------------

def formatAngles(angles: dict[int, float]) -> str:
    """
    Format angles as one line, labeled by servo name

    Args:
        angles: Degrees per servo ID

    Returns:
        A line like "1 gripper 0.0 | 2 wrist rotation -12.5 | ..."
    """

    return " | ".join(f"{servoId} {SERVO_NAMES[servoId]} {angles[servoId]:.1f}" for servoId in sorted(angles))




# ---------------------------------
# Commands
# ---------------------------------

def showStatus(arm: ArmDriver) -> None:
    """
    Print the supply voltage and every servo's angle

    Args:
        arm: An open ArmDriver
    """

    print(f"Supply: {arm.readBatteryVoltage():.2f} V")
    print(f"Angles: {formatAngles(arm.readPositions())}")


def moveOne(arm: ArmDriver, servoId: int, degrees: float, durationMs: int) -> bool:
    """
    Send one servo move, printing whether the driver accepted or rejected it

    Args:
        arm: An open ArmDriver
        servoId: The servo to move
        degrees: Target angle from center
        durationMs: Move time in milliseconds

    Returns:
        True if the move was accepted and sent
    """

    try:
        arm.moveServo(servoId, degrees, durationMs=durationMs)
    except (ArmError, ValueError) as error:
        print(f"REJECTED: {error}. Nothing was sent to the arm.")
        return False

    print(f"ACCEPTED: servo {servoId} ({SERVO_NAMES.get(servoId, '?')}) moved to {degrees} degrees in {durationMs} ms")

    return True


def runTeachCommand(arm: ArmDriver, library: PoseLibrary, words: list[str], poseMs: int) -> None:
    """
    Run one line typed at the teach prompt

    Args:
        arm: An open ArmDriver
        library: Poses and buttons, saved to disk on every change
        words: The typed line, split on spaces (first word is the command)
        poseMs: Move time for `go`, in milliseconds
    """

    command, arguments = words[0].lower(), words[1:]

    if (command == "show"):
        print(formatAngles(arm.readPositions()))
    elif (command == "free"):
        servoIds = [int(word) for word in arguments] or None
        arm.servosOff(servoIds)
        print(f"Servos {'all' if servoIds is None else servoIds} are free - move them by hand, then 'save <name>' or 'hold'")
    elif (command == "hold"):
        print(f"Holding: {formatAngles(arm.stop())}")
    elif ((command == "save") and (len(arguments) == 1)):
        # Reading before holding (holding re-sends exactly what was read)
        angles = arm.stop()
        library.poses[arguments[0]] = checkPoseAngles(arguments[0], angles)
        library.save()
        print(f"Saved pose '{arguments[0]}': {formatAngles(angles)}")
    elif ((command == "go") and (len(arguments) == 1)):
        ArmActions(arm, library, poseMs).moveToPose(arguments[0])
        print(f"At pose '{arguments[0]}'")
    elif ((command == "button") and (len(arguments) >= 2)):
        unknown = [poseName for poseName in arguments[1:] if (poseName not in library.poses)]

        if (unknown):
            print(f"Save these poses first: {unknown}")
            return

        library.buttons[arguments[0]] = arguments[1:]
        library.save()
        print(f"Saved button '{arguments[0]}': {' -> '.join(arguments[1:])}")
    elif (command == "list"):
        for name, angles in library.poses.items():
            print(f"pose {name}: {formatAngles(angles)}")
        for name, sequence in library.buttons.items():
            print(f"button {name}: {' -> '.join(sequence)}")
    else:
        print(TEACH_HELP)


def teach(arm: ArmDriver, posesPath: Path, poseMs: int) -> None:
    """
    Interactive loop for teaching named poses by hand

    Args:
        arm: An open ArmDriver
        posesPath: The poses file to read and write (created if missing)
        poseMs: Move time for `go`, in milliseconds
    """

    library = loadPoses(posesPath) if posesPath.exists() else PoseLibrary(posesPath)

    print(f"Teaching poses into {posesPath}. Type 'help' for commands.")

    while (True):
        try:
            line = input("teach> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            line = "quit"

        if (not line):
            continue

        if (line.lower() in ("quit", "exit")):
            arm.stop()
            print("Holding every servo where it is. Done.")
            return

        # One bad line (unknown pose, typo, lost USB) shouldn't end the session
        try:
            runTeachCommand(arm, library, line.split(), poseMs)
        except (ArmError, PoseError, ValueError, KeyError) as error:
            print(f"[ERROR] {error}")




# ---------------------------------
# Entry point
# ---------------------------------

def buildParser() -> argparse.ArgumentParser:
    """
    Define the `arm` command's subcommands and options

    Returns:
        The argument parser
    """

    parser = argparse.ArgumentParser(prog="arm", description="Drive the xArm 1S directly (no model)")
    parser.add_argument("--poses", default=DEFAULT_POSES_FILE, help=f"Poses file (default: {DEFAULT_POSES_FILE})")
    parser.add_argument("--ms", type=int, default=DEFAULT_POSE_MS, help=f"Move time in milliseconds (default: {DEFAULT_POSE_MS})")

    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("status", help="Supply voltage and every servo's angle")

    moveParser = subcommands.add_parser("move", help="Move one servo (prints ACCEPTED or REJECTED)")
    moveParser.add_argument("servo", type=int, help="Servo ID, 1 (gripper) to 6 (base)")
    moveParser.add_argument("degrees", type=float, help="Target angle from center")

    poseParser = subcommands.add_parser("pose", help="Move to a named pose from the poses file")
    poseParser.add_argument("name")

    pressParser = subcommands.add_parser("press", help="Press a named button from the poses file")
    pressParser.add_argument("name")

    subcommands.add_parser("teach", help="Teach named poses by hand")

    return parser


def main() -> None:
    """
    Parse the command line, open the arm, and run one command
    """

    args = buildParser().parse_args()

    try:
        arm = openArm()

        if (args.command == "status"):
            showStatus(arm)
        elif (args.command == "move"):
            sys.exit(0 if moveOne(arm, args.servo, args.degrees, args.ms) else 1)
        elif (args.command in ("pose", "press")):
            accepted, message = ArmActions(arm, loadPoses(args.poses), args.ms).run(args.command.upper(), args.name)
            print(message)
            sys.exit(0 if accepted else 1)
        elif (args.command == "teach"):
            teach(arm, Path(args.poses), args.ms)
    except (ArmError, PoseError) as error:
        print(f"[ERROR] {error}")
        sys.exit(1)


if (__name__ == "__main__"):
    main()