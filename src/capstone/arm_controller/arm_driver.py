"""
Driver for the Hiwonder xArm 1S robot arm, over the controller board's micro-USB port

Hardware:
- 6 bus servos (LX-15D / LX-225) daisy-chained to one controller board on the arm's bottom plate
- Servo IDs: 1 gripper, 2 wrist rotation, 3 wrist, 4 elbow, 5 shoulder, 6 base
- The board shows up as a USB HID device (VID 0x0483, PID 0x5750), not a serial port
- The servos need the board's own 7.5 V supply (the USB cable only carries data)

Units:
- Every angle going in or out of this driver is in degrees, with 0 at each servo's center
- The servos use position units from 0 to 1000 across 240 degrees (500 = center, 0.24 degrees per unit)
- Converted here, not by the xarm library (its float "degree mode" assumes 250 degrees across the range)

Safety:
- Every move is checked against SERVO_LIMITS before anything is sent (one bad target rejects the whole move)
- SERVO_LIMITS starts at ±110 degrees per servo (inside the ±120 physical range, until the real limits are measured on the arm)
- `stop()` holds every servo where it is right now (the board has no stop command for single moves)
- `servosOff()` cuts holding torque, so the arm goes limp and drops under its own weight

Library:
- Talks to the board through the `xarm` package (`xarm.Controller("USB")`, which needs `hidapi`)
- ArmDriver takes the controller as an argument, so tests pass in a fake one and never need the arm
- Run it with:
    from capstone.arm_controller.arm_driver import openArm
    arm = openArm()
    arm.moveServos({6: 30.0, 4: -15.0})
"""

# ---------------------------------
# Imports
# ---------------------------------

# Hiwonder's controller library (talks to the board over USB HID)
import xarm




# ---------------------------------
# Constants
# ---------------------------------

# Servos
SERVO_NAMES = {1: "gripper", 2: "wrist rotation", 3: "wrist", 4: "elbow", 5: "shoulder", 6: "base"}

# Position units (Hiwonder's servo spec)
UNITS_MIN = 0
UNITS_MAX = 1000
UNITS_CENTER = 500
DEGREES_ACROSS_RANGE = 240  # Full 0-1000 sweep
DEGREES_MAX = DEGREES_ACROSS_RANGE / 2  # 120 degrees either side of center

# Allowed range per servo in degrees (±110 for now, 10 degrees clear of each end - narrow each one after measuring on the arm)
SERVO_LIMITS = {
    1: (-110.0, 110.0),  # Gripper
    2: (-110.0, 110.0),  # Wrist rotation
    3: (-110.0, 110.0),  # Wrist
    4: (-110.0, 110.0),  # Elbow
    5: (-110.0, 110.0),  # Shoulder
    6: (-110.0, 110.0)  # Base
}

# Move timing
DEFAULT_MOVE_MS = 1000  # Time a move takes to reach its target, unless the caller says otherwise
MAX_MOVE_MS = 65535  # Largest duration the move packet holds (16 bits)
HOLD_MOVE_MS = 50  # stop()'s hold move (target = current position, so nearly instant)




# ---------------------------------
# Errors
# ---------------------------------

class ArmError(Exception):
    """
    Base class for every error this driver raises
    """


class ArmNotFoundError(ArmError):
    """
    The arm's controller board couldn't be opened over USB
    """


class ArmCommunicationError(ArmError):
    """
    The board was open, but a command or read failed (unplugged mid-session, no reply, garbled reply)
    """


class UnknownServoError(ArmError):
    """
    A servo ID outside 1-6
    """


class ServoLimitError(ArmError):
    """
    A target outside that servo's range in SERVO_LIMITS
    """




# ---------------------------------
# Unit conversion
# ---------------------------------

def degreesToUnits(degrees: float) -> int:
    """
    Convert an angle from center into the servo's 0-1000 position units

    Args:
        degrees: Angle from the servo's center, -120 to 120

    Returns:
        The nearest position unit (always an int, since the xarm library treats floats as its own degree mode)

    Raises:
        ValueError: If degrees is outside the servo's physical range
    """

    if (not (-DEGREES_MAX <= degrees <= DEGREES_MAX)):
        raise ValueError(f"{degrees} degrees is outside the servo's range (-{DEGREES_MAX} to {DEGREES_MAX})")

    return round(UNITS_CENTER + (degrees * (UNITS_MAX - UNITS_MIN) / DEGREES_ACROSS_RANGE))


def unitsToDegrees(units: int) -> float:
    """
    Convert the servo's 0-1000 position units into an angle from center

    Args:
        units: A position read from the servo

    Returns:
        Degrees from center (0.24 degree steps)
    """

    return ((units - UNITS_CENTER) * DEGREES_ACROSS_RANGE / (UNITS_MAX - UNITS_MIN))




# ---------------------------------
# Checking requests before sending
# ---------------------------------

def checkServoId(servoId: int) -> None:
    """
    Make sure a servo ID is one of the arm's six

    Args:
        servoId: The ID to check

    Raises:
        UnknownServoError: If servoId isn't 1-6
    """

    if (servoId not in SERVO_NAMES):
        raise UnknownServoError(f"No servo {servoId} (the xArm 1S has servos 1-6)")


def checkTarget(servoId: int, degrees: float) -> None:
    """
    Make sure a target angle is inside that servo's allowed range

    Args:
        servoId: The servo being moved
        degrees: The requested angle from center

    Raises:
        UnknownServoError: If servoId isn't 1-6
        ServoLimitError: If degrees is outside SERVO_LIMITS for that servo (NaN included)
    """

    checkServoId(servoId)

    low, high = SERVO_LIMITS[servoId]

    # Written as "not inside" so NaN fails too (every comparison with NaN is False)
    if (not (low <= degrees <= high)):
        raise ServoLimitError(f"Servo {servoId} ({SERVO_NAMES[servoId]}) can't go to {degrees} degrees (allowed: {low} to {high})")


def checkDuration(durationMs: int) -> None:
    """
    Make sure a move duration fits in the move packet

    Args:
        durationMs: Requested move time in milliseconds

    Raises:
        ValueError: If durationMs is below 1 or above MAX_MOVE_MS
    """

    if (not (1 <= durationMs <= MAX_MOVE_MS)):
        raise ValueError(f"Move duration must be 1-{MAX_MOVE_MS} ms (got {durationMs})")




# ---------------------------------
# Talking to the board
# ---------------------------------

def callController(description: str, function, *args, **kwargs):
    """
    Run one controller call, turning any failure into an ArmCommunicationError

    - The xarm library raises bare Exception on a bad reply and IndexError on no reply, so there's no narrower type to catch
    - ValueError passes through untouched (that's a bad argument from this driver, not a link problem)

    Args:
        description: What the call was doing, for the error message (ex. "reading servo 3")
        function: The controller method to call
        *args: Passed through to function
        **kwargs: Passed through to function

    Returns:
        Whatever function returns

    Raises:
        ArmCommunicationError: If the call fails for any reason other than ValueError
    """

    try:
        return function(*args, **kwargs)
    except ValueError:
        raise
    except Exception as error:
        raise ArmCommunicationError(f"Lost contact with the arm while {description} ({type(error).__name__}: {error})") from error




# ---------------------------------
# Arm driver
# ---------------------------------

class ArmDriver:
    """
    Moves, reads, and stops the xArm 1S through an open xarm.Controller (or a fake one in tests)

    - Every method takes and returns degrees from center
    - Moves are checked against SERVO_LIMITS before anything is sent
    """

    def __init__(self, controller) -> None:
        """
        Wrap an already-open controller

        Args:
            controller: An xarm.Controller, or any object with the same setPosition / getPosition / servoOff / getBatteryVoltage methods
        """

        self.controller = controller

    def moveServos(self, targets: dict[int, float], durationMs: int = DEFAULT_MOVE_MS, wait: bool = True) -> None:
        """
        Move one or more servos together, all arriving at the same time

        Args:
            targets: Degrees from center per servo ID (ex. {6: 30.0, 4: -15.0})
            durationMs: How long the move takes (longer = slower and gentler)
            wait: Block until the move should be done (the board doesn't report when it actually finishes)

        Raises:
            ValueError: If targets is empty, or durationMs is out of range
            UnknownServoError: If any servo ID isn't 1-6
            ServoLimitError: If any target is outside its servo's limits
        """

        if (not targets):
            raise ValueError("No servos to move")

        # Checking everything before sending anything (one bad target rejects the whole move)
        checkDuration(durationMs)
        for servoId, degrees in targets.items():
            checkTarget(servoId, degrees)

        unitTargets = [(servoId, degreesToUnits(degrees)) for servoId, degrees in targets.items()]

        callController("moving servos", self.controller.setPosition, unitTargets, duration=durationMs, wait=wait)

    def moveServo(self, servoId: int, degrees: float, durationMs: int = DEFAULT_MOVE_MS, wait: bool = True) -> None:
        """
        Move a single servo (shorthand for moveServos with one target)

        Args:
            servoId: The servo to move, 1-6
            degrees: Target angle from center
            durationMs: How long the move takes
            wait: Block until the move should be done

        Raises:
            ValueError: If durationMs is out of range
            UnknownServoError: If servoId isn't 1-6
            ServoLimitError: If degrees is outside that servo's limits
        """

        self.moveServos({servoId: degrees}, durationMs, wait)

    def readPosition(self, servoId: int) -> float:
        """
        Read where one servo is right now

        Args:
            servoId: The servo to read, 1-6

        Returns:
            Its angle in degrees from center

        Raises:
            UnknownServoError: If servoId isn't 1-6
        """

        checkServoId(servoId)

        units = callController(f"reading servo {servoId}", self.controller.getPosition, servoId)

        return unitsToDegrees(units)

    def readPositions(self) -> dict[int, float]:
        """
        Read where every servo is right now

        - One read per servo, not the library's batch read (that one silently leaves missing servos at center)

        Returns:
            Degrees from center per servo ID, 1-6
        """

        return {servoId: self.readPosition(servoId) for servoId in SERVO_NAMES}

    def stop(self) -> dict[int, float]:
        """
        Hold every servo where it is right now, cutting short any move in progress

        - No limit check (a servo pushed past its limit by hand still has to hold where it is)

        Returns:
            The angles each servo is now holding, in degrees from center
        """

        # Reading raw units, not degrees (sending back exactly what was read, no rounding)
        heldUnits = [(servoId, callController(f"reading servo {servoId}", self.controller.getPosition, servoId)) for servoId in SERVO_NAMES]

        callController("holding position", self.controller.setPosition, heldUnits, duration=HOLD_MOVE_MS, wait=False)

        return {servoId: unitsToDegrees(units) for servoId, units in heldUnits}

    def servosOff(self, servoIds: list[int] | None = None) -> None:
        """
        Cut holding torque so the servos go limp (the arm drops under its own weight - support it first)

        Args:
            servoIds: Which servos to turn off (all six if None)

        Raises:
            UnknownServoError: If any servo ID isn't 1-6
        """

        if (servoIds is not None):
            for servoId in servoIds:
                checkServoId(servoId)

        callController("turning servos off", self.controller.servoOff, servoIds)

    def readBatteryVoltage(self) -> float:
        """
        Read the controller board's supply voltage

        - The board beeps a low-voltage alarm below about 6.4 V

        Returns:
            Supply voltage in volts

        Raises:
            ArmCommunicationError: If the board didn't answer
        """

        volts = callController("reading the supply voltage", self.controller.getBatteryVoltage)

        # The library returns None instead of raising when the reply doesn't come back
        if (volts is None):
            raise ArmCommunicationError("The arm didn't answer the supply voltage read")

        return volts




# ---------------------------------
# Connecting
# ---------------------------------

def openArm(serialNumber: str | None = None) -> ArmDriver:
    """
    Open the arm's controller board over USB and wrap it in an ArmDriver

    Args:
        serialNumber: The board's USB serial number (only needed with more than one board plugged in)

    Returns:
        An ArmDriver ready to move the arm

    Raises:
        ArmNotFoundError: If the board can't be opened (with what to check)
    """

    try:
        controller = xarm.Controller("USB" + (serialNumber or ""))
    except OSError as error:
        raise ArmNotFoundError(
            f"Can't open the xArm over USB ({error}). "
            "Check that the micro-USB cable is plugged into the board on the arm's bottom plate, "
            "the board's switch is on, and nothing else (ex. Hiwonder's PC software) has it open. "
            "On Linux, the board's USB device also needs read/write permission for your user (see the udev rule in README.md)."
        ) from error

    return ArmDriver(controller)