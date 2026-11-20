"""
Tests for arm_driver.py's unit conversion, limit checks, and commands to the arm

- Every test drives a FakeController, never the real arm (no USB, no hidapi device needed)
- The fake mirrors the xarm library's interface, and rejects float positions the way the real one would misread them
"""

# ---------------------------------
# Imports
# ---------------------------------

# Standard library
import math

# Test framework
import pytest

# Code under test
import capstone.arm_controller.arm_driver as arm_driver
from capstone.arm_controller.arm_driver import (
    ArmCommunicationError,
    ArmDriver,
    ArmNotFoundError,
    ServoLimitError,
    UnknownServoError
)




# ---------------------------------
# Test helpers
# ---------------------------------

class FakeController:
    """
    Stand-in for xarm.Controller that records every command instead of moving anything

    - All six servos start at center (500 units)
    - setPosition jumps the stored positions straight to their targets
    - failReads and batteryVolts simulate a board that stops answering
    """

    def __init__(self) -> None:
        self.positions = {servoId: 500 for servoId in range(1, 7)}
        self.moves = []  # (targets, duration, wait) per setPosition call
        self.servoOffCalls = []  # The servos argument per servoOff call
        self.batteryVolts = 7.5
        self.failReads = False

    def setPosition(self, servos, position=None, duration=1000, wait=False):
        # Real library reads a float as its own ±125 degree mode, so floats here mean a driver bug
        for servoId, units in servos:
            if (not isinstance(units, int)):
                raise TypeError(f"Servo {servoId} got a non-int position ({units!r})")

        self.moves.append((list(servos), duration, wait))
        for servoId, units in servos:
            self.positions[servoId] = units

    def getPosition(self, servos, degrees=False):
        if (self.failReads):
            raise Exception("Function 'getPosition' recv error.")  # Same bare Exception the real library raises

        return self.positions[servos]

    def servoOff(self, servos=None):
        self.servoOffCalls.append(servos)

    def getBatteryVoltage(self):
        return self.batteryVolts


@pytest.fixture
def fake():
    """
    A fresh FakeController per test
    """

    return FakeController()


@pytest.fixture
def arm(fake):
    """
    An ArmDriver wrapped around the test's FakeController
    """

    return ArmDriver(fake)




# ---------------------------------
# Unit conversion tests
# ---------------------------------

def test_degrees_to_units_maps_center_and_both_ends():
    """
    0 degrees should be center (500), and ±120 degrees the two ends of the 0-1000 range
    """

    assert (arm_driver.degreesToUnits(0.0) == 500)
    assert (arm_driver.degreesToUnits(120.0) == 1000)
    assert (arm_driver.degreesToUnits(-120.0) == 0)
    assert (arm_driver.degreesToUnits(30.0) == 625)


def test_degrees_to_units_always_returns_an_int():
    """
    A fractional result should round to an int unit (a float would hit the library's degree mode)
    """

    units = arm_driver.degreesToUnits(10.1)

    assert (isinstance(units, int))
    assert (units == 542)


def test_degrees_to_units_rejects_angles_past_the_physical_range():
    """
    Anything past ±120 degrees has no position unit
    """

    with pytest.raises(ValueError):
        arm_driver.degreesToUnits(120.5)


def test_units_to_degrees_round_trips():
    """
    Converting units to degrees and back should land on the same unit
    """

    for units in (0, 1, 250, 500, 625, 999, 1000):
        assert (arm_driver.degreesToUnits(arm_driver.unitsToDegrees(units)) == units)




# ---------------------------------
# moveServos tests
# ---------------------------------

def test_move_servos_sends_one_packet_in_units(arm, fake):
    """
    A multi-servo move should go out as a single setPosition call, in units, with the default duration
    """

    arm.moveServos({6: 30.0, 4: -12.0})

    assert (fake.moves == [([(6, 625), (4, 450)], 1000, True)])


def test_move_servos_passes_duration_and_wait_through(arm, fake):
    """
    A caller-chosen duration and wait=False should reach the controller unchanged
    """

    arm.moveServos({1: 10.0}, durationMs=2500, wait=False)

    assert (fake.moves[0][1:] == (2500, False))


def test_move_servo_is_a_single_target_move_servos(arm, fake):
    """
    moveServo should send the same packet moveServos would for one target
    """

    arm.moveServo(5, -30.0, durationMs=800)

    assert (fake.moves == [([(5, 375)], 800, True)])


def test_move_servos_rejects_a_target_past_the_default_limit(arm, fake):
    """
    The default limit is ±110 degrees, so 110 should go out and 115 be refused (even though the servo can reach 115)
    """

    arm.moveServo(6, 110.0)

    with pytest.raises(ServoLimitError):
        arm.moveServo(6, 115.0)

    assert (len(fake.moves) == 1)


def test_default_limits_stay_inside_the_physical_range():
    """
    Every servo's limits should sit inside ±120 degrees (a wider limit would pass the check, then fail the unit conversion)
    """

    for low, high in arm_driver.SERVO_LIMITS.values():
        assert (-arm_driver.DEGREES_MAX <= low < high <= arm_driver.DEGREES_MAX)


def test_move_servos_sends_nothing_if_any_target_is_bad(arm, fake, monkeypatch):
    """
    One out-of-limit target should reject the whole move, including the valid targets
    """

    monkeypatch.setitem(arm_driver.SERVO_LIMITS, 6, (-90.0, 90.0))

    with pytest.raises(ServoLimitError):
        arm.moveServos({5: 10.0, 6: 95.0})

    assert (fake.moves == [])


def test_move_servos_enforces_narrowed_limits(arm, fake, monkeypatch):
    """
    A narrowed limit should allow its own edge but nothing past it
    """

    monkeypatch.setitem(arm_driver.SERVO_LIMITS, 1, (-20.0, 45.0))

    arm.moveServo(1, 45.0)

    with pytest.raises(ServoLimitError):
        arm.moveServo(1, -20.5)

    assert (len(fake.moves) == 1)


def test_move_servos_rejects_nan(arm, fake):
    """
    A NaN target (ex. from a bad calculation upstream) should be refused, not sent
    """

    with pytest.raises(ServoLimitError):
        arm.moveServo(3, math.nan)

    assert (fake.moves == [])


def test_move_servos_rejects_unknown_servo_ids(arm, fake):
    """
    Servo IDs outside 1-6 should raise before anything is sent
    """

    for servoId in (0, 7):
        with pytest.raises(UnknownServoError):
            arm.moveServo(servoId, 0.0)

    assert (fake.moves == [])


def test_move_servos_rejects_out_of_range_durations(arm, fake):
    """
    Durations must fit the move packet's 16-bit field, and be at least 1 ms
    """

    for durationMs in (0, -5, 65536):
        with pytest.raises(ValueError):
            arm.moveServo(1, 0.0, durationMs=durationMs)

    assert (fake.moves == [])


def test_move_servos_rejects_an_empty_move(arm):
    """
    No targets means nothing to send
    """

    with pytest.raises(ValueError):
        arm.moveServos({})


def test_move_servos_wraps_link_failures(arm, fake, monkeypatch):
    """
    A failed write (ex. USB unplugged mid-session) should come back as ArmCommunicationError
    """

    def unplugged(*args, **kwargs):
        raise OSError("write error")

    monkeypatch.setattr(fake, "setPosition", unplugged)

    with pytest.raises(ArmCommunicationError):
        arm.moveServo(1, 0.0)




# ---------------------------------
# Reading tests
# ---------------------------------

def test_read_position_returns_degrees(arm, fake):
    """
    A raw unit read should come back as degrees from center
    """

    fake.positions[2] = 375

    assert (arm.readPosition(2) == -30.0)


def test_read_positions_covers_all_six_servos(arm, fake):
    """
    readPositions should return every servo, keyed 1-6
    """

    fake.positions[6] = 1000

    positions = arm.readPositions()

    assert (sorted(positions) == [1, 2, 3, 4, 5, 6])
    assert (positions[6] == 120.0)
    assert (positions[1] == 0.0)


def test_read_position_rejects_unknown_servo_ids(arm):
    """
    Reading a servo that doesn't exist should raise, not ask the board
    """

    with pytest.raises(UnknownServoError):
        arm.readPosition(9)


def test_read_failure_raises_communication_error(arm, fake):
    """
    The library's bare Exception on a bad reply should come back as ArmCommunicationError
    """

    fake.failReads = True

    with pytest.raises(ArmCommunicationError, match="reading servo 4"):
        arm.readPosition(4)


def test_battery_voltage_returns_volts(arm, fake):
    """
    The board's voltage should pass straight through
    """

    fake.batteryVolts = 7.42

    assert (arm.readBatteryVoltage() == 7.42)


def test_battery_voltage_raises_when_the_board_doesnt_answer(arm, fake):
    """
    The library returns None on no reply, which should become an ArmCommunicationError
    """

    fake.batteryVolts = None

    with pytest.raises(ArmCommunicationError):
        arm.readBatteryVoltage()




# ---------------------------------
# Stopping tests
# ---------------------------------

def test_stop_holds_every_servo_where_it_is(arm, fake):
    """
    stop should send each servo's current units straight back, with the short hold duration and no wait
    """

    fake.positions.update({1: 500, 2: 400, 3: 600, 4: 700, 5: 300, 6: 812})

    held = arm.stop()

    assert (fake.moves == [([(1, 500), (2, 400), (3, 600), (4, 700), (5, 300), (6, 812)], arm_driver.HOLD_MOVE_MS, False)])
    assert (held[2] == -24.0)


def test_stop_ignores_limits(arm, fake, monkeypatch):
    """
    A servo pushed past its limit by hand should still be held, not refused
    """

    monkeypatch.setitem(arm_driver.SERVO_LIMITS, 6, (-10.0, 10.0))
    fake.positions[6] = 900

    arm.stop()

    assert (fake.moves[0][0][5] == (6, 900))


def test_servos_off_defaults_to_all_six(arm, fake):
    """
    No argument should pass None through (the library turns that into all six)
    """

    arm.servosOff()

    assert (fake.servoOffCalls == [None])


def test_servos_off_passes_chosen_servos(arm, fake):
    """
    A list of IDs should reach the controller unchanged
    """

    arm.servosOff([1, 2])

    assert (fake.servoOffCalls == [[1, 2]])


def test_servos_off_rejects_unknown_servo_ids(arm, fake):
    """
    One bad ID should raise before any servo is turned off
    """

    with pytest.raises(UnknownServoError):
        arm.servosOff([1, 8])

    assert (fake.servoOffCalls == [])




# ---------------------------------
# openArm tests
# ---------------------------------

def test_open_arm_wraps_the_usb_controller(monkeypatch):
    """
    openArm should open the board as "USB" plus the serial number, if one was given
    """

    opened = []

    def recordPort(port):
        opened.append(port)
        return FakeController()

    monkeypatch.setattr(arm_driver.xarm, "Controller", recordPort)

    assert (isinstance(arm_driver.openArm(), ArmDriver))
    arm_driver.openArm("ABC123")

    assert (opened == ["USB", "USBABC123"])


def test_open_arm_raises_not_found_with_instructions(monkeypatch):
    """
    hidapi's "open failed" should become ArmNotFoundError, with what to check
    """

    def noBoard(port):
        raise OSError("open failed")

    monkeypatch.setattr(arm_driver.xarm, "Controller", noBoard)

    with pytest.raises(ArmNotFoundError, match="micro-USB"):
        arm_driver.openArm()