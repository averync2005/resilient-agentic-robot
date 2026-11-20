"""
Arm controller: everything that drives the Hiwonder xArm 1S robot arm

Modules:
- `arm_driver.py` -- Moves, reads, and stops the arm's six servos over USB (degrees in and out, per-servo limits)
- `poses.py` -- Named poses and button sequences in poses.yaml, checked against the servo limits on load
- `arm_actions.py` -- The named-pose interlock the edge agent runs ("POSE <name>" / "PRESS <name>" only for approved names)
- `arm_cli.py` -- The `arm` command (status, move, pose, press, teach)

Planned:
- MCP tools and a ROS 2 Action node over the same actions
"""