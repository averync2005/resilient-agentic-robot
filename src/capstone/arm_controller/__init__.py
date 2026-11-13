"""
Arm controller: everything that drives the Hiwonder xArm 1S robot arm

Modules:
- `arm_driver.py`  Moves, reads, and stops the arm's six servos over USB (degrees in and out, per-servo limits)

Planned:
- The edge agent's MCP tools, built on top of `arm_driver.py`
"""