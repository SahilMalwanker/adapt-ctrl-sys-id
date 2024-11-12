"""adapt-ctrl-sys-id: from measured data to adaptive control and a real-time digital twin.

The package bundles one end-to-end workflow for two rotary mechatronic systems:

* a DC servo with an inertia disk (static map, PT1 identification, RLS,
  frequency-response validation, cascade position control, adaptive control) and
* a propeller-driven pendulum with a movable slider (first-principles model,
  parameter estimation from hardware data, linearisation, digital 2-DOF PI-P
  cascade control and a TCP digital twin).
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"
