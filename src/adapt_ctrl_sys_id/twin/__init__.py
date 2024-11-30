"""Real-time digital twin of the propeller pendulum (TCP server + client).

The wire protocol is line based and deliberately simple so that MATLAB,
Python or a microcontroller can talk to the twin::

    client -> twin : START
    twin -> client : <PP><t>0.0100 </t><PA>12.3456 </PA><PAD>-1.2345 </PAD></PP>
    client -> twin : 37.500000;0.000000          (rotor command [%]; slider [mm])
    ...
    client -> twin : EXIT
"""

from .client import TwinClient, run_closed_loop
from .physics import CALIBRATED, PendulumTwin, TwinConfig
from .protocol import DEFAULT_PORT, format_command, format_state, parse_command, parse_state
from .server import TwinServer
from .timing import sampling_report

__all__ = [
    "CALIBRATED",
    "DEFAULT_PORT",
    "PendulumTwin",
    "TwinClient",
    "TwinConfig",
    "TwinServer",
    "format_command",
    "format_state",
    "parse_command",
    "parse_state",
    "run_closed_loop",
    "sampling_report",
]
