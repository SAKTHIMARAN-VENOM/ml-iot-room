"""
pir_tracker.py
==============
Separate, modular PIR Motion Tracking and Combined Occupancy Heuristic Layer.

Key Principles:
1. Genuine PIR HIGH signals register motion events with timestamp tracking.
2. PIR LOW does NOT imply the room is empty (occupants sit quietly or sleep).
3. Distinguishes transient motion events from sustained room occupancy.
4. Combines the 3-feature Environmental ML prediction with PIR state using
   an explicit, customizable decision matrix.
5. Heuristic disclaimer: This rule layer is an empirical heuristic, not a
   scientifically validated model until verified with physical hardware data.
"""

import time
from typing import Dict, Any, Optional


class PIRMotionTracker:
    def __init__(self, default_timeout_seconds: int = 300):
        """
        Initialize the PIR motion tracker.
        :param default_timeout_seconds: Time window (in seconds) after a motion event
                                       during which presence is considered active.
                                       Default is 300s (5 minutes).
        """
        self.timeout_seconds: int = default_timeout_seconds
        self.last_motion_timestamp: Optional[float] = None
        self.last_pin_state: int = 0  # 0 = LOW, 1 = HIGH
        self.last_source: str = "init"
        self.total_motion_events: int = 0

    def register_signal(self, pin_state: int, source: str = "manual_test") -> Dict[str, Any]:
        """
        Register a PIR pin state signal (0 or 1).
        If pin_state is 1 (HIGH), marks a fresh motion event timestamp.
        """
        current_time = time.time()
        self.last_pin_state = 1 if pin_state else 0
        self.last_source = source

        if self.last_pin_state == 1:
            self.last_motion_timestamp = current_time
            self.total_motion_events += 1

        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        """Get the current PIR status, elapsed time since motion, and timeout state."""
        current_time = time.time()
        
        if self.last_motion_timestamp is None:
            seconds_since_motion = None
            is_within_timeout = False
        else:
            seconds_since_motion = round(current_time - self.last_motion_timestamp, 1)
            is_within_timeout = seconds_since_motion <= self.timeout_seconds

        return {
            'pin_state': self.last_pin_state,
            'pin_label': 'HIGH (Motion)' if self.last_pin_state == 1 else 'LOW (No Motion)',
            'last_motion_timestamp': self.last_motion_timestamp,
            'seconds_since_last_motion': seconds_since_motion,
            'timeout_seconds': self.timeout_seconds,
            'is_within_timeout': is_within_timeout,
            'total_motion_events': self.total_motion_events,
            'last_source': self.last_source,
        }

    def set_timeout(self, seconds: int) -> int:
        """Update configurable occupancy timeout."""
        if seconds < 10:
            raise ValueError("Timeout must be at least 10 seconds.")
        self.timeout_seconds = int(seconds)
        return self.timeout_seconds

    def reset(self):
        """Reset the motion history."""
        self.last_motion_timestamp = None
        self.last_pin_state = 0
        self.total_motion_events = 0


def combine_occupancy_decision(
    ml_prediction: int,
    ml_probability: float,
    pir_status: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Explicit heuristic decision logic combining Environmental ML prediction
    with PIR motion tracking.

    Truth Table / Decision Scenarios:
    -------------------------------------------------------------------------
    | Case | PIR Pin | Within Timeout | ML Predict | Combined Status             |
    |------|---------|----------------|------------|-----------------------------|
    | 1    | HIGH    | Yes            | Occupied   | Confirmed Occupied (Active) |
    | 2    | LOW     | Yes            | Occupied   | Confirmed Occupied (Recent) |
    | 3    | LOW     | No             | Occupied   | Occupied (Still Occupant)   |
    | 4    | HIGH    | Yes            | Unoccupied | Motion Alert / Transitory   |
    | 5    | LOW     | Yes            | Unoccupied | Cooling / Awaiting Timeout  |
    | 6    | LOW     | No             | Unoccupied | Confirmed Unoccupied        |
    -------------------------------------------------------------------------
    """
    is_motion_active = (pir_status.get('pin_state') == 1)
    is_within_timeout = pir_status.get('is_within_timeout', False)
    seconds_since = pir_status.get('seconds_since_last_motion')

    # Case 1: Active Motion + ML Occupied
    if is_motion_active and ml_prediction == 1:
        status = "Confirmed Occupied"
        badge_variant = "success"
        confidence = "Very High"
        code = "OCCUPIED_ACTIVE_MOTION"
        reasoning = (
            "Dual-verification: Physical infrared motion is actively detected and "
            "environmental sensors (Light, Temp, Humidity) align with an occupied room."
        )

    # Case 2: PIR within timeout (recent motion) + ML Occupied
    elif is_within_timeout and ml_prediction == 1:
        status = "Confirmed Occupied"
        badge_variant = "success"
        confidence = "High"
        code = "OCCUPIED_RECENT_MOTION"
        reasoning = (
            f"Physical motion was registered {seconds_since}s ago (within {pir_status['timeout_seconds']}s timeout), "
            "and environmental sensors confirm human presence."
        )

    # Case 3: PIR timed out (LOW) + ML Occupied (Classic 'seated occupant' scenario)
    elif not is_within_timeout and ml_prediction == 1:
        status = "Occupied (Still/Seated Presence)"
        badge_variant = "warning"
        confidence = "Moderate-High"
        code = "OCCUPIED_STILL_PRESENCE"
        reasoning = (
            "PIR sensor detects no current movement (occupant likely sitting still, reading, or working), "
            "but environmental conditions (light/temperature/humidity) strongly predict occupancy. "
            "PIR LOW does NOT mean room is empty."
        )

    # Case 4: PIR Active Motion + ML Unoccupied
    elif is_motion_active and ml_prediction == 0:
        status = "Transitory Motion Detected"
        badge_variant = "info"
        confidence = "Moderate"
        code = "MOTION_TRANSIENT"
        reasoning = (
            "Infrared motion triggered, but environmental conditions (e.g. dark room or baseline climate) "
            "do not yet reflect sustained human presence. Possible doorway walk-through or pet/curtain motion."
        )

    # Case 5: PIR within timeout + ML Unoccupied
    elif is_within_timeout and ml_prediction == 0:
        status = "Awaiting Verification"
        badge_variant = "info"
        confidence = "Low-Moderate"
        code = "MOTION_COOLING_OFF"
        reasoning = (
            f"Motion was registered {seconds_since}s ago, but environmental metrics remain at empty room levels. "
            "Holding status until timeout expires."
        )

    # Case 6: PIR Inactive + ML Unoccupied
    else:
        status = "Confirmed Unoccupied"
        badge_variant = "secondary"
        confidence = "High"
        code = "UNOCCUPIED_CLEAR"
        reasoning = (
            "Both layers agree: No recent motion detected past the timeout window, and "
            "environmental sensors confirm the room is vacant."
        )

    return {
        'combined_status': status,
        'status_code': code,
        'badge_variant': badge_variant,
        'confidence_level': confidence,
        'decision_reasoning': reasoning,
        'is_occupied_overall': (status.startswith("Confirmed Occupied") or "Still/Seated" in status),
        'disclaimer': (
            "Notice: This combined decision logic is an empirical sensor-fusion heuristic. "
            "It must be calibrated with live ESP32 deployment data before being claimed as scientifically validated."
        )
    }
