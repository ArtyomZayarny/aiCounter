"""
Meter profile models for different utility meter types.
Defines validation rules and formatting for each meter type.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class MeterProfile:
    """Profile defining validation rules for a specific meter type."""
    
    name: str
    resource: str  # "gas", "water", "electricity"
    total_digits: int  # Total number of digits (e.g., 8 for "01814511")
    integer_digits: int  # Digits before decimal point (e.g., 5 for "01814")
    fraction_digits: int  # Digits after decimal point (e.g., 3 for "511")
    allow_leading_zero: bool = True  # Whether leading zeros are allowed
    min_value: int = 0
    max_jump_percent: int = 300  # Maximum allowed jump compared to previous reading
    
    def format_reading(self, value_int: int) -> str:
        """Format integer reading as string with leading zeros if needed."""
        if self.allow_leading_zero:
            return str(value_int).zfill(self.total_digits)
        return str(value_int)
    
    def parse_reading(self, value_str: str) -> Optional[int]:
        """Parse string reading to integer."""
        try:
            return int(value_str)
        except (ValueError, TypeError):
            return None


# Default profiles for common meter types
DEFAULT_PROFILES = {
    "gas": MeterProfile(
        name="Gas Meter (Default)",
        resource="gas",
        total_digits=8,
        integer_digits=5,
        fraction_digits=3,
        allow_leading_zero=True,
        min_value=0,
        max_jump_percent=300,
    ),
    "water": MeterProfile(
        name="Water Meter (Default)",
        resource="water",
        total_digits=8,
        integer_digits=5,
        fraction_digits=3,
        allow_leading_zero=True,
        min_value=0,
        max_jump_percent=300,
    ),
    "electricity": MeterProfile(
        name="Electricity Meter (Default)",
        resource="electricity",
        total_digits=7,
        integer_digits=6,
        fraction_digits=1,
        allow_leading_zero=True,
        min_value=0,
        max_jump_percent=300,
    ),
}


def get_profile(utility_type: str) -> MeterProfile:
    """Get meter profile for given utility type."""
    utility_type_lower = utility_type.lower()
    return DEFAULT_PROFILES.get(utility_type_lower, DEFAULT_PROFILES["gas"])

