"""
OCR output normalizer.
Normalizes raw OCR text to valid meter reading according to meter profile.
"""
import re
import logging
from typing import Optional, Tuple

from app.models.meter_profile import MeterProfile

logger = logging.getLogger(__name__)


def normalize_ocr(
    raw_text: str,
    profile: MeterProfile,
    prev_value: Optional[int] = None,
) -> Tuple[Optional[str], str]:
    """
    Normalize raw OCR output to valid meter reading.
    
    Args:
        raw_text: Raw text from OCR engine
        profile: Meter profile with validation rules
        prev_value: Previous reading value (for validation)
        
    Returns:
        Tuple of (normalized_reading_str, status)
        - normalized_reading_str: Formatted reading string (e.g., "01814511") or None
        - status: Status code ("ok", "no_digits", "bad_length", "decreased", "too_large_jump")
    """
    if not raw_text:
        return None, "no_digits"
    
    # 1. Extract all digits from raw text
    digits = re.findall(r"\d", raw_text)
    if not digits:
        logger.warning(f"No digits found in raw text: {raw_text}")
        return None, "no_digits"
    
    s = "".join(digits)
    logger.info(f"Extracted digits from OCR: {s} (from raw: {raw_text[:50]})")
    
    L = profile.total_digits
    
    # 2. Simplified approach: if we have a good crop, OCR should see mostly the reading
    # Just find the longest sequence starting with 0 that's close to expected length
    if len(s) > L:
        # Simple strategy: find sequences starting with 0, prefer longer ones in middle of text
        best_candidate = None
        best_score = 0
        
        # Look for sequences of L-1, L, or L+1 digits starting with 0
        for target_length in [L-1, L, L+1]:
            if target_length < 4:
                continue
            for start in range(len(s) - target_length + 1):
                candidate = s[start:start + target_length]
                
                if not candidate.startswith('0'):
                    continue
                
                # Skip if it's clearly a year (2012, 2021, etc.)
                if len(candidate) >= 4:
                    try:
                        first_four = int(candidate[:4])
                        if 1900 <= first_four <= 2100:
                            continue  # Skip years completely
                    except ValueError:
                        pass
                
                # Simple scoring: prefer exact length, prefer middle position
                score = 0
                if len(candidate) == L:
                    score = 100  # Perfect length
                elif len(candidate) == L - 1:
                    score = 80   # Can pad
                elif len(candidate) == L + 1:
                    score = 70   # Can trim
                
                # Prefer sequences in the middle 60% of text (avoid very start and very end)
                position_ratio = start / len(s) if len(s) > 0 else 0.5
                if 0.2 <= position_ratio <= 0.8:
                    score += 20  # Bonus for middle position
                elif position_ratio < 0.2:
                    score += 10  # OK if early
                else:
                    score -= 10  # Penalty for very late (likely serial number)
                
                if score > best_score:
                    best_score = score
                    best_candidate = (candidate, start, len(candidate))
        
        if best_candidate:
            candidate, start_pos, cand_len = best_candidate
            
            # Adjust length if needed
            if cand_len == L - 1:
                # Try to get next digit from original string
                if start_pos + cand_len < len(s):
                    next_digit = s[start_pos + cand_len]
                    s = candidate + next_digit
                    logger.info(f"Padded {candidate} with next digit '{next_digit}': {s}")
                else:
                    s = candidate + '0'
                    logger.info(f"Padded {candidate} with '0': {s}")
            elif cand_len == L + 1:
                s = candidate[:L]
                logger.info(f"Trimmed {candidate} to {s}")
            else:
                s = candidate
                logger.info(f"Selected: {s} (position {start_pos})")
        else:
            # Fallback: just take first L digits starting with 0, or last L digits
            found = False
            for start in range(len(s) - L + 1):
                candidate = s[start:start + L]
                if candidate.startswith('0'):
                    s = candidate
                    logger.info(f"Fallback: first sequence starting with 0: {s}")
                    found = True
                    break
            if not found:
                s = s[-L:]
                logger.info(f"Last fallback: last {L} digits: {s}")
    
    # 3. If shorter and leading zeros are allowed - pad with zeros
    if len(s) < L and profile.allow_leading_zero:
        s = s.zfill(L)
        logger.info(f"Padded with leading zeros: {s}")
    
    # 4. Final length check
    if len(s) != L:
        logger.warning(
            f"Length mismatch: got {len(s)} digits, expected {L}. "
            f"Value: {s}, raw: {raw_text[:50]}"
        )
        return None, "bad_length"
    
    # 5. Convert to integer for validation checks
    try:
        value_int = int(s)
    except ValueError:
        logger.error(f"Failed to convert to int: {s}")
        return None, "bad_length"
    
    # 6. Check minimum value
    if value_int < profile.min_value:
        logger.warning(f"Value {value_int} below minimum {profile.min_value}")
        return None, "below_minimum"
    
    # 7. Check against previous reading (if available)
    if prev_value is not None:
        # Check for decrease
        if value_int < prev_value:
            logger.warning(
                f"Reading decreased: {value_int} < {prev_value} "
                f"(diff: {prev_value - value_int})"
            )
            return s, "decreased"  # Return value but mark as suspicious
        
        # Check for abnormal jump
        if prev_value > 0:
            jump = value_int - prev_value
            jump_percent = (jump / prev_value) * 100
            
            if jump_percent > profile.max_jump_percent:
                logger.warning(
                    f"Abnormal jump: {value_int} from {prev_value} "
                    f"({jump_percent:.1f}% > {profile.max_jump_percent}%)"
                )
                return s, "too_large_jump"  # Return value but mark as suspicious
    
    logger.info(f"Normalized reading: {s} (int: {value_int})")
    return s, "ok"

