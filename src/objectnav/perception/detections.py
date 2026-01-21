from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass(frozen=True) # Immutable (read-only) dataclass
class Detection:
    cls: int
    conf: float
    xyxy: Tuple[float, float, float, float]  # x1,y1,x2,y2
    track_id: Optional[int] = None
