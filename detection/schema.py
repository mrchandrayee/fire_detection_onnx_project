from dataclasses import dataclass
from typing import Tuple

@dataclass
class Det:
    """One detected object. Box coordinates are x1,y1,x2,y2 in source-frame pixels."""
    xyxy: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
