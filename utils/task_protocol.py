from dataclasses import dataclass, field
from typing import List, Optional
import time

@dataclass
class TaskProtocol:
    x_columns: List[str] = field(default_factory=list)
    y_columns: List[str] = field(default_factory=list)
    one_hot_encoded_x: List[str] = field(default_factory=list)
    one_hot_encoded_y: List[str] = field(default_factory=list)
    start_time: Optional[float] = None
    additional_info: dict = field(default_factory=dict)  # libovolná data navíc