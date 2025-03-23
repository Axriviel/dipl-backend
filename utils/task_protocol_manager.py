from dataclasses import dataclass, field
from typing import List, Optional
import time

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict

@dataclass
class TaskProtocol:
    x_columns: List[str] = field(default_factory=list)
    y_columns: List[str] = field(default_factory=list)
    one_hot_encoded_x: List[str] = field(default_factory=list)
    one_hot_encoded_y: List[str] = field(default_factory=list)
    start_time: Optional[float] = None
    finished_at: Optional[str] = None
    additional_info: Dict[str, any] = field(default_factory=dict)

    def log(self, key: str, value):
        """Univerzální metoda pro logování hodnot – uloží do správného atributu nebo do additional_info."""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.additional_info[key] = value

    def log_dict(self, data: Dict[str, any]):
        """Umožní zalogovat víc klíčů najednou (např. columns_info dict)."""
        for key, value in data.items():
            self.log(key, value)

    def to_dict(self) -> dict:
        """Vrací celý log jako serializovatelný slovník."""
        return asdict(self)


class TaskProtocolManager:
    def __init__(self):
        self.logs = {}

    def init_user(self, user_id):
        if user_id not in self.logs:
            self.logs[user_id] = TaskProtocol()

    def get_log(self, user_id) -> Optional[TaskProtocol]:
        return self.logs.get(user_id)

    def reset_user(self, user_id):
        self.logs.pop(user_id, None)

    def log_dict(self, user_id, data: Dict[str, any]):
        self.init_user(user_id)
        self.logs[user_id].log_dict(data)

    def log_item(self, user_id, key: str, value):
        self.init_user(user_id)
        self.logs[user_id].log(key, value)

    def get_log_as_dict(self, user_id) -> dict:
        return self.logs.get(user_id).to_dict() if user_id in self.logs else {}

task_protocol_manager = TaskProtocolManager()