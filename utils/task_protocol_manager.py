from dataclasses import dataclass, field
from typing import List, Optional
import time

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict

@dataclass
class ModelLog:
    model_id: Optional[str] = None
    architecture: Dict[str, any] = field(default_factory=dict)
    parameters: Dict[str, any] = field(default_factory=dict)
    results: Optional[float] = field(default_factory=float)
    history: List[str] = field(default_factory=list)
    timestamp: Optional[str] = None

@dataclass
class EpochLog:
    epoch_number: int
    model_info: Optional[Dict[str, any]] = field(default_factory=dict)
    limits: List[Dict[str, any]] = field(default_factory=list)
    # results: Optional[Dict[str, any]] = field(default_factory=dict)
    timestamp: Optional[str] = None
    models: List[ModelLog] = field(default_factory=list)

    def add_model(self, architecture=None, parameters=None, history=None, results=None, model_id=None):
        model_log = ModelLog(
            model_id=model_id,
            architecture=architecture or {},
            parameters=parameters or {},
            history=history or [],
            results=results or 0,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
        self.models.append(model_log)

@dataclass
class TaskProtocol:
    x_columns: List[str] = field(default_factory=list)
    y_columns: List[str] = field(default_factory=list)
    one_hot_encoded_x: List[str] = field(default_factory=list)
    one_hot_encoded_y: List[str] = field(default_factory=list)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    limit_growth: Optional[str] = None
    additional_info: Dict[str, any] = field(default_factory=dict)
    epochs: List[EpochLog] = field(default_factory=list)

    def log(self, key: str, value):
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.additional_info[key] = value

    def get_or_create_epoch(self, epoch_number: int) -> EpochLog:
        for epoch in self.epochs:
            if epoch.epoch_number == epoch_number:
                return epoch
        new_epoch = EpochLog(epoch_number=epoch_number, timestamp=time.strftime('%Y-%m-%d %H:%M:%S'))
        self.epochs.append(new_epoch)
        return new_epoch

    def log_model_to_epoch(self, epoch_number: int, architecture=None, parameters=None, history=None, results=None, model_id=None):
        epoch = self.get_or_create_epoch(epoch_number)
        epoch.add_model(
            architecture=architecture,
            parameters=parameters,
            results=results,
            history=history,
            model_id=model_id
        )

    def log_dict(self, data: Dict[str, any]):
        for key, value in data.items():
            self.log(key, value)

    def to_dict(self) -> dict:
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