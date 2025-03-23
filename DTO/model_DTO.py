from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelDTO:
    id: int
    name: str
    metric_value: float
    watched_metric: str
    metric_values_history: Optional[List[float]]  # JSON pole pro historické hodnoty
    used_opt_method: str
    used_designer: str
    used_tags: str
    task_protocol: str
    dataset: str

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'metric_value': self.metric_value,
            'watched_metric': self.watched_metric,
            'metric_values_history': self.metric_values_history,
            'used_opt_method': self.used_opt_method,
            "used_tags": self.used_tags,
            "used_designer": self.used_designer,
            "task_protocol": self.task_protocol,
            'dataset': self.dataset,
        }
