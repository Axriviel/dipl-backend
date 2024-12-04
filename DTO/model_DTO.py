from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelDTO:
    id: int
    name: str
    accuracy: float
    metric_value: float
    watched_metric: str
    metric_values_history: Optional[List[float]]  # JSON pole pro historické hodnoty
    used_opt_method: str
    error: float
    dataset: str

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'accuracy': self.accuracy,
            'metric_value': self.metric_value,
            'watched_metric': self.watched_metric,
            'metric_values_history': self.metric_values_history,
            'used_opt_method': self.used_opt_method,
            'error': self.error,
            'dataset': self.dataset,
        }
