from dataclasses import dataclass
from datetime import datetime

@dataclass
class FeedbackDTO:
    id: int
    feedback: str
    timestamp: datetime
    user: str 

    def to_dict(self):
        return {
            'id': self.id,
            'feedback': self.feedback,
            'timestamp': self.timestamp,
            'user': self.user
        }
