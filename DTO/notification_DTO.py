from dataclasses import dataclass
from datetime import datetime

@dataclass
class NotificationDTO:
    id: int
    message: str
    timestamp: datetime
    was_read: bool
    user: int

    def to_dict(self):
        return {
            'id': self.id,
            'message': self.message,
            'timestamp': self.timestamp,
            'was_read': self.was_read,
            'user': self.user
        }
