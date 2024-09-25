class NotificationService:
    def __init__(self, repository):
        self.repository =  repository
    
    def getAll(self):
        return self.repository.getAll()

    def create_notification(self, for_user_id, message):
        self.repository.create_notification(for_user_id, message)
