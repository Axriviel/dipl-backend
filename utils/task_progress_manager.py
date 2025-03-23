class TaskProgressManager:
    def __init__(self):
        self.task_progress = {}

    def update_progress(self, user_id, progress):
        self.task_progress[user_id] = progress

    def get_progress(self, user_id):
        return self.task_progress.get(user_id, 0)
    
    def reset_user(self, user_id):
        self.task_progress.pop(user_id)

# Vytvoření instance pro sdílení mezi moduly
progress_manager = TaskProgressManager()