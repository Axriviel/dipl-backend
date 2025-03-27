class TaskProgressManager:
    def __init__(self):
        self.task_progress = {}

    def update_progress(self, user_id, progress):
        self.task_progress[user_id] = progress

    def get_progress(self, user_id):
        return self.task_progress.get(user_id, 0)
    
    def reset_user(self, user_id):
        self.task_progress.pop(user_id)

class TerminationManager:
    def __init__(self):
        self.terminate_task = {}

    def terminate_user_task(self, user_id):
        self.terminate_task[user_id] = True

    def is_terminated(self, user_id):
        return self.terminate_task.get(user_id, False)
    
    def reset_user(self, user_id):
        if self.terminate_task.get(user_id, False):
             self.terminate_task.pop(user_id)

progress_manager = TaskProgressManager()
termination_manager = TerminationManager()