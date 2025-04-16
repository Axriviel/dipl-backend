from tensorflow.keras.callbacks import Callback

class TaskProgressManager:
    def __init__(self):
        self.task_progress = {}

    def update_progress(self, user_id, progress):
        self.task_progress[user_id] = progress

    def get_progress(self, user_id):
        return self.task_progress.get(user_id, -1)
    
    def reset_user(self, user_id):
        self.task_progress.pop(user_id)

class GrowthLimiterManager:
    def __init__(self):
        self.user_growth = {}

    def set_growth(self, user_id, growth_function, max_progress):
        self.user_growth[user_id] = {
            "growth_function": growth_function,
            "current_progress": 1,
            "max_progress": max_progress
        }

    def update_progress(self, user_id, increment=1):
        if user_id in self.user_growth:
            self.user_growth[user_id]["current_progress"] += increment

    def get_growth_function(self, user_id):
        return self.user_growth.get(user_id, {}).get("growth_function", None)

    def get_current_progress(self, user_id):
        return self.user_growth.get(user_id, {}).get("current_progress", -1)

    def get_max_progress(self, user_id):
        return self.user_growth.get(user_id, {}).get("max_progress", -1)

    def get_growth_info(self, user_id):
        return self.user_growth.get(user_id, None)

    def reset_user(self, user_id):
        self.user_growth.pop(user_id, None)



class ExternalTerminationCallback(Callback):
    def __init__(self, user_id):
        super().__init__()
        self.user_id = user_id

    def on_batch_end(self, batch, logs=None):

        if termination_manager.is_terminated(self.user_id):
            self.model.stop_training = True

class TerminationManager:
    def __init__(self):
        self.terminate_task = {}

    def terminate_user_task(self, user_id):
        if(progress_manager.get_progress(user_id)!= -1):
            self.terminate_task[user_id] = True

    def is_terminated(self, user_id):
        return self.terminate_task.get(user_id, False)
    
    def reset_user(self, user_id):
        if self.terminate_task.get(user_id, False):
             self.terminate_task.pop(user_id)

progress_manager = TaskProgressManager()
termination_manager = TerminationManager()
growth_limiter_manager = GrowthLimiterManager()