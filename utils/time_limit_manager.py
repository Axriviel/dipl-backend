import time

class TimeLimitManager:
    def __init__(self):
        self.start_times = {}
        self.time_limits = {}

    def add_user(self, user_id, time_limit_seconds):
        self.start_times[user_id] = time.time()
        self.time_limits[user_id] = float(time_limit_seconds)

    def has_time_exceeded(self, user_id):
        if user_id not in self.start_times:
            return False 

        elapsed = time.time() - self.start_times[user_id]
        print("timer check:", elapsed, self.time_limits.get(user_id, 0))
        return elapsed > self.time_limits.get(user_id, 0)

    def reset_user(self, user_id):
        if self.start_times.get(user_id, False):
            self.start_times.pop(user_id, None)
        if self.time_limits.get(user_id, False):
            self.time_limits.pop(user_id, None)

time_limit_manager = TimeLimitManager()
