import time

class TimeLimitManager:
    def __init__(self):
        self.start_times = {}
        self.time_limits = {}

    def add_user(self, user_id, time_limit_seconds):
        """Uloží čas začátku a časový limit v sekundách pro daného uživatele."""
        self.start_times[user_id] = time.time()
        self.time_limits[user_id] = float(time_limit_seconds)

    def has_time_exceeded(self, user_id):
        """Vrací True, pokud už uplynul časový limit pro daného uživatele."""
        if user_id not in self.start_times:
            return False  # nebo raise výjimku, podle potřeby

        elapsed = time.time() - self.start_times[user_id]
        print("timer check:", elapsed, self.time_limits.get(user_id, 0))
        return elapsed > self.time_limits.get(user_id, 0)

    def reset_user(self, user_id):
        """Odstraní informace o uživateli (např. po dokončení nebo zrušení tasku)."""
        self.start_times.pop(user_id, None)
        self.time_limits.pop(user_id, None)

# Vytvoření instance pro sdílení mezi moduly
time_limit_manager = TimeLimitManager()
