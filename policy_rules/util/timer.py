import time

from termcolor import colored


class TimerContextManager:
    def __init__(self, description, end=""):
        self.description = description
        if len(end) > 0:
            end = ". " + end
        self.end = end

    def __enter__(self):
        print(colored(f"Started {self.description}...", "magenta"))
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if traceback is not None:
            return
        end_time = time.time()
        execution_time = end_time - self.start_time
        if self.description:
            print(colored(f"Finished {self.description} in {execution_time}s" + self.end, "blue"))

    def get_time(self):
        return time.time() - self.start_time
