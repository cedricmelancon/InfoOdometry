from timeit import default_timer as timer


class SequenceTimer:
    def __init__(self):
        self.curr_time = timer()
        self.avg_time = 0  # average running time for one step
        self.cnt_step = 0
        self.last_time_elapsed = 0

    def tictoc(self):
        curr_time = timer()
        self.last_time_elapsed = curr_time - self.curr_time
        self.curr_time = curr_time
        self.cnt_step += 1
        self.avg_time += (self.last_time_elapsed - self.avg_time) / self.cnt_step

    def get_last_time_elapsed(self):
        return self.last_time_elapsed

    def get_remaining_time(self, curr_step, total_step):
        return (total_step - curr_step) * self.avg_time