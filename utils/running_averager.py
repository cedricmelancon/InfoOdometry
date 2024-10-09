class RunningAverager:
    def __init__(self):
        self.avg = 0
        self.cnt = 0

    def append(self, value):
        self.cnt += 1
        self.avg += (value - self.avg) / self.cnt

    def item(self):
        return self.avg

    def cnt(self):
        return self.cnt