class ProgressBar:
    def __init__(self, maxNum, progressBarSize=100):
        self.maxNum = maxNum
        self.progressBarSize = progressBarSize
        self.updateStep = min(100, max(1, round(maxNum / progressBarSize / 2)))
        self.nextUpdateNum = self.updateStep
        self.currStep = 0
        self.printStatus(self.currStep)

    def update(self, stepSize=1):
        self.moveTo(self.currStep + stepSize)

    def moveTo(self, stepValue):
        self.currStep = stepValue
        if self.currStep >= self.nextUpdateNum:
            self.printStatus(self.currStep)
            self.nextUpdateNum = (int(self.currStep / self.updateStep) + 1) * self.updateStep

    def printStatus(self, currNum):
        percentage = currNum / self.maxNum
        numItems = round(percentage * self.progressBarSize)
        numEmpties = self.progressBarSize - numItems
        print("\r[{0}{1}] {2}%".format("=" * numItems, " " * numEmpties, round(percentage * 100, 1)), end="")

    def finish(self):
        self.printStatus(self.maxNum)
        # create a new line
        print()
