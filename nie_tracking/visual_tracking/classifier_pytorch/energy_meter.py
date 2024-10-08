import sys
from time import time, sleep
from threading import Timer
import pynvml
import os
from tensorboardX import SummaryWriter


class EnergyMeter:
    def __init__(self, writer: SummaryWriter, period=0.01, dir="./"):
        assert period >= 0.005, "Measurement period below than 5ms"
        self.period = period
        pynvml.nvmlInit()
        self.gpu_handles = [
            pynvml.nvmlDeviceGetHandleByIndex(idx)
            for idx in range(pynvml.nvmlDeviceGetCount())
        ]
        self.dir = dir
        self.writer = writer
        self.writer.add_scalar("xtras/energy_usage", 0, 0)

    def __enter__(self):
        self.done = False
        self.steps = 0
        self.energy = 0
        self.next_t = time()
        self.run()
        return self

    def _get_energy_usage(self):
        energy = 0
        for handle in self.gpu_handles:
            power = pynvml.nvmlDeviceGetPowerUsage(handle)
            energy += power / 1000.0 * self.period
        return energy

    def run(self):
        if not self.done:
            self.t = Timer(self.next_t - time(), self.run)
            self.t.start()
            self.next_t += self.period
            self.steps = self.steps + 1
            self.energy = self.energy + self._get_energy_usage()
            if self.steps % 100 == 0:
                self.writer.add_scalar("xtras/energy_usage", self.energy, self.steps)

    def __exit__(self, type, value, traceback):
        self.done = True
        self.t.cancel()
        total = f"energy_used_{int(self.energy)}J"
        with open(os.path.join(self.dir, f"{total}.log"), "w") as file:
            file.write(f"Total energy used: {int(self.energy)}J")
        print(f"Total energy used: {self.energy}J")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        period = float(sys.argv[1])
    else:
        period = 0.01

    em = EnergyMeter(period)
    with em:
        # put code you what to measure energy of here
        sleep(2)
