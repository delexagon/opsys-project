import sys
import math
import heapq

class Rand48:
    def __init__(self, n, exp_avg, bound):
        self.num = n*(2**16) + 0x330E
        self.exp_avg = exp_avg
        self.bound = bound
        
    def next(self):
        self.num = (0x5DEECE66D * self.num + 0xB) % (2**48)
        return self.num / 2**48
        
    def next_exp(self):
        while True:
            val = -math.log(self.next()) / self.exp_avg
            if not (val > self.bound):
                break
        return val

class Process:
    def __init__(self, rand, lamb):
        self.arrival_time = math.floor(rand.next_exp())
        self.cpu_bursts = math.ceil(rand.next()*100)
        self.tau = math.ceil(1/lamb)
        self.bursts = []
        self.current_burst = 0
        self.current_burst_time = 0
        for burst in range(self.cpu_bursts*2-1):
            if burst % 2 == 0:
                self.bursts.append(math.ceil(rand.next_exp()))
            else:
                self.bursts.append(10*math.ceil(rand.next_exp()))
    
    def print(self):
        print("arrival time {}ms; tau {}ms; {} CPU bursts:".format(self.arrival_time, self.tau, len(self.bursts)//2+1))
        for i in range(0, len(self.bursts)-1, 2):
            print("--> CPU burst {}ms --> I/O burst {}ms".format(self.bursts[i], self.bursts[i+1]))
        print("--> CPU burst {}ms".format(self.bursts[-1]))
        
    
        
    
class CPU:
    def __init__(self, process_num, seed, lamb, bound, switch_time, alpha, rr_time_slice):
        rand = Rand48(seed, lamb, bound)
        self.processes = []
        for i in range(process_num):
            self.processes.append(Process(rand, lamb))
        
    def print(self):
        proc_name_array = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i in range(len(self.processes)):
            print("Process {}: ".format(proc_name_array[i]), end='')
            self.processes[i].print()
            
    def run():
        return
    
    
    
    
    
if __name__ == "__main__":
    if(len(sys.argv) < 8):
        print("Not enough arguments")
        exit()
    process_num = int(sys.argv[1])
    seed = int(sys.argv[2])
    lamb = float(sys.argv[3])
    bound = float(sys.argv[4])
    switch_time = int(sys.argv[5])
    alpha = float(sys.argv[6])
    rr_time_slice = int(sys.argv[7])
    cpu = CPU(process_num, seed, lamb, bound, switch_time, alpha, rr_time_slice)
    cpu.print()

