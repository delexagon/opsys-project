import sys
import math
from heapq import heapify, heappush, heappop

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
        cpu_bursts = math.ceil(rand.next()*100)
        self.tau = math.ceil(1/lamb)
        self.bursts = []
        self.current_time = 0
        self.current_burst = 0
        self.current_time_in_burst = 0
        for burst in range(cpu_bursts*2-1):
            if burst % 2 == 0:
                self.bursts.append(math.ceil(rand.next_exp()))
            else:
                self.bursts.append(10*math.ceil(rand.next_exp()))
                
    def get_remaining_bursts(self):
        return len(self.bursts)//2 - self.current_burst//2
    
    # Returns the time remaining in the cpu burst
    def start_burst(self, current_time):
        self.current_time = current_time
        return self.bursts[self.current_burst] - self.current_time_in_burst
        
    def stop_burst(self, current_time):
        self.current_time_in_burst += current_time-self.current_time
        self.current_time = current_time
        
    def finish_burst(self):
        self.current_time_in_burst = 0
        self.current_burst += 1
        if self.current_burst >= len(self.bursts):
            return True
        return False
    
    def print(self):
        print("arrival time {}ms; tau {}ms; {} CPU bursts:".format(self.arrival_time, self.tau, len(self.bursts)//2+1))
        for i in range(0, len(self.bursts)-1, 2):
            print("--> CPU burst {}ms --> I/O burst {}ms".format(self.bursts[i], self.bursts[i+1]))
        print("--> CPU burst {}ms".format(self.bursts[-1]))
        
    
        
    
class CPU:
    def __init__(self, process_num, seed, lamb, bound, switch_time, alpha, rr_time_slice, algorithm):
        rand = Rand48(seed, lamb, bound)
        self.algorithm = algorithm
        self.processes = []
        self.switch_time = switch_time/2
        self.rr_time_slice = rr_time_slice
        self.current_process = None
        self.events_queue = []
        heapify(self.events_queue)
        self.process_queue = []
        for i in range(process_num):
            self.processes.append(Process(rand, lamb))
            heappush(self.events_queue, (self.processes[i].arrival_time, i, "arrival"))
        
    def print(self):
        proc_name_array = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i in range(len(self.processes)):
            print("Process {}: ".format(proc_name_array[i]), end='')
            self.processes[i].print()
            
    def queue_string(self):
        proc_name_array = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if len(self.process_queue) == 0:
            return "empty"
        else:
            return ' '.join(map(lambda x : proc_name_array[x], self.process_queue))
            
    def print_event(self, current_time, process_num, event_string):
        proc_name_array = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if(process_num == None or process_num == -1):
            print("time {}ms: {} [Q: {}]"\
            .format(int(current_time), event_string, self.queue_string()))
        else:
            if(self.algorithm == "SRT" or self.algorithm == "SJF"):
                print("time {}ms: Process {} (tau {}ms) {} [Q: {}]"\
                .format(int(current_time), proc_name_array[process_num], int(processes[process_num].tau), event_string, self.queue_string()))
            else:
                print("time {}ms: Process {} {} [Q: {}]"\
                .format(int(current_time), proc_name_array[process_num], event_string, self.queue_string()))
            
    def run(self):
        self.print_event(0, None, "Simulator started for {}".format(self.algorithm))
        while(len(self.events_queue) != 0):
            current_time, process_num, event_type = heappop(self.events_queue)
            if event_type == "arrival":
                self.process_queue.append(process_num)
                self.print_event(current_time, process_num, "arrived; added to ready queue")
                if self.current_process == None:
                    self.process_queue.pop(0)
                    heappush(self.events_queue, (current_time+self.switch_time, process_num, "switch_in"))
            elif event_type == "switch_out":
                self.current_process = None
                if len(self.process_queue) != 0:
                    process_num = self.process_queue.pop(0)
                    heappush(self.events_queue, (current_time+self.switch_time, process_num, "switch_in"))
            elif event_type == "switch_in":
                self.current_process = process_num
                cpu_burst_time = self.processes[process_num].start_burst(current_time)
                self.print_event(current_time, process_num, "started using the CPU for {}ms burst".format(cpu_burst_time))
                heappush(self.events_queue, (current_time+cpu_burst_time, process_num, "cpu_finish"))
            elif event_type == "cpu_finish":
                terminated = self.processes[self.current_process].finish_burst()
                if not terminated:
                    bursts_left = self.processes[process_num].get_remaining_bursts()
                    self.print_event(current_time, process_num, "completed a CPU burst; {} bursts to go".format(self.processes[process_num].get_remaining_bursts()))
                    io_burst_time = self.processes[self.current_process].start_burst(current_time)
                    self.print_event(current_time, process_num, "switching out of CPU; will block on I/O until time {}ms".format(int(current_time+io_burst_time)))
                    heappush(self.events_queue, (current_time+io_burst_time+self.switch_time, self.current_process, "io_finish"))
                else:
                    self.print_event(current_time, process_num, "terminated")
                heappush(self.events_queue, (current_time+self.switch_time, None, "switch_out"))
            elif event_type == "io_finish":
                self.processes[process_num].finish_burst()
                self.process_queue.append(process_num)
                self.print_event(current_time, process_num, "completed I/O; added to ready queue")
                if self.current_process == None:
                    process_num = self.process_queue.pop(0)
                    heappush(self.events_queue, (current_time+self.switch_time, process_num, "switch_in"))
        self.print_event(current_time, None, "Simulator ended for {}".format(self.algorithm))
    
    
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
    cpu = CPU(process_num, seed, lamb, bound, switch_time, alpha, rr_time_slice, "FCFS")
    cpu.print()
    print()
    cpu.run()
    
    
