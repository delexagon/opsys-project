import sys
import math
from heapq import heapify, heappush, heappop


proc_name_array = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

class Rand48:
    """Emulates the rand48 family of functions in C"""
    
    def __init__(self, n, exp_avg, bound):
        """Initializes the Rand48 to the same value as srand48(long) in C
        Additional variables allow the exponential generation to meet project specifications
        without having to pass them in for every call"""
        self.num = n*(2**16) + 0x330E
        self.exp_avg = exp_avg
        self.bound = bound
        
    def next(self):
        """Returns the next value of drand48() in C"""
        self.num = (0x5DEECE66D * self.num + 0xB) % (2**48)
        return self.num / 2**48
        
    def next_exp(self):
        """Returns an exponential distribution as per project specifications"""
        while True:
            val = -math.log(self.next()) / self.exp_avg
            if not (val > self.bound):
                break
        return val

class Process:
    """Initializes the process to random values as per the project's specifications"""
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
                
    def get_remaining_cpu_bursts(self):
        """Returns an integer representing the number of remaining cpu bursts for this process"""
        return len(self.bursts)//2 - self.current_burst//2
        
    def remaining_time_estimate(self, current_time):
        return self.tau - self.current_time_in_burst - current_time + self.current_time
    
    def start_burst(self, current_time):
        """Returns the time remaining in the current burst, cpu or io"""
        self.current_time = current_time
        if len(self.bursts) <= self.current_burst:
            print(len(self.bursts), file=sys.stderr)
            print(self.current_burst, file=sys.stderr)
            exit()
        return self.bursts[self.current_burst] - self.current_time_in_burst
        
    def remaining_time(self, current_time):
        return self.bursts[self.current_burst] - self.current_time_in_burst - current_time + self.current_time
        
    def total_burst_time(self):
        return self.bursts[self.current_burst]
        
    def check_correct_time(self, current_time):
        return (current_time-self.current_time + self.current_time_in_burst) == self.bursts[self.current_burst]
        
    def stop_burst(self, current_time):
        """Halts this process so that it can be resumed later, for preemption"""
        self.current_time_in_burst += current_time-self.current_time
        self.current_time = current_time
        return self.bursts[self.current_burst] - self.current_time_in_burst
        
    def finish_burst(self):
        """Moves this process to the next burst and returns if the process is terminated
        Note that this means that the CPU has to explicitly tell the process to progress."""
        self.current_time_in_burst = 0
        self.current_burst += 1
        if self.current_burst >= len(self.bursts):
            return False
        return self.bursts[self.current_burst-1]
    
    def print(self):
        """Prints a description of this process as per project specifications"""
        print("arrival time {}ms; tau {}ms; {} CPU burst{}:"\
            .format(self.arrival_time, self.tau, len(self.bursts)//2+1, "" if len(self.bursts)//2+1 == 1 else "s"))
        for i in range(0, len(self.bursts)-1, 2):
            print("--> CPU burst {}ms --> I/O burst {}ms".format(self.bursts[i], self.bursts[i+1]))
        print("--> CPU burst {}ms".format(self.bursts[-1]))
        
    
        
    
class CPU:

    def __init__(self, process_num, seed, lamb, bound, switch_time, alpha, rr_time_slice):
        """Initializes the CPU to default values"""
        # Variables needed to create the random number generator during reset
        self.reset_vals = (seed, lamb, bound)
        # Switching threads off and on are considered separate events,
        # so this time is halved
        self.switch_time = switch_time/2
        self.alpha = alpha
        self.rr_time_slice = rr_time_slice
        self.reset()
        self.total_burst_time = 0
        self.num_bursts = 0
        self.num_context_switches = 0
        self.num_preemptions = 0
            
    def reset(self):
        """Reinitializes the CPU to default values"""
        # Randomization is only used for creating processes, so this variable is not class specific
        rand = Rand48(self.reset_vals[0], self.reset_vals[1], self.reset_vals[2])
        self.processes = []
        self.lock = False
        # Probably necessary to keep track of for preemption
        # Note that this is an integer index of self.processes[], not an actual Process object
        self.current_process = None
        # Events are added to a heap, and popped out from it in temporal order
        # TODO: Currently, 'tiebreakers' for events are not implemented
        self.events_queue = []
        self.process_queue = []
        for i in range(process_num):
            # Processes generate themselves using the random number generator
            self.processes.append(Process(rand, lamb))
            # Events have a particular syntax, a tuple of three parts:
            # First, the time at which the event occurs
            # Second, the process that the event affects
            # Third, the type of event which is occurring
            heappush(self.events_queue, (self.processes[i].arrival_time, "d_arrival", i))
        
    def print(self):
        """Prints a description of all the processes present in this CPU as per project specifications"""
        for i in range(len(self.processes)):
            print("Process {}: ".format(proc_name_array[i]), end='')
            self.processes[i].print()
        print()
        
    def recalculate_tau(self, time, process, burst_time):
        old_tau = self.processes[process].tau
        new_tau = math.ceil(alpha*burst_time + (1-alpha) * old_tau)
        self.processes[process].tau = new_tau
        self.print_event(time, None, f"Recalculated tau for process {proc_name_array[process]}: old tau {old_tau}ms; new tau {new_tau}ms")
            
    def queue_string(self):
        """Returns a string representing the current state of the queue in this CPU"""
        if len(self.process_queue) == 0:
            return "empty"
        else:
            return ' '.join(map(lambda x : proc_name_array[x], self.process_queue))
            
    def print_event(self, current_time, process_num, event_string):
        """Algorithm print statements all have a very similar format, so this will automatically add flavor text"""
        if(process_num == None or process_num == -1):
            print("time {}ms: {} [Q: {}]"\
                .format(int(current_time), event_string, self.queue_string()))
        else:
            if(self.algorithm == "SRT" or self.algorithm == "SJF"):
                print("time {}ms: Process {} (tau {}ms) {} [Q: {}]"\
                    .format(int(current_time), proc_name_array[process_num], int(self.processes[process_num].tau), event_string, self.queue_string()))
            else:
                print("time {}ms: Process {} {} [Q: {}]"\
                    .format(int(current_time), proc_name_array[process_num], event_string, self.queue_string()))
                    
    def output_file(self, current_time, algorithm):
        """ Generalized function for outputting algorithm statistics """
        f = open("simout.txt", "a")  # Append to file for consecutive calls
        f.write(f"Algorithm {algorithm}\n")
        avg_burst_time = self.total_burst_time/self.num_bursts
        total_context_switch_time = self.num_context_switches * self.switch_time * 2
        avg_wait_time = (current_time - (self.total_burst_time + total_context_switch_time))/self.num_bursts
        avg_turnaround_time = (avg_burst_time+avg_wait_time +
                               total_context_switch_time/self.num_context_switches)/self.num_bursts
        utilization = self.total_burst_time/current_time * 100
        f.write(f"-- average CPU burst time: {math.ceil(avg_burst_time)}.000 ms\n")
        f.write(f"-- average wait time: {math.ceil(avg_wait_time)}.000 ms\n")
        f.write(f"-- average turnaround time: {math.ceil(avg_turnaround_time)}.000 ms\n")
        f.write(f"-- total number of context switches: {self.num_context_switches}\n")
        f.write(f"-- total number of preemptions: {self.num_preemptions}\n")
        f.write(f"-- CPU utilization: {math.ceil(utilization)}.000%\n")
        f.close()
            
    # The code here is meant to be fairly generalized; copy it over and start modifying it to fit the other algorithms.
    # A lot of code will be common between each algorithm; we can move these into their own functions.
    # I hope my comments and variable names are descriptive enough that it's fairly obvious what's going on here;
    # if not, just ask.
    def run_fcfs(self):
        """Runs the FCFS algorithm and prints results"""
        self.algorithm = "FCFS"
        # Syntax of print_event: Time of event, the integer of the process related to the event (None if not applicable), string of what needs to be printed for the event.
        # Things like the current queue and the tau of the process are automatically printed.
        self.print_event(0, None, "Simulator started for {}".format(self.algorithm))
        while(len(self.events_queue) != 0):
            # Take in the next event to be processed
            current_time, event_type, process_num  = heappop(self.events_queue)
            # Events that come in are handled based on what they are
            if event_type == "d_arrival":
                self.process_queue.append(process_num)
                self.print_event(current_time, process_num, "arrived; added to ready queue")
                if self.current_process == None:
                    heappush(self.events_queue, (current_time+self.switch_time, "b_switch_in", process_num))
            # Switching the current thread out and switching a new thread in are considered separate events;
            # this is to avoid complicated calculations beforehand and so that events arriving during switchout time will be noticed in the queue.
            elif event_type == "switch_out":
                self.current_process = None
                if len(self.process_queue) != 0:
                    heappush(self.events_queue, (current_time+self.switch_time, "b_switch_in", process_num))
            # Note neither type of switching uses process_num, switching in draws from the start of the queue
            elif event_type == "b_switch_in":
                if self.current_process == None:
                    self.current_process = self.process_queue.pop(0)
                    cpu_burst_time = self.processes[self.current_process].start_burst(current_time)
                    self.total_burst_time += cpu_burst_time
                    self.num_bursts += 1
                    self.num_context_switches += 1  # Only count context switches according to pdf pg. 8
                    self.print_event(current_time, self.current_process, "started using the CPU for {}ms burst".format(cpu_burst_time))
                    heappush(self.events_queue, (current_time+cpu_burst_time, "a_cpu_finish", self.current_process))
            elif event_type == "a_cpu_finish":
                # This needs to be called to allow the process to move on
                running = self.processes[self.current_process].finish_burst()
                if running:
                    bursts_left = self.processes[process_num].get_remaining_cpu_bursts()
                    # Note that all plural words in print statements have to check whether they should have an 's' at the end
                    self.print_event(current_time, process_num, "completed a CPU burst; {} burst{} to go".format(bursts_left, "" if bursts_left == 1 else "s"))
                    io_burst_time = self.processes[self.current_process].start_burst(current_time)
                    # Note the IO block only starts *after* the process is switched out of the CPU;
                    # This is considered during the c_io_finish call
                    self.print_event(current_time, process_num, "switching out of CPU; will block on I/O until time {}ms"\
                        .format(int(current_time+io_burst_time+self.switch_time)))
                    heappush(self.events_queue, (current_time+io_burst_time+self.switch_time, "c_io_finish", self.current_process))
                else:
                    self.print_event(current_time, process_num, "terminated")
                heappush(self.events_queue, (current_time+self.switch_time, "switch_out", process_num))
            elif event_type == "c_io_finish":
                self.processes[process_num].finish_burst()
                self.process_queue.append(process_num)
                self.print_event(current_time, process_num, "completed I/O; added to ready queue")
                if self.current_process == None:
                    heappush(self.events_queue, (current_time+self.switch_time, "b_switch_in", process_num))
        self.print_event(current_time, None, "Simulator ended for {}".format(self.algorithm))
        self.output_file(current_time, "FCFS")
        print()
    
    def run_sjf(self):
        """Runs the SJF algorithm and prints results"""
        self.algorithm = "SJF"
        self.print_event(0, None, "Simulator started for {}".format(self.algorithm))
        while(len(self.events_queue) != 0):
            current_time, event_type, process_num = heappop(self.events_queue)
            if event_type == "d_arrival":
                tau = self.processes[process_num].tau
                added = False
                for i in range(len(self.process_queue)):
                    proctau = self.processes[self.process_queue[i]].tau
                    if proctau > tau:
                        self.process_queue.insert(i, process_num)
                        added = True
                        break
                    if proctau == tau and process_num < self.process_queue[i]:
                        self.process_queue.insert(i, process_num)
                        added = True
                        break
                if not added:
                    self.process_queue.append(process_num)
                self.print_event(current_time, process_num, "arrived; added to ready queue")
                if self.current_process == None:
                    new_process = self.process_queue.pop(0)
                    heappush(self.events_queue, (current_time+self.switch_time, "b_switch_in", new_process))
                    self.lock = True
            elif event_type == "switch_out":
                self.lock = False
                self.current_process = None
                if (not self.lock) and len(self.process_queue) != 0:
                    new_process = self.process_queue.pop(0)
                    heappush(self.events_queue, (current_time+self.switch_time, "b_switch_in", new_process))
                    self.lock = True
            elif event_type == "b_switch_in":
                if self.current_process == None:
                    self.lock = True
                    self.current_process = process_num
                    cpu_burst_time = self.processes[self.current_process].start_burst(current_time)
                    self.print_event(current_time, self.current_process, "started using the CPU for {}ms burst".format(cpu_burst_time))
                    heappush(self.events_queue, (current_time+cpu_burst_time, "a_cpu_finish", self.current_process))
                else:
                    print("Error: Attempting to switch in process when lock should be in place", file=sys.stderr)
                    sys.exit()
            elif event_type == "a_cpu_finish":
                running = self.processes[self.current_process].finish_burst()
                if running:
                    bursts_left = self.processes[process_num].get_remaining_cpu_bursts()
                    self.print_event(current_time, process_num, "completed a CPU burst; {} burst{} to go".format(bursts_left, "" if bursts_left == 1 else "s"))
                    self.recalculate_tau(current_time, process_num, running)
                    io_burst_time = self.processes[self.current_process].start_burst(current_time)
                    self.print_event(current_time, None, "Process {} switching out of CPU; will block on I/O until time {}ms"\
                        .format(proc_name_array[process_num], int(current_time+io_burst_time+self.switch_time)))
                    heappush(self.events_queue, (current_time+io_burst_time+self.switch_time, "c_io_finish", self.current_process))
                else:
                    self.print_event(current_time, None, "Process {} terminated".format(proc_name_array[process_num]))
                heappush(self.events_queue, (current_time+self.switch_time, "switch_out", process_num))
            elif event_type == "c_io_finish":
                running = self.processes[process_num].finish_burst()
                if not running:
                    print("Error: Process should not end on IO burst", file=sys.stderr)
                    sys.exit()
                tau = self.processes[process_num].tau
                added = False
                for i in range(len(self.process_queue)):
                    proctau = self.processes[self.process_queue[i]].tau
                    if proctau > tau:
                        self.process_queue.insert(i, process_num)
                        added = True
                        break
                    if proctau == tau and process_num < self.process_queue[i]:
                        self.process_queue.insert(i, process_num)
                        added = True
                        break
                if not added:
                    self.process_queue.append(process_num)
                self.print_event(current_time, process_num, "completed I/O; added to ready queue")
                if not self.lock:
                    new_process = self.process_queue.pop(0)
                    heappush(self.events_queue, (current_time+self.switch_time, "b_switch_in", new_process))
                    self.lock = True
        self.print_event(current_time, None, "Simulator ended for {}".format(self.algorithm))
        print()
    
    def run_srt(self):
        """Runs the SRT algorithm and prints results"""
        self.algorithm = "SRT"
        self.print_event(0, None, "Simulator started for {}".format(self.algorithm))
        switching_out = False
        last_resolved = 0
        while(len(self.events_queue) != 0):
            current_time, event_type, process_num = heappop(self.events_queue)
            if event_type == "d_arrival":
                tau = self.processes[process_num].tau - self.processes[process_num].current_time_in_burst
                added = False
                for i in range(len(self.process_queue)):
                    proctau = self.processes[self.process_queue[i]].tau - self.processes[self.process_queue[i]].current_time_in_burst
                    if proctau > tau:
                        self.process_queue.insert(i, process_num)
                        added = True
                        break
                    if proctau == tau and process_num < self.process_queue[i]:
                        self.process_queue.insert(i, process_num)
                        added = True
                        break
                if not added:
                    self.process_queue.append(process_num)
                self.print_event(current_time, process_num, "arrived; added to ready queue")
                if self.current_process == None:
                    new_process = self.process_queue.pop(0)
                    heappush(self.events_queue, (current_time+self.switch_time, "b_switch_in", new_process))
                    self.lock = True
            elif event_type == "switch_out":
                switching_out = False
                self.lock = False
                self.current_process = None
                if (not self.lock) and len(self.process_queue) != 0:
                    new_process = self.process_queue.pop(0)
                    heappush(self.events_queue, (current_time+self.switch_time, "b_switch_in", new_process))
                    self.lock = True
            elif event_type == "b_switch_in":
                if self.current_process == None:
                    self.lock = True
                    self.current_process = process_num
                    cpu_burst_time = self.processes[self.current_process].start_burst(current_time)
                    cpu_original_burst_time = self.processes[self.current_process].total_burst_time()
                    if cpu_burst_time == cpu_original_burst_time:
                        self.print_event(current_time, self.current_process, "started using the CPU for {}ms burst".format(cpu_burst_time))
                    else:
                        self.print_event(current_time, self.current_process, "started using the CPU for remaining {}ms of {}ms burst".format(int(cpu_burst_time), cpu_original_burst_time))
                    if len(self.process_queue) > 0 and self.processes[process_num].tau - self.processes[process_num].current_time_in_burst > self.processes[self.process_queue[0]].tau - self.processes[self.process_queue[0]].current_time_in_burst:
                        self.print_event(current_time, self.process_queue[0], "will preempt {}".format(proc_name_array[process_num]))
                        tau = self.processes[process_num].tau
                        added = False
                        for i in range(len(self.process_queue)):
                            proctau = self.processes[self.process_queue[i]].tau - self.processes[self.process_queue[i]].current_time_in_burst
                            if proctau > tau:
                                self.process_queue.insert(i, process_num)
                                added = True
                                break
                            if proctau == tau and process_num < self.process_queue[i]:
                                self.process_queue.insert(i, process_num)
                                added = True
                                break
                        if not added:
                            self.process_queue.append(process_num)
                        heappush(self.events_queue, (current_time+self.switch_time, "switch_out", self.current_process))
                        switching_out = True
                    else:
                        heappush(self.events_queue, (current_time+cpu_burst_time, "a_cpu_finish", self.current_process))
                else:
                    print("Error: Attempting to switch in process when lock should be in place", file=sys.stderr)
                    sys.exit()
            elif event_type == "a_cpu_finish":
                if not switching_out:
                    if self.current_process == process_num and self.processes[self.current_process].check_correct_time(current_time):
                        running = self.processes[self.current_process].finish_burst()
                        if running:
                            bursts_left = self.processes[process_num].get_remaining_cpu_bursts()
                            self.print_event(current_time, process_num, "completed a CPU burst; {} burst{} to go".format(bursts_left, "" if bursts_left == 1 else "s"))
                            self.recalculate_tau(current_time, process_num, running)
                            io_burst_time = self.processes[self.current_process].start_burst(current_time)
                            self.print_event(current_time, None, "Process {} switching out of CPU; will block on I/O until time {}ms"\
                                .format(proc_name_array[process_num], int(current_time+io_burst_time+self.switch_time)))
                            heappush(self.events_queue, (current_time+io_burst_time+self.switch_time, "c_io_finish", self.current_process))
                        else:
                            self.print_event(current_time, None, "Process {} terminated".format(proc_name_array[process_num]))
                        heappush(self.events_queue, (current_time+self.switch_time, "switch_out", process_num))
                        switching_out = True
            elif event_type == "c_io_finish":
                running = self.processes[process_num].finish_burst()
                if not running:
                    print("Error: Process should not end on IO burst", file=sys.stderr)
                    sys.exit()
                tau = self.processes[process_num].tau
                added = False
                for i in range(len(self.process_queue)):
                    proctau = self.processes[self.process_queue[i]].tau - self.processes[self.process_queue[i]].current_time_in_burst
                    if proctau > tau:
                        self.process_queue.insert(i, process_num)
                        added = True
                        break
                    if proctau == tau and process_num < self.process_queue[i]:
                        self.process_queue.insert(i, process_num)
                        added = True
                        break
                if not added:
                    self.process_queue.append(process_num)
                if self.current_process != None and not switching_out and tau < self.processes[self.current_process].remaining_time_estimate(current_time):
                    printed = True
                    old_process = self.current_process
                    tau = self.processes[old_process].tau - self.processes[old_process].current_time_in_burst
                    self.processes[old_process].stop_burst(current_time)
                    self.print_event(current_time, process_num, "completed I/O; preempting {}".format(proc_name_array[self.current_process]))
                    added = False
                    for i in range(len(self.process_queue)):
                        proctau = self.processes[self.process_queue[i]].tau - self.processes[self.process_queue[i]].current_time_in_burst
                        if proctau > tau:
                            self.process_queue.insert(i, old_process)
                            added = True
                            break
                        if proctau == tau and old_process < self.process_queue[i]:
                            self.process_queue.insert(i, old_process)
                            added = True
                            break
                    if not added:
                        self.process_queue.append(old_process)
                    heappush(self.events_queue, (current_time+self.switch_time, "switch_out", self.current_process))
                    switching_out = True
                else:
                    self.print_event(current_time, process_num, "completed I/O; added to ready queue")
                if not self.lock:
                    heappush(self.events_queue, (current_time, "z_resolve_io_adds", process_num))
            elif event_type == "z_resolve_io_adds":
                # The logic of the output files requires me to add all IO operations that occur at the same time before
                # adding any one of them into the queue
                if last_resolved != current_time:
                    new_process = self.process_queue.pop(0)
                    heappush(self.events_queue, (current_time+self.switch_time, "b_switch_in", new_process))
                    self.lock = True
                last_resolved = current_time
        self.print_event(current_time, None, "Simulator ended for {}".format(self.algorithm))
        print()
    
    def run_rr(self):
        """Runs the RR algorithm and prints results"""
        self.algorithm = "RR"
        self.print_event(0, None, "Simulator started for {} with time slice {}ms".format(self.algorithm, self.rr_time_slice))
        switching_out = False
        last_resolved = 0
        while(len(self.events_queue) != 0):
            current_time, event_type, process_num = heappop(self.events_queue)
            if event_type == "d_arrival":
                self.process_queue.append(process_num)
                self.print_event(current_time, process_num, "arrived; added to ready queue")
                if self.current_process == None:
                    new_process = self.process_queue.pop(0)
                    heappush(self.events_queue, (current_time+self.switch_time, "b_switch_in", new_process))
                    self.lock = True
            elif event_type == "switch_out":
                switching_out = False
                self.lock = False
                self.current_process = None
                if (not self.lock) and len(self.process_queue) != 0:
                    new_process = self.process_queue.pop(0)
                    heappush(self.events_queue, (current_time+self.switch_time, "b_switch_in", new_process))
                    self.lock = True
            elif event_type == "b_switch_in":
                if self.current_process == None:
                    self.lock = True
                    self.current_process = process_num
                    cpu_burst_time = self.processes[self.current_process].start_burst(current_time)
                    cpu_original_burst_time = self.processes[self.current_process].total_burst_time()
                    if cpu_burst_time == cpu_original_burst_time:
                        self.print_event(current_time, self.current_process, "started using the CPU for {}ms burst".format(cpu_burst_time))
                    else:
                        self.print_event(current_time, self.current_process, "started using the CPU for remaining {}ms of {}ms burst".format(int(cpu_burst_time), cpu_original_burst_time))
                    if cpu_burst_time <= self.rr_time_slice:
                        heappush(self.events_queue, (current_time+cpu_burst_time, "a_cpu_finish", self.current_process))
                    else:
                        heappush(self.events_queue, (current_time+self.rr_time_slice, "rr_time_expired", self.current_process))
                else:
                    print("Error: Attempting to switch in process when lock should be in place", file=sys.stderr)
                    sys.exit()
            elif event_type == "rr_time_expired":
                if len(self.process_queue) > 0:
                    old_process = self.current_process
                    time_remaining = self.processes[old_process].stop_burst(current_time)
                    self.print_event(current_time, None, "Time slice expired; process {} preempted with {}ms remaining".format(proc_name_array[self.current_process], int(time_remaining)))
                    self.process_queue.append(old_process)
                    heappush(self.events_queue, (current_time+self.switch_time, "switch_out", self.current_process))
                    switching_out = True
                else:
                    cpu_burst_time = self.processes[self.current_process].remaining_time(current_time)
                    self.print_event(current_time, None, "Time slice expired; no preemption because ready queue is empty".format(proc_name_array[self.current_process], int(time_remaining)))
                    if cpu_burst_time <= self.rr_time_slice:
                        heappush(self.events_queue, (current_time+cpu_burst_time, "a_cpu_finish", self.current_process))
                    else:
                        heappush(self.events_queue, (current_time+self.rr_time_slice, "rr_time_expired", self.current_process))
            elif event_type == "a_cpu_finish":
                if not switching_out:
                    running = self.processes[self.current_process].finish_burst()
                    if running:
                        bursts_left = self.processes[process_num].get_remaining_cpu_bursts()
                        self.print_event(current_time, process_num, "completed a CPU burst; {} burst{} to go".format(bursts_left, "" if bursts_left == 1 else "s"))
                        io_burst_time = self.processes[self.current_process].start_burst(current_time)
                        self.print_event(current_time, None, "Process {} switching out of CPU; will block on I/O until time {}ms"\
                            .format(proc_name_array[process_num], int(current_time+io_burst_time+self.switch_time)))
                        heappush(self.events_queue, (current_time+io_burst_time+self.switch_time, "c_io_finish", self.current_process))
                    else:
                        self.print_event(current_time, None, "Process {} terminated".format(proc_name_array[process_num]))
                    heappush(self.events_queue, (current_time+self.switch_time, "switch_out", process_num))
                    switching_out = True
            elif event_type == "c_io_finish":
                running = self.processes[process_num].finish_burst()
                if not running:
                    print("Error: Process should not end on IO burst", file=sys.stderr)
                    sys.exit()
                self.process_queue.append(process_num)
                self.print_event(current_time, process_num, "completed I/O; added to ready queue")
                if not self.lock:
                    heappush(self.events_queue, (current_time, "z_resolve_io_adds", process_num))
            elif event_type == "z_resolve_io_adds":
                # The logic of the output files requires me to add all IO operations that occur at the same time before
                # adding any one of them into the queue
                if last_resolved != current_time:
                    new_process = self.process_queue.pop(0)
                    heappush(self.events_queue, (current_time+self.switch_time, "b_switch_in", new_process))
                    self.lock = True
                last_resolved = current_time
        self.print_event(current_time, None, "Simulator ended for {}".format(self.algorithm))
        print()
    
    
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
    f = open("simout.txt", "w")
    f.truncate(0)
    f.close()
    cpu.print()
    cpu.run_fcfs()
    cpu.reset()
    cpu.run_sjf()
    cpu.reset()
    cpu.run_srt()
    cpu.reset()
    cpu.run_rr()
    

