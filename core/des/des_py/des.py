from code import interact
import torch
import numpy as np
import progressbar
import des_cpp
import time
import matplotlib.pyplot as plt
# import copy


# import ..core.arrival_process.BatchArrivalProcess
# from core.arrival_process import BatchCIR, BatchArrivalProcess


def infinite_server_queue(arrival_ls, service_ls, eval_t_ls):
    # calculate number of occupied servers at time t
    # given arrival time and service time for each customer
    # service_ls = sampler(size=len(arrival_ls))
    if (type(eval_t_ls) is int) or (type(eval_t_ls) is float):
        eval_t_ls = np.array(eval_t_ls)
    end_ls = arrival_ls + service_ls

    return np.array([np.sum((end_ls >= t) & (arrival_ls <= t)) for t in eval_t_ls])


def infinite_server_queue_batch(batch_arrival_process, eval_t_ls):
    # calculate number of occupied servers at time t

    batch_n_occupied = np.zeros(
        (batch_arrival_process.n_arrival_process, len(eval_t_ls)))
    for i in progressbar.progressbar(range(batch_arrival_process.n_arrival_process)):
        batch_n_occupied[i, :] = infinite_server_queue(
            batch_arrival_process.arrival_process_ls[i].arrival_ls,
            batch_arrival_process.arrival_process_ls[i].service_ls,
            eval_t_ls)
    return batch_n_occupied


class EventList():
    def __init__(self):
        self.event_list = np.array([])
        self.event_time_list = torch.tensor([])
        # self.event_time_ls = []
        self.n_event = 0

    def add_event(self, new_event):
        self.event_list = np.concatenate(([new_event], self.event_list))
        if self.empty():
            self.event_time_list = new_event['time'].unsqueeze(0)
        else:
            self.event_time_list = torch.cat(
                (new_event['time'].unsqueeze(0), self.event_time_list))
            # self.event_time_list = np.concatenate(([new_event['time']], self.event_time_list))

        self.n_event += 1

    def empty(self):
        return self.n_event == 0

    def sort(self):
        if self.n_event != 1:
            order = torch.argsort(self.event_time_list)
            # order = np.argsort(self.event_time_list)
            self.event_list, self.event_time_list = self.event_list[
                order], self.event_time_list[order]

    def next_event(self):
        self.sort()
        next_event = self.event_list[0]
        self.event_list = self.event_list[1:]
        self.event_time_list = self.event_time_list[1:]
        self.n_event -= 1
        return next_event

    def print_status(self):
        print('{} events in event list.'.format(self.n_event))


class Queue():
    def __init__(self):
        self.queue = []

    def append(self, customer, put_back=False):
        if not put_back:
            self.queue.append(customer)
        else:
            self.queue.insert(0, customer)

    def empty(self):
        return len(self.queue) == 0

    def get(self):
        if self.empty():
            raise Exception('Queue is empty')
        else:
            return self.queue.pop(0)


class ScheduledServer():
    def __init__(self, shift_timepoint_ls, shift_type_ls):
        # shift_type can be 'idle', 'busy', 'home'
        # this server is in 'home' before the first shift (t < shift_timepoint_ls[0])
        # this server keeps its status as shift_type_ls[-1] after the last shift (t >= shift_timepoint_ls[-1])
        assert len(shift_type_ls) == len(shift_timepoint_ls)
        # assert work_end_point_ls is in strictly increasing order
        assert torch.min(shift_timepoint_ls[1:] - shift_timepoint_ls[:-1]) > 0

        self._shift_timepoint_ls = shift_timepoint_ls
        self._shift_type_ls = shift_type_ls
        self._remove_duplicate()
        self._n_shift = len(self._shift_timepoint_ls)

    def _whether_idle(self, t):
        # return True and shift_id if server is idle at time t
        # return False and shift_id if server is busy or home at time t

        # e.g, _shift_timepoint_ls = [2,4,6,8]
        # then (-inf, 2) shift_id = -1, [2,4) shift_id = 0, [4,6) shift_id = 1, [6,8) shift_id = 2, [8,inf) shift_id = 3

        if t < self._shift_timepoint_ls[0]:
            return False, -1
        shift_id = torch.where(self._shift_timepoint_ls <= t)[0][-1]
        return self._shift_type_ls[shift_id] == 'idle', shift_id

    def _remove_duplicate(self):
        shift_type_ls = np.array(self._shift_type_ls)
        keep_index = np.where(shift_type_ls[1:] != shift_type_ls[:-1])[0] + 1
        keep_index = np.insert(keep_index, 0, 0)
        self._shift_timepoint_ls = self._shift_timepoint_ls[keep_index]
        self._shift_type_ls = shift_type_ls[keep_index].tolist()

    def idle_till(self, t):
        """Return the timepoint when server becomes idle at time t

        if server is idle at t, return the timepoint when server status is not idle anymore,
        when this server will keep idle, return inf;

        else if server is busy or home at t but will become idle in the future, return -1;

        else if server is busy or home at t and will not become idle in the future, return -2;
        """
        whether_idle, shift_id = self._whether_idle(t)
        if whether_idle:
            if shift_id + 1 == self._n_shift:
                return torch.tensor(float('inf'))
            else:
                return self._shift_timepoint_ls[shift_id + 1]
        elif 'idle' in self._shift_type_ls[shift_id+1:]:
            return torch.tensor(-1)
        else:
            return torch.tensor(-2)

    def set_busy_till(self, busy_start, busy_end):
        # set server status to busy from busy_start till time busy_end
        shift_id = torch.where(self._shift_timepoint_ls <= busy_start)[0][-1]
        self._shift_timepoint_ls = torch.cat((self._shift_timepoint_ls[:shift_id+1], torch.tensor(
            [busy_start, busy_end]), self._shift_timepoint_ls[shift_id+1:]))
        self._shift_type_ls = np.insert(
            self._shift_type_ls, shift_id + 1, ['busy', 'idle'])
        self._n_shift += 2

    def copy(self):
        return ScheduledServer(self._shift_timepoint_ls, self._shift_type_ls)


class ChangingServerCluster():
    def __init__(self, server_ls):
        self.server_ls = server_ls

    def copy(self):
        return ChangingServerCluster([_.copy() for _ in self.server_ls])

    def assign_server(self, t, service_time, first=True):
        """Get server id.

        randomly return the server that is idle from time t to t + service_time

        return -1 if no server meet the requirement at time t but at least one server can fulfill the requirement in the future

        return -2 if no server meet the requirement at time t and all servers will keep in 'home' or 'busy' status forever

        service_time > 0
        """

        # find all idle servers and compute idle till when
        idle_till_ls = [server.idle_till(t) for server in self.server_ls]
        service_end = t + service_time

        can_fullfill = []
        for i, idle_till in enumerate(idle_till_ls):
            if idle_till == torch.tensor(-2):
                can_fullfill.append(-2)
                # -2 means this server will keep in 'home' or 'busy' status forever
            elif idle_till == torch.tensor(-1) or idle_till < service_end:
                can_fullfill.append(-1)
                # -1 means this server cannot fulfill the requirement at time t but can fulfill the requirement in the future
            else:
                can_fullfill.append(0)
                # 0 means this server can fulfill the requirement at time t

        if 0 in can_fullfill:
            if first:
                # 固定返回第一个可以fulfill的server
                server_id = can_fullfill.index(0)
            else:
                server_id = np.random.choice(
                    np.where([i == 0 for i in can_fullfill])[0])
                # set this server to busy, return server_id
                # 这里有个小问题，如果 t 或者 t + service_time 和 self.server_ls[server_id]._shift_timepoint_ls 里的某个值完全相等的话，会不会有bug: 没影响
            self.server_ls[server_id].set_busy_till(t, t + service_time)
            return server_id
        elif -1 in can_fullfill:
            return -1
        else:
            return -2

    def idle(self, t, server_id):
        # nothing need to do here
        return 0


def get_server_ls(break_point, n_server):
    # given #server in each time interval, and the interval breakpoint
    # generate server_ls

    assert len(n_server) + 1 == len(break_point)
    max_n_server = max(n_server)
    server_ls = []

    for i in range(1, max_n_server+1):
        status_ls = ['idle' if n_server[k] >= i else 'busy' for k in range(
            len(break_point)-1)] + ['busy']
        server_ls.append(ScheduledServer(
            torch.tensor(break_point), status_ls))
    return server_ls


def batch_multi_server_queue(batch_arrival_process, servers):
    batch_wait_ls = np.ndarray(
        (batch_arrival_process.n_arrival_process), dtype=np.object)
    for i in progressbar.progressbar(range(batch_arrival_process.n_arrival_process)):
        batch_wait_ls[i, ] = multi_server_queue(torch.tensor(batch_arrival_process.arrival_process_ls[i].arrival_ls),
                                                torch.tensor(batch_arrival_process.arrival_process_ls[i].service_ls), servers)
    return batch_wait_ls


def multi_server_queue(arrival_ls, service_ls, servers):
    """
    nan in wait_ls or exit_ls means the customer is not served before end of service
    """
    event_list = EventList()
    queue = Queue()
    N = len(arrival_ls)
    for i in range(N):
        i_th_arrival_event = {'type': 'new_arrival',
                              'time': arrival_ls[i], 'service': service_ls[i], 'arrival_id': i}
        event_list.add_event(i_th_arrival_event)

    wait_ls = torch.ones(N) * float('nan')
    exit_ls = torch.ones(N) * float('nan')

    if N > 0:
        del i_th_arrival_event, i, N

    # make sure do not modify the input servers
    # servers = copy.deepcopy(servers)  # cost long time. 但是和 deep copy 这个函数本身无关，看 profiling 的结果知道，大多数的时间不在这里

    servers = servers.copy()  # the self defined copy function is still slow

    while not event_list.empty():
        # event_list.print_status()
        next_event = event_list.next_event()
        if next_event['type'] == 'new_arrival':
            if queue.empty():
                server_id = servers.assign_server(
                    next_event['time'], next_event['service'])
                if server_id == -1:
                    customer = {'arrival': next_event['time'],
                                'service': next_event['service'],
                                'arrival_id': next_event['arrival_id']}
                    queue.append(customer)
                elif server_id == -2:
                    break
                else:
                    arrival_id = next_event['arrival_id']
                    event_busy_till = {'type': 'server_busy_till', 'time': next_event['time'] + next_event['service'], 'arrival_id': arrival_id,
                                       'server_id': server_id}
                    wait_ls[arrival_id] = 0
                    event_list.add_event(event_busy_till)
            else:
                customer = {'arrival': next_event['time'],
                            'service': next_event['service'],
                            'arrival_id': next_event['arrival_id']}
                queue.append(customer)

        if next_event['type'] == 'server_busy_till':
            # one server becomes idle, but this server may not able to accept new customer
            server_id = next_event['server_id']
            servers.idle(next_event['time'], server_id)
            arrival_id = next_event['arrival_id']
            exit_time = next_event['time']
            exit_ls[arrival_id] = exit_time

            if not queue.empty():
                next_customer = queue.get()
                server_id = servers.assign_server(
                    exit_time, next_customer['service'])
                if server_id == -1:
                    queue.append(next_customer, put_back=True)
                elif server_id == -2:
                    break
                else:
                    arrival_id = next_customer['arrival_id']
                    event_busy_till = {'time': exit_time + next_customer['service'], 'arrival_id': arrival_id,
                                       'server_id': server_id, 'type': 'server_busy_till'}
                    wait_ls[next_customer['arrival_id']] = exit_time - \
                        next_customer['arrival']
                    event_list.add_event(event_busy_till)
    return wait_ls, exit_ls


if __name__ == '__main__':
    P = 22
    interval_len = 0.5
    T = P * interval_len

    change_point = np.linspace(0, T, P+1)
    n_server_ls = [40]*P

    server_ls = get_server_ls(change_point, n_server_ls)

    servers = ChangingServerCluster(server_ls)

    # lognormal distribution with
    # # mean 206.44 and variance 23,667 (in seconds) as estimated from the data. Each waiting call
    # lognormal_var = 206.44/3600
    # lognormal_mean = 23667/3600**2

    lognormal_var = 0.1
    lognormal_mean = 1
    normal_sigma = (
        np.log(lognormal_var / lognormal_mean ** 2 + 1))**0.5
    normal_mean = np.log(lognormal_mean) - \
        normal_sigma ** 2 / 2

    # service_rate = 1
    # sampler = lambda size: np.random.exponential(1/service_rate, size=size)
    def sampler(size): return np.random.lognormal(
        mean=normal_mean, sigma=normal_sigma, size=size)

    wgan_arrival_epoch_ls = np.ndarray(100, dtype=object)

    n_rep = 50
    for i in range(n_rep):
        wgan_arrival_epoch_ls[i] = torch.tensor(
            np.sort(np.random.uniform(0, T, 1000)))

    lim = 1000000

    # get time needed for running the simulation
    start_time = time.time()
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    service_ls_rec = []
    for i in progressbar.progressbar(range(n_rep)):
        service_ls = sampler(len(wgan_arrival_epoch_ls[i]))
        service_ls_rec.append(service_ls)
        wait_ls, exit_ls = multi_server_queue(
            wgan_arrival_epoch_ls[0], service_ls, servers)
        plt.plot(wgan_arrival_epoch_ls[0][:lim],
                 wait_ls[:lim], label='rep {}'.format(i))
    # plt.legend()
    end_time = time.time()
    print('time needed for running the simulation for one rep: ',
          (end_time - start_time)/n_rep)

    # plot the result
    start_time = time.time()
    # plt.subplot(122)
    for i in progressbar.progressbar(range(n_rep)):
        a = wgan_arrival_epoch_ls[i].tolist()
        b = service_ls_rec[i].tolist()
        c = change_point.tolist()
        wait_ls = des_cpp.multi_server_queue(
            a, b, c, n_server_ls, False)
        # set -1 to nan
        wait_ls[wait_ls == -1] = torch.nan
        plt.plot(wgan_arrival_epoch_ls[i][:lim],
                 wait_ls[:lim], label='rep {}'.format(i))
    # plt.legend()
    end_time = time.time()
    # plt.show()

    print('time needed for running the simulation for one rep: ',
          (end_time - start_time)/n_rep)
