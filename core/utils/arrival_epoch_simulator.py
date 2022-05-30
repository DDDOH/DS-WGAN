import numpy as np


def arrival_epoch_simulator(arrival_count_mat, interval_length):
    """Simulate arrival epochs from arrival count.
    Args:
        arrival_count_ls (list): 
        interval_length (int): 
    """
    arrival_count_mat = np.maximum(arrival_count_mat.astype(int), 0)
    n_arrival_count = np.shape(arrival_count_mat)[0]
    P = np.shape(arrival_count_mat)[1]
    arrival_epoch_ls = np.ndarray((n_arrival_count,), dtype=np.object)
    for i in range(n_arrival_count):
        total_arrival = np.sum(arrival_count_mat[i, :])
        total_arrival = total_arrival
        arrival_ls = np.zeros(int(total_arrival))
        index = 0
        for j in range(P):
            interval_start = j * interval_length
            interval_end = interval_start + interval_length
            arrival_ls_one_interval = np.random.uniform(
                interval_start, interval_end, size=arrival_count_mat[i, j])
            arrival_ls[index:index+arrival_count_mat[i, j]
                       ] = arrival_ls_one_interval
            index += arrival_count_mat[i, j]
        arrival_epoch_ls[i] = np.sort(arrival_ls)
    return arrival_epoch_ls


if __name__ == '__main__':
    arrival_count_ls = np.array([10, 20, 30, 15])
    interval_length = 5
    arrival_ls = arrival_epoch_simulator(arrival_count_ls, interval_length)
