import unittest

from main import ScheduledServer, ChangingServerCluster
import torch


class TestScheduledServer(unittest.TestCase):
    def test_idle_till(self):
        server_1 = ScheduledServer(torch.tensor(
            [0.0, 9.0], requires_grad=True), ['idle', 'home'])
        server_2 = ScheduledServer(torch.tensor(
            [2.0, 3.0, 7.0, 8.0], requires_grad=True), ['idle', 'busy', 'idle', 'home'])
        server_3 = ScheduledServer(torch.tensor([2.0, 3.0, 8.0, 10.0, 12.0], requires_grad=True), [
                                   'idle', 'busy', 'idle', 'home', 'idle'])

        server_4 = ScheduledServer(torch.tensor(
            [0.0, 1.0, 2.0, 3.0, 8.0, 9.0], requires_grad=True), ['idle', 'idle', 'idle', 'idle', 'idle', 'home'])

        server_5 = ScheduledServer(torch.tensor([0.0, 1.0, 2.0, 3.0, 8.0,
                                                 9.0, 10.0, 11.0, 12.0, 12.5,
                                                 13.0, 14.0], requires_grad=True),
                                                ['idle', 'idle', 'idle', 'idle', 'idle',
                                                'home', 'home', 'idle', 'idle', 'home',
                                                'idle', 'home'])

        self.assertEqual(server_1.idle_till(3), torch.tensor(9))
        self.assertEqual(server_2.idle_till(10), torch.tensor(-2))
        self.assertEqual(server_2.idle_till(1), torch.tensor(-1))
        self.assertEqual(server_2.idle_till(2), torch.tensor(3.0))
        self.assertEqual(server_2.idle_till(3), torch.tensor(-1))
        self.assertEqual(server_4.idle_till(1.5), torch.tensor(9.0))
        self.assertEqual(server_5.idle_till(11.0), torch.tensor(12.5))


class TestChangingServerCluster(unittest.TestCase):
    def test(self):
        server_1 = ScheduledServer(torch.tensor([0.0, 3.0]), ['idle', 'home'])
        server_2 = ScheduledServer(torch.tensor([2.0, 3.0, 7.0, 8.0]), [
                                   'idle', 'busy', 'idle', 'home'])
        server_3 = ScheduledServer(torch.tensor([2.0, 3.0, 8.0, 10.0, 12.0]), [
                                   'idle', 'busy', 'idle', 'home', 'busy'])

        server_ls = [server_1, server_2, server_3]
        servers = ChangingServerCluster(server_ls)
        self.assertEqual(servers.assign_server(
            t=-1, service_time=1, first=True), -1)
        self.assertEqual(servers.assign_server(
            t=2, service_time=1, first=True), 0)
        self.assertEqual(servers.assign_server(
            t=3, service_time=0.1, first=True), -1)
        self.assertEqual(servers.assign_server(
            t=2, service_time=1.5, first=True), -1)
        self.assertEqual(servers.assign_server(
            t=8, service_time=2, first=True), 2)
        self.assertEqual(servers.assign_server(
            t=13, service_time=1, first=True), -2)


if __name__ == '__main__':
    # _ = TestScheduledServer()
    # _.test_idle_till()

    unittest.main()
