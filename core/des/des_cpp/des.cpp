#include <iostream>
#include <utility>
#include <vector>
#include <list>
//#include <pybind11/pybind11.h>
#include <algorithm>
#include <numeric>
#include <limits>
#include <random>
#include <ctime>
#include <stdlib.h>
#include "des.hpp"

// TODO use unit test https://www.jetbrains.com/help/clion/unit-testing-tutorial.html#adding-framework
using namespace std;

// for Event class
ostream &operator<<(ostream &os, const Event &event)
{
    os << "event time\t" << event.time << " event type\t" << event.type << " service length\t"
       << event.service << " arrival_id\t"
       << event.arrival_id << " server_id\t" << event.server_id;
    return os;
}

Event::Event(float time, string type, float service, int &arrival_id) : time(time), type(type), service(service), arrival_id(arrival_id)
{
    // assert type == 'arrival'
    // assert(type == "arrival");
    if(type != "arrival"){
        exit (EXIT_FAILURE);
    }

    // set the unused server_id to -1
    server_id = -1;
}

Event::Event(float time, string type, int &arrival_id, int &server_id) : time(time), type(type), arrival_id(arrival_id), server_id(server_id), service(-1)
{
    // assert(type == "busy_till");
    if(type != "busy_till"){
        exit (EXIT_FAILURE);
    }
    // set the unused service to -1
}

float Event::get_time() const
{
    return time;
}

vector<float>
multi_server_queue(const vector<float> &arrival_ls, const vector<float> &service_ls, const vector<float> &break_point,
                   const vector<int> &n_server, bool verbose)
{
    ScheduledServerList servers(break_point, n_server);

    EventList event_list;
    Queue queue;

    int N = arrival_ls.size();
    for (int i = 0; i < N; i++)
    {
        Event new_event(arrival_ls[i], "arrival", service_ls[i], i);
        if (verbose)
        {
            cout << new_event << endl;
        }

        event_list.add_event(new_event);
    }

    // wait_ls is vector<float> of length N and all elements are -1
    vector<float> wait_ls(N, -1);
    if (verbose)
    {
        cout << "initialize wait_ls" << endl;
    }
    int server_id;

    while (not event_list.empty())
    {
        Event next_event = event_list.next_event();
        if (verbose)
        {
            cout << "next event: " << next_event << endl;
        }
        if (next_event.type == "arrival")
        {
            if (queue.empty())
            {
                server_id = servers.assign_server(next_event.time, next_event.service, true);
                switch (server_id)
                {
                case -1:
                {
                    Customer customer(next_event.time, next_event.service, next_event.arrival_id);
                    queue.append(customer);
                    if (verbose)
                    {
                        cout << "append customer with id " << customer.arrival_id << " to queue" << endl;
                    }
                    break;
                }
                case -2:
                {
                    return wait_ls;
                    break;
                }
                default:
                {
                    //  Event(torch::Tensor time, string type, int &arrival_id, int &server_id) :
                    Event event_busy_till(next_event.time + next_event.service, "busy_till", next_event.arrival_id,
                                          server_id);
                    event_list.add_event(event_busy_till);
                    //                        if (verbose) {cout << "Assigned server " << server_id << " for arrival_id " << next_event.arrival_id << endl;}

                    if (verbose)
                    {
                        cout << "add event: " << event_busy_till << endl;
                    }
                    wait_ls[next_event.arrival_id] = 0;
                }
                }
            }
            else
            {
                Customer customer(next_event.time, next_event.service, next_event.arrival_id);
                queue.append(customer);
                if (verbose)
                {
                    cout << "append customer with id " << customer.arrival_id << " to queue" << endl;
                }
            }
        }
        if (next_event.type == "busy_till")
        {
            server_id = next_event.server_id;
            int arrival_id = next_event.arrival_id;
            float exit_time = next_event.time;

            if (not queue.empty())
            {
                Customer next_customer = queue.get();
                server_id = servers.assign_server(exit_time, next_customer.service, true);
                switch (server_id)
                {
                case -1:
                {
                    queue.append(next_customer, true);
                    break;
                }
                case -2:
                {
                    return wait_ls;
                    break;
                }
                default:
                {
                    arrival_id = next_customer.arrival_id;
                    Event event_busy_till(exit_time + next_customer.service, "busy_till", arrival_id, server_id);
                    wait_ls[arrival_id] = exit_time - next_customer.arrival;
                    if (verbose)
                    {
                        cout << "Assigned server " << server_id << " for arrival_id " << arrival_id << endl;
                    }
                    event_list.add_event(event_busy_till);
                }
                }
            }
        }
    }
    return wait_ls;
}

float ScheduledServer::idle_till(float t)
{
    Struct result = whether_idle(t);
    if (result.whether_idle)
    {
        if (result.shift_id + 1 == n_shift)
        {
            float inf = numeric_limits<float>::infinity();
            return inf;
        }
        else
        {
            return shift_time_ls[result.shift_id + 1];
        }
    }
    else
    {
        // if shift_type_ls index from result.shift_id + 1 to end contains idle
        if (find(shift_type_ls.begin() + result.shift_id + 1, shift_type_ls.end(), "idle") !=
            shift_type_ls.end())
        {
            return -1;
        }
        else
        {
            return -2;
        }
    }
}

int main()
{
    // test ScheduledServerList
    if (false)
    {
        // 22-6-5
        int T = 3;
        vector<float> break_point;
        for (int i = 0; i <= T; i++)
        {
            break_point.push_back(i);
        }
        vector<int> n_server = {4, 2, 3};

        ScheduledServerList servers(break_point, n_server);

        // assert(servers.server_ls[0].shift_time_ls[0] == 0);
        // assert(servers.server_ls[0].shift_time_ls[1] == 3);
        // assert(servers.server_ls[0].shift_type_ls[0] == "idle");
        // assert(servers.server_ls[0].shift_type_ls[1] == "busy");
        // assert(servers.server_ls[0].shift_type_ls.size() == 2);

        // assert(servers.server_ls[1].shift_time_ls[0] == 0);
        // assert(servers.server_ls[1].shift_time_ls[1] == 3);
        // assert(servers.server_ls[1].shift_type_ls[0] == "idle");
        // assert(servers.server_ls[1].shift_type_ls[1] == "busy");
        // assert(servers.server_ls[1].shift_type_ls.size() == 2);

        // assert(servers.server_ls[2].shift_time_ls[0] == 0);
        // assert(servers.server_ls[2].shift_time_ls[1] == 1);
        // assert(servers.server_ls[2].shift_time_ls[2] == 2);
        // assert(servers.server_ls[2].shift_time_ls[3] == 3);
        // assert(servers.server_ls[2].shift_type_ls[0] == "idle");
        // assert(servers.server_ls[2].shift_type_ls[1] == "busy");
        // assert(servers.server_ls[2].shift_type_ls[2] == "idle");
        // assert(servers.server_ls[2].shift_type_ls[3] == "busy");
        // assert(servers.server_ls[2].shift_type_ls.size() == 4);

        // assert(servers.server_ls[3].shift_time_ls[0] == 0);
        // assert(servers.server_ls[3].shift_time_ls[1] == 1);
        // assert(servers.server_ls[3].shift_type_ls[0] == "idle");
        // assert(servers.server_ls[3].shift_type_ls[1] == "busy");
        // assert(servers.server_ls[3].shift_type_ls.size() == 2);

        cout << "ScheduledServerList test passed" << endl;
    }

    // test infinite server_queue
    //    auto requires_grad = torch::TensorOptions().requires_grad(true);
    //    torch::Tensor arrival_ls = at::cumsum(torch::rand(10000, requires_grad), 0);
    //    torch::Tensor service_ls = torch::rand(10000, requires_grad);
    //    torch::Tensor eval_t_ls = torch::linspace(0.5, 10.5, 10);
    //    torch::Tensor n_occupied_ls = infinite_server_queue(arrival_ls, service_ls, eval_t_ls);

    //    cout << n_occupied_ls << endl;

    // test multi_server_queue
    if (true)
    {
        int n_arrival = 10000;
        int T = 20;
        int n_interval = 5;

        float mean_service = 0.2;

        default_random_engine generator;
        uniform_real_distribution<float> arrival_distribution(0.0, T);
        uniform_real_distribution<float> service_distribution(0.0, 2 * mean_service);
        //
        // arrival_ls is a vector of length n_arrival, elements are random float between 0 and T
        vector<float> arrival_ls;
        vector<float> service_ls;
        //
        //
        int n_rep = 1;

        for (int j = 0; j < n_rep; j++)
        {

            arrival_ls.clear();
            service_ls.clear();

            for (int i = 0; i < n_arrival; i++)
            {
                arrival_ls.push_back(arrival_distribution(generator));
                service_ls.push_back(service_distribution(generator));
            }

            vector<float> break_point;
            for (int i = 0; i <= n_interval; i++)
            {
                break_point.push_back(i * T / n_interval);
            }
            // n_server is vector of length k, elements are all 3

            vector<int> n_server(n_interval, int(n_arrival * mean_service) / T + 1);
            EventList event_list;
            Queue queue;

            clock_t start = clock();
            vector<float> wait_ls = multi_server_queue(arrival_ls, service_ls, break_point, n_server, false);
            clock_t stop = clock();
            for (int i = 0; i < wait_ls.size(); i++)
            {
                cout << wait_ls[i] << ' ';
            }
            cout << endl;
            double elapsed = (double)(stop - start) / CLOCKS_PER_SEC;
            printf("\nTime elapsed: %.5f\n", elapsed);
        }
    }
}
