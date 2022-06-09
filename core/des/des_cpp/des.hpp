#include <string>
#include <iostream>
#include <utility>
#include <vector>
#include <list>
#include <algorithm>
#include <numeric>
#include <limits>
#include <random>
#include <ctime>
#include <stdlib.h>

using namespace std;

class Event
{
public:
    float time;
    string type;
    float service;
    int arrival_id;
    int server_id;

    Event(float time, string type, float service, int &arrival_id);

    Event(float time, string type, int &arrival_id, int &server_id);

    float get_time() const;

    friend ostream &operator<<(ostream &os, const Event &event);
};

class EventList
{
public:
    list<Event> event_list;

    EventList() = default;

    void add_event(Event new_event)
    {
        // add the new event to the event list
        // maintain the event list in ascending order of time

        // get the id that insert new_event to
        event_list.push_back(new_event);
    }

    bool empty() const
    {
        return event_list.empty();
    }

    Event next_event()
    {
        // we actually don't need to call sort_event_list every time, just maintain a sorted list in the beginning and
        // put new event to correct position.
        //        sort_event_list();

        //        print_event_list();
        // get the event with the smallest time, and remove it from the list
        //        Event next_event = event_list.front();
        //        event_list.erase(event_list.begin());

        // find the smallest time in the event list

        auto minElement = min_element(
            event_list.begin(), event_list.end(),
            [](const Event &a, const Event &b)
            { return a.get_time() < b.get_time(); });

        Event next_event = *minElement;
        event_list.erase(minElement);
        return next_event;
    }

    //    void sort_event_list() {
    //        // sort the event_list by time in ascending order
    //        sort(event_list.begin(), event_list.end(),
    //             [](const Event &a, const Event &b) { return a.get_time() < b.get_time(); });
    //    }

    void print_event_list()
    {
        for (auto &event : event_list)
        {
            cout << event << endl;
        }
    }
};

class Customer
{
public:
    float arrival;
    float service;
    int arrival_id;

    Customer(float time, float service, int arrival_id) : arrival(time), service(service), arrival_id(arrival_id) {}
};

class Queue
{
public:
    vector<Customer> customer_list;

    Queue() = default;

    void append(Customer new_customer, bool put_back = false)
    {
        if (not put_back)
            customer_list.push_back(new_customer);
        else
            customer_list.insert(customer_list.begin(), new_customer);
    }

    bool empty() const
    {
        return customer_list.empty();
    }

    Customer get()
    {
        Customer customer = customer_list[0];
        customer_list.erase(customer_list.begin());
        return customer;
    }
};

class ScheduledServer
{
public:
    vector<float> shift_time_ls;
    vector<string> shift_type_ls;
    //    torch::TensorAccessor<float, 1> shift_time_ls_a = torch::TensorAccessor<float, 1>(nullptr, nullptr, nullptr);

    ScheduledServer(vector<float> shift_time_ls, vector<string> shift_type_ls)
    {
        //        shift_time_ls_a = shift_time_ls.accessor<float, 1>();
        // TODO combine two neighboring identical shift type into one shift type
        // the following code is now replaced by remove_duplicate (not tested)
        //        for (int i = 1; i < shift_time_ls.size(); i++) {
        //            string curr_shift_type = shift_type_ls[i];
        //            string prev_shift_type = shift_type_ls[i - 1];
        //            if (curr_shift_type == prev_shift_type) {
        //                shift_time_ls.erase(shift_time_ls.begin() + i);
        //                shift_type_ls.erase(shift_type_ls.begin() + i);
        //                i -= 1;
        //            }
        //        }
        this->shift_time_ls = shift_time_ls;
        this->shift_type_ls = shift_type_ls;
        remove_duplicate();
        n_shift = this->shift_time_ls.size();
    }

    float idle_till(float t);

    void set_busy_till(float busy_start, float busy_end)
    {
        // set server status to busy from busy_start till time busy_end
        // verify the code
        int shift_id = get_shift_id(busy_start);
        shift_time_ls.insert(shift_time_ls.begin() + shift_id + 1, busy_start);
        shift_time_ls.insert(shift_time_ls.begin() + shift_id + 2, busy_end);
        shift_type_ls.insert(shift_type_ls.begin() + shift_id + 1, "busy");
        shift_type_ls.insert(shift_type_ls.begin() + shift_id + 2, "idle");
    }

private:
    int n_shift;
    struct whether_idle_shift_id
    {
        bool whether_idle;
        int shift_id;
    };

    typedef struct whether_idle_shift_id Struct;

    Struct whether_idle(float t)
    {
        Struct result;
        if (t < shift_time_ls[0])
        {
            result.whether_idle = false;
            result.shift_id = -1;
        }
        else
        {
            // find the last element in shift_time_ls that is smaller than or equal to t
            result.shift_id = get_shift_id(t);
            result.whether_idle = (shift_type_ls[result.shift_id] == "idle");
        }
        return result;
    }

    int get_shift_id(float t)
    {
        int index = lower_bound(shift_time_ls.begin(), shift_time_ls.end(), t) - shift_time_ls.begin() - 1;
        return index;
    }

    void remove_duplicate()
    {

        vector<int> keep_index;
        keep_index.push_back(0);
        for (int i = 1; i < shift_type_ls.size(); i++)
        {
            if (shift_type_ls[i] != shift_type_ls[i - 1])
            {
                keep_index.push_back(i);
            }
        }

        // TODO fix some bug here, mainly keep_index here

        vector<string> shift_type_ls_new;
        vector<float> shift_time_ls_new;
        for (int i = 0; i < keep_index.size(); i++)
        {
            shift_type_ls_new.push_back(shift_type_ls[keep_index[i]]);
            shift_time_ls_new.push_back(shift_time_ls[keep_index[i]]);
        }
        shift_type_ls = shift_type_ls_new;
        shift_time_ls = shift_time_ls_new;
    };
};

class ScheduledServerList
{
public:
    vector<ScheduledServer> server_ls;
    ScheduledServerList(vector<ScheduledServer> server_ls)
    {
        this->server_ls = server_ls;
    }

    ScheduledServerList(vector<float> break_point, vector<int> n_server)
    {

        // assert(break_point.size() == n_server.size() + 1);
        if (break_point.size() != n_server.size() + 1)
        {
            cout << "break_point.size() != n_server.size() + 1" << endl;
            exit(1);
        }
        int max_n_server = *max_element(n_server.begin(), n_server.end());
        server_ls = vector<ScheduledServer>();
        for (int i = 0; i < max_n_server; i++)
        {
            // torch::Tensor &shift_time_ls, vector<string> &shift_type_ls) {
            vector<string> status_ls;
            for (int k = 0; k < break_point.size() - 1; k++)
            {
                if (n_server[k] >= i + 1)
                {
                    status_ls.emplace_back("idle");
                }
                else
                {
                    status_ls.emplace_back("busy");
                }
            }
            status_ls.emplace_back("busy");
            server_ls.emplace_back(ScheduledServer(break_point, status_ls));
        }
    }

    int assign_server(float t, float service_time, bool first = false)
    {
        /*Get server id.

        randomly return the server that is idle from time t to t + service_time

        return -1 if no server meet the requirement at time t but at least one server can fulfill the requirement in the future

        return -2 if no server meet the requirement at time t and all servers will keep in 'home' or 'busy' status forever

        service_time > 0
        */
        // find all idle servers and compute idle till when
        vector<float> idle_till_ls;
        for (auto &server_l : server_ls)
        {
            idle_till_ls.push_back(server_l.idle_till(t));
            //            cout << "idle_till: " << idle_till_ls.back().item<float>() << endl;
        }

        float service_end = t + service_time;

        std::vector<int> can_fullfill;
        for (auto idle_till : idle_till_ls)
        {
            if (idle_till == -2)
            {
                // -2 means this server will keep in 'home' or 'busy' status forever
                can_fullfill.push_back(-2);
            }
            else if ((idle_till == -1) or (service_end > idle_till))
            {
                // -1 means this server cannot fulfill the requirement at time t but can fulfill the requirement in the future
                can_fullfill.push_back(-1);
            }
            else
            {
                // 0 means this server can fulfill the requirement at time t
                can_fullfill.push_back(0);
            }
        }

        int server_id;

        auto can_fullfill_index_0 = std::find(can_fullfill.begin(), can_fullfill.end(), 0);
        // if can_fullfill contains 0
        if (can_fullfill_index_0 != can_fullfill.end())
        {
            if (first)
                server_id = distance(can_fullfill.begin(), can_fullfill_index_0);
            else
            {
                // get location of 0 in can_fullfill
                vector<int> can_fullfill_index;
                for (int i = 0; i < can_fullfill.size(); ++i)
                {
                    if (can_fullfill[i] == 0)
                    {
                        can_fullfill_index.push_back(i);
                    }
                }
                // randomly select one from can_fullfill_index
                server_id = can_fullfill_index[rand() % can_fullfill_index.size()];
            }
            server_ls[server_id].set_busy_till(t, t + service_time);
            return server_id;
        }
        else if (std::find(can_fullfill.begin(), can_fullfill.end(), -1) != can_fullfill.end())
        {
            return -1;
        }
        else
        {
            return -2;
        }
    }
};

vector<float> multi_server_queue(const vector<float> &arrival_ls, const vector<float> &service_ls, const vector<float> &break_point,
                                 const vector<int> &n_server, bool verbose = false);
