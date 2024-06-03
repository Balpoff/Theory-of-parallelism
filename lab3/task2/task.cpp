#include <queue>
#include <unordered_map>
#include <cmath>
#include <thread>
#include <condition_variable>
#include <chrono>
#include <mutex>
#include <future>
#include <iostream>

std::mutex thread_lock;


template <typename T>
class safe_que {
    std::queue<T> que;
    std::mutex que_lock;
public:
    class safe_que_return_node {
        T* value = 0;
    public:
        void set_value(T val) {
            this->value = new T(val);
        }
        T* get_value() {
            return this->value;
        }
        ~safe_que_return_node() {
            delete value;
        }
    };
    void push(T val) {
        que_lock.lock();
        que.push(val);
        que_lock.unlock();
    }
    bool empty() {
        return que.empty();
    }
    T pop() {
        que_lock.lock();
        T val = que.front();
        que.pop();
        que_lock.unlock();
        return val;
    }
};

template <typename T>
class Task {
public:
    Task() { };
    virtual void say_name() = 0;
    virtual T do_task() = 0;
};

template <typename T>
class SinTask : public Task<T> {
    T arg1;
public:
    std::string task_name = "SinTask";
    SinTask(T arg1) {
        this->arg1 = arg1;
    }
    void say_name() {
        std::cout << task_name;
    }
    T do_task() {
        return std::sin(this->arg1);
    }
};
template <typename T>
class SqrtTask : public Task<T> {
    T arg1;
public:
    std::string task_name = "SqrtTask";
    SqrtTask(T arg1) {
        this->arg1 = arg1;
    }
    void say_name() {
        std::cout << task_name;
    }
    T do_task() {
        return std::sqrt(this->arg1);
    }
};
template <typename T>
class PowTask : public Task<T> {
    T arg1, arg2;
public:
    std::string task_name = "PowTask";
    PowTask(T arg1, T arg2) {
        this->arg1 = arg1;
        this->arg2 = arg2;
    }
    void say_name() {
        std::cout << task_name;
    }
    T do_task() {
        int asd= 0;
        for (int i = 0; i < 100000; i++) {
            asd = i;
        }
        return std::pow(this->arg1, this->arg2);
    }
};

template <typename T>
class Server {
private:
    struct task_with_id {
        Task<T>* task;
        size_t id;
    };
    std::condition_variable server_check;
    std::condition_variable client_check;
    size_t num_of_workers = 1;
    safe_que<task_with_id> task_que;
    safe_que<size_t> free_ids;
    size_t max_id = 0;
    std::unordered_map<size_t, T> task_result;
    std::vector<std::thread> event_thread_pool;
    bool running = false;
    bool stopped = true;
    std::mutex server_lock;
    std::mutex cv_client_lock;
    void event_loop() {
        while (running) {
            // locker to lock eventloop until the task is arrived
            std::unique_lock<std::mutex> locker(server_lock);
            while (task_que.empty()) {
                //std::cout << std::this_thread::get_id() << " thread have no tasks\n";
                // cv may work falsely, for this reason using while loop where we check that the task queue isn't empty
                server_check.wait(locker);
                //std::cout << std::this_thread::get_id() << " got the task signal\n";
                if (!running)
                    return;
            }
            locker.unlock();
            task_with_id task_struct;
            task_struct.id = -1;
            server_lock.lock();
            if (!task_que.empty())
                task_struct = task_que.pop();
            server_lock.unlock();
            if (task_struct.id != -1) {
                T return_value = task_struct.task->do_task();
                cv_client_lock.lock();
                task_result.insert(std::make_pair(task_struct.id, return_value));
                client_check.notify_all();
                cv_client_lock.unlock();
            }
        }
    }
    // give id to task. server give free id from queue or generate new id otherwise
    size_t get_free_id() {
        server_lock.lock();
        if (free_ids.empty()) {
            free_ids.push(max_id++);
        }
        size_t free_id = free_ids.pop();
        server_lock.unlock();
        return free_id;
    }
public:
    ~Server() {
        if (!stopped) {
            this->stop();
        }
    }
    void start(size_t num_of_workers = 1) {
        running = true;
        stopped = false;
        for (int i = 0; i < num_of_workers; i++) {
            event_thread_pool.push_back(std::thread(&Server::event_loop, this));
        }
    }
    void stop() {
        running = false;
        stopped = true;
        server_check.notify_all();
        for (std::thread& event_thread : this->event_thread_pool) {
            event_thread.join();
        }
    }
    size_t add_task(Task<T>* task) {
        size_t free_id = get_free_id();
        task_with_id task_to_add = { task, free_id };
        server_lock.lock();
        task_que.push(std::move(task_to_add));
        server_check.notify_one();
        server_lock.unlock();
        return task_to_add.id;
    }
    T request_result(size_t id) {
        std::unique_lock<std::mutex> locker(cv_client_lock);
        while (task_result.find(id) == task_result.end()) {
            client_check.wait(locker);
        }
        T result = task_result.at(id);
        server_lock.lock();
        task_result.erase(id);
        free_ids.push(id);
        server_lock.unlock();
        return result;
    }
};

template <typename T>
void give_task_to_server(Server<T>* server, Task<T>* task, int num_of_tasks) {
    std::vector<size_t> task_ids(num_of_tasks);
    for (int i = 0; i < num_of_tasks; i++) {
        task_ids[i] = server->add_task(task);
    }
    for (auto task_id : task_ids) {
        T result = server->request_result(task_id);
        //thread_lock.lock();
        //std::cout << result << " " << task_id << "\n";
        //thread_lock.unlock();
    }
}

int main() {
    Server<float> server;
    server.start(10);
    std::vector<Task<float>*> tasks = { new PowTask<float>(5.0f, 2.0f),
                                        new SinTask<float>(3.14 / 6),
                                        new SqrtTask<float>(25)
    };
    std::vector<std::thread> threads_l;
    const auto start{ std::chrono::steady_clock::now() };
    for (Task<float>* task : tasks) {
        std::thread th(give_task_to_server<float>, &server, task, 100000);
        threads_l.push_back(std::move(th));
    }
    for (auto& thread : threads_l) {
        thread.join();
    }
    server.stop();
    const auto end{ std::chrono::steady_clock::now() };
    const std::chrono::duration<double> elapsed_seconds{ end - start };
    std::cout << elapsed_seconds.count() << "\n";
    return 0;
}