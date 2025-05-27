#pragma once

#include <thread>
#include <functional>
#include <condition_variable>
#include <mutex>
#include <chrono>

class ThreadTimer {
public:
	enum Status {
		Stop       = 0,
		Start      = 1,
		Start_Once,
		Destroyed
	};

	inline ThreadTimer()
		: ThreadTimer(std::chrono::milliseconds(1000), {}) {}

	template <class Rep, class Period>
	inline ThreadTimer(std::chrono::duration<Rep, Period> interval, std::function<void()> callback, bool start = false)
		: status((Status)start), callback(callback), thread(timer_impl, this) {
		setInterval(interval);
	}

	inline ~ThreadTimer() {
		status = Destroyed;
		cd.notify_one();
		thread.join();
	}

	inline Status getStatus() const {
		return status;
	}

	template <class Rep, class Period>
	inline void setInterval(std::chrono::duration<Rep, Period> interval) {
		std::lock_guard<std::mutex> guard(mutex_params);
		this->interval = interval;
		if (status != Stop) cd.notify_one();
	}

	inline void setCallback(std::function<void()> callback) {
		std::lock_guard<std::mutex> guard(mutex_params);
		this->callback = callback;
	}

	inline void start() {
		std::lock_guard<std::mutex> guard(mutex_params);
		status = Start;
		cd.notify_one();
	}

	inline void start_once() {
		std::lock_guard<std::mutex> guard(mutex_params);
		status = Start_Once;
		cd.notify_one();
	}

	inline void stop() {
		std::lock_guard<std::mutex> guard(mutex_params);
		status = Stop;
		cd.notify_one();
	}

private:
	static void timer_impl(ThreadTimer* timer) {
		auto& mutex   = timer->mutex_cd;
		auto& cd      = timer->cd;
		auto& status  = timer->status;
		
		while (status != Destroyed) {
			if (status == Start || status == Start_Once) {
				std::unique_lock<std::mutex> lock(mutex);
				auto res = cd.wait_for(lock, timer->interval);
				if (res == std::cv_status::no_timeout) continue;

				timer->callback();
				if (status == Start_Once) status = Stop;
			} else if (status == Stop) {
				std::unique_lock<std::mutex> lock(mutex);
				cd.wait(lock); // idle
			}
		}
	}

private:
	Status                   status;
	std::chrono::nanoseconds interval;
	std::function<void()>    callback;
	std::mutex               mutex_params;
	std::mutex               mutex_cd;
	std::condition_variable  cd;
	std::thread              thread;
};
