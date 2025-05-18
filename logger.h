// logger.h - Comprehensive logging system

#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <chrono>
#include <mutex>
#include <atomic>
#include <fstream>
#include <sstream>
#include <iostream>

namespace tensor {

    // Log levels
    enum class LogLevel {
        DEBUG,
        INFO,
        WARN,
        ERROR,
        FATAL
    };

    // Forward declarations
    class LogSink;
    class Logger;

    // Singleton logger access
    Logger& get_logger();

    // Base class for log sinks
    class LogSink {
    public:
        virtual ~LogSink() = default;
        virtual void write(LogLevel level, const std::string& message,
            const std::unordered_map<std::string, std::string>& context) = 0;
        virtual void flush() = 0;
        virtual std::string name() const = 0;
    };

    // Console output sink
    class ConsoleSink : public LogSink {
    public:
        ConsoleSink(bool use_colors = true);
        void write(LogLevel level, const std::string& message,
            const std::unordered_map<std::string, std::string>& context) override;
        void flush() override;
        std::string name() const override;

    private:
        bool use_colors_;
        std::mutex console_mutex_;
    };

    // File output sink
    class FileSink : public LogSink {
    public:
        FileSink(const std::string& filename, bool append = true);
        ~FileSink();

        void write(LogLevel level, const std::string& message,
            const std::unordered_map<std::string, std::string>& context) override;
        void flush() override;
        std::string name() const override;

        // File rotation options
        void set_max_size(size_t max_size_bytes);
        void set_rotation_pattern(const std::string& pattern);

    private:
        std::string filename_;
        std::ofstream file_;
        std::mutex file_mutex_;
        size_t max_size_bytes_ = 0;
        std::string rotation_pattern_;

        void check_rotation();
    };

    // Main logger class
    class Logger {
    public:
        // Get singleton instance
        static Logger& instance();

        // Basic logging functions
        void log(LogLevel level, const std::string& message);
        void debug(const std::string& message);
        void info(const std::string& message);
        void warn(const std::string& message);
        void error(const std::string& message);
        void fatal(const std::string& message);

        // Configuration
        void set_level(LogLevel level);
        LogLevel get_level() const;

        // Sink management
        void add_sink(std::shared_ptr<LogSink> sink);
        void remove_sink(const std::string& sink_name);
        void clear_sinks();

        // Context management
        void add_context(const std::string& key, const std::string& value);
        void clear_context();

        // Format control
        void set_format(const std::string& format);

        // Performance tracking
        void start_timer(const std::string& name);
        void stop_timer(const std::string& name);

        // Async features
        void enable_async(size_t queue_size = 1000);
        void disable_async();
        void flush();

    private:
        Logger();
        ~Logger();

        // Prevent copying
        Logger(const Logger&) = delete;
        Logger& operator=(const Logger&) = delete;

        LogLevel level_ = LogLevel::INFO;
        std::vector<std::shared_ptr<LogSink>> sinks_;
        std::unordered_map<std::string, std::string> context_;
        std::string format_ = "[%time%] [%level%] %message%";
        std::mutex mutex_;

        // Timer tracking
        std::unordered_map<std::string, std::chrono::time_point<std::chrono::high_resolution_clock>> timers_;

        // Async logging
        bool async_enabled_ = false;
        std::shared_ptr<class AsyncLogQueue> async_queue_;

        // Helper methods
        std::string format_message(LogLevel level, const std::string& message);
    };

    // Convenient macros
#define LOG_DEBUG(message) tensor::get_logger().debug(message)
#define LOG_INFO(message) tensor::get_logger().info(message)
#define LOG_WARN(message) tensor::get_logger().warn(message)
#define LOG_ERROR(message) tensor::get_logger().error(message)
#define LOG_FATAL(message) tensor::get_logger().fatal(message)

// Context logging macro
#define LOG_WITH_CONTEXT(level, message, ...) /* implementation */

// Performance tracking macros
#define TIMED_BLOCK(name) /* implementation */

} // namespace tensor

#endif // LOGGER_H