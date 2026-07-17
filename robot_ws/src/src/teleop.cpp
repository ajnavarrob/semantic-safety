#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/int32.hpp>

#include <ncurses.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <sys/types.h>
#include <unistd.h>

using namespace std::chrono_literals;

class TeleopNode : public rclcpp::Node
{
public:
    TeleopNode()
    : Node("teleop_node")
    {
        /*
         * Teleoperation parameters
         */
        max_forward_velocity_ =
            this->declare_parameter<double>("max_forward_velocity", 0.90);

        max_backward_velocity_ =
            this->declare_parameter<double>("max_backward_velocity", 0.90);

        max_lateral_velocity_ =
            this->declare_parameter<double>("max_lateral_velocity", 0.90);

        max_yaw_velocity_ =
            this->declare_parameter<double>("max_yaw_velocity", 0.80);

        linear_step_ =
            this->declare_parameter<double>("linear_step", 0.10);

        yaw_step_ =
            this->declare_parameter<double>("yaw_step", 0.10);

        /*
         * Heartbeat parameters
         */
        ethernet_interface_ =
            this->declare_parameter<std::string>(
                "ethernet_interface",
                "eth0");

        heartbeat_host_ =
            this->declare_parameter<std::string>(
                "heartbeat_host",
                "192.168.123.222");

        heartbeat_period_ms_ =
            this->declare_parameter<int>(
                "heartbeat_period_ms",
                500);

        heartbeat_timeout_seconds_ =
            this->declare_parameter<int>(
                "heartbeat_timeout_seconds",
                1);

        heartbeat_failure_limit_ =
            this->declare_parameter<int>(
                "heartbeat_failure_limit",
                2);

        zero_publish_cycles_ =
            this->declare_parameter<int>(
                "zero_publish_cycles",
                10);

        /*
         * ROS publishers
         */
        velocity_publisher_ =
            this->create_publisher<geometry_msgs::msg::Twist>(
                "u_des",
                10);

        key_publisher_ =
            this->create_publisher<std_msgs::msg::Int32>(
                "key_press",
                10);

        /*
         * Initialize ncurses
         */
        initscr();
        cbreak();
        noecho();
        keypad(stdscr, TRUE);

        // Make getch() nonblocking.
        nodelay(stdscr, TRUE);

        // Hide terminal cursor.
        curs_set(0);

        printw("Unitree teleoperation\n");
        printw("---------------------\n");
        printw("Up/down:    forward/backward\n");
        printw("Left/right: lateral motion\n");
        printw(", and .:    yaw\n");
        printw("Space:      stop\n");
        printw("q:          quit\n\n");
        printw("Heartbeat host: %s via %s\n",
               heartbeat_host_.c_str(),
               ethernet_interface_.c_str());
        refresh();

        /*
         * Keyboard and velocity-publishing timer.
         */
        keyboard_timer_ =
            this->create_wall_timer(
                50ms,
                std::bind(
                    &TeleopNode::keyboard_callback,
                    this));

        RCLCPP_INFO(
            this->get_logger(),
            "Velocity bounds: x_fwd=%.2f, x_bwd=%.2f, y=%.2f, yaw=%.2f",
            max_forward_velocity_,
            max_backward_velocity_,
            max_lateral_velocity_,
            max_yaw_velocity_);

        RCLCPP_INFO(
            this->get_logger(),
            "Monitoring PC heartbeat at %s through interface '%s'",
            heartbeat_host_.c_str(),
            ethernet_interface_.c_str());

        /*
         * Run heartbeat checks in a separate thread.
         *
         * This prevents ping timeouts from blocking keyboard handling.
         */
        heartbeat_thread_ =
            std::thread(
                &TeleopNode::heartbeat_loop,
                this);
    }

    ~TeleopNode() override
    {
        stop_heartbeat_thread_.store(true);

        if (heartbeat_thread_.joinable()) {
            heartbeat_thread_.join();
        }

        publish_zero_velocity();

        endwin();
    }

private:
    /*
     * Keyboard handling
     */
    void keyboard_callback()
    {
        if (emergency_shutdown_started_.load()) {
            return;
        }

        const int key = getch();

        bool command_changed = false;

        {
            std::lock_guard<std::mutex> lock(command_mutex_);

            switch (key) {
                case KEY_UP:
                    command_.linear.x += linear_step_;
                    command_changed = true;
                    break;

                case KEY_DOWN:
                    command_.linear.x -= linear_step_;
                    command_changed = true;
                    break;

                case KEY_LEFT:
                    command_.linear.y += linear_step_;
                    command_changed = true;
                    break;

                case KEY_RIGHT:
                    command_.linear.y -= linear_step_;
                    command_changed = true;
                    break;

                case ',':
                    command_.angular.z += yaw_step_;
                    command_changed = true;
                    break;

                case '.':
                    command_.angular.z -= yaw_step_;
                    command_changed = true;
                    break;

                case ' ':
                    set_command_to_zero_locked();
                    command_changed = true;
                    break;

                case 'q':
                case 'Q':
                    set_command_to_zero_locked();
                    velocity_publisher_->publish(command_);

                    stop_heartbeat_thread_.store(true);
                    rclcpp::shutdown();
                    return;

                case ERR:
                    // No key was pressed.
                    break;

                default:
                    /*
                     * Preserve publication of keyboard input for other nodes.
                     */
                    publish_key(key);
                    break;
            }

            if (command_changed) {
                clamp_command_locked();
                idle_counter_ = 0;

                publish_key(key);
            } else {
                ++idle_counter_;
            }

            /*
             * Stop the robot if no relevant key has been received recently.
             *
             * At 50 ms per callback, 20 cycles is approximately one second.
             */
            if (idle_counter_ >= idle_limit_) {
                set_command_to_zero_locked();
            }

            velocity_publisher_->publish(command_);
        }
    }

    void publish_key(const int key)
    {
        if (key == ERR || emergency_shutdown_started_.load()) {
            return;
        }

        std_msgs::msg::Int32 message;
        message.data = key;
        key_publisher_->publish(message);
    }

    void clamp_command_locked()
    {
        command_.linear.x =
            std::clamp(
                command_.linear.x,
                -max_backward_velocity_,
                max_forward_velocity_);

        command_.linear.y =
            std::clamp(
                command_.linear.y,
                -max_lateral_velocity_,
                max_lateral_velocity_);

        command_.angular.z =
            std::clamp(
                command_.angular.z,
                -max_yaw_velocity_,
                max_yaw_velocity_);
    }

    void set_command_to_zero_locked()
    {
        command_.linear.x = 0.0;
        command_.linear.y = 0.0;
        command_.linear.z = 0.0;

        command_.angular.x = 0.0;
        command_.angular.y = 0.0;
        command_.angular.z = 0.0;
    }

    void publish_zero_velocity()
    {
        geometry_msgs::msg::Twist zero_command;

        if (velocity_publisher_) {
            velocity_publisher_->publish(zero_command);
        }
    }

    /*
     * Heartbeat monitoring
     */
    void heartbeat_loop()
    {
        int consecutive_failures = 0;
        bool heartbeat_seen_once = false;

        while (
            rclcpp::ok() &&
            !stop_heartbeat_thread_.load() &&
            !emergency_shutdown_started_.load())
        {
            const bool reachable = heartbeat_is_reachable();

            if (reachable) {
                consecutive_failures = 0;

                if (!heartbeat_seen_once) {
                    heartbeat_seen_once = true;

                    RCLCPP_WARN(
                        this->get_logger(),
                        "Communication watchdog ARMED: %s is reachable through '%s'",
                        heartbeat_host_.c_str(),
                        ethernet_interface_.c_str());
                }
            } else if (heartbeat_seen_once) {
                ++consecutive_failures;

                RCLCPP_WARN(
                    this->get_logger(),
                    "Heartbeat to %s failed: %d/%d",
                    heartbeat_host_.c_str(),
                    consecutive_failures,
                    heartbeat_failure_limit_);

                if (
                    consecutive_failures >=
                    heartbeat_failure_limit_)
                {
                    RCLCPP_ERROR(
                        this->get_logger(),
                        "Communication with %s was lost",
                        heartbeat_host_.c_str());

                    trigger_emergency_shutdown();
                    return;
                }
            } else {
                /*
                 * Do not shut down at startup before the PC has been seen.
                 *
                 * This prevents an immediate shutdown if teleOp starts before
                 * the PC or network connection is ready.
                 */
                RCLCPP_WARN(
                    this->get_logger(),
                    "Waiting for initial heartbeat from %s",
                    heartbeat_host_.c_str());
            }

            sleep_interruptibly(heartbeat_period_ms_);
        }
    }

    bool heartbeat_is_reachable() const
    {
        /*
         * -I: force ping through the desired interface
         * -c 1: send one packet
         * -W: timeout for receiving a response
         *
         * The heartbeat thread is separate from ROS keyboard callbacks, so
         * this blocking command does not freeze keyboard control.
         */
        const std::string command =
            "ping"
            " -I " + shell_quote(ethernet_interface_) +
            " -c 1"
            " -W " + std::to_string(heartbeat_timeout_seconds_) +
            " " + shell_quote(heartbeat_host_) +
            " >/dev/null 2>&1";

        const int result = std::system(command.c_str());

        return result == 0;
    }

    static std::string shell_quote(const std::string &value)
    {
        /*
         * Parameters currently contain fixed IP/interface values, but quoting
         * them also prevents accidental shell interpretation.
         */
        std::string quoted = "'";

        for (const char character : value) {
            if (character == '\'') {
                quoted += "'\\''";
            } else {
                quoted += character;
            }
        }

        quoted += "'";
        return quoted;
    }

    void sleep_interruptibly(const int total_milliseconds)
    {
        constexpr int sleep_slice_ms = 50;

        int elapsed_ms = 0;

        while (
            elapsed_ms < total_milliseconds &&
            rclcpp::ok() &&
            !stop_heartbeat_thread_.load() &&
            !emergency_shutdown_started_.load())
        {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(sleep_slice_ms));

            elapsed_ms += sleep_slice_ms;
        }
    }

    /*
     * Emergency shutdown
     */
    void trigger_emergency_shutdown()
    {
        /*
         * Ensure this sequence can execute only once.
         */
        bool expected = false;

        if (!emergency_shutdown_started_.compare_exchange_strong(
                expected,
                true))
        {
            return;
        }

        RCLCPP_ERROR(
            this->get_logger(),
            "EMERGENCY SHUTDOWN TRIGGERED");

        /*
         * Preserve the command that existed at the instant communication
         * failed for diagnostics. It is not republished.
         */
        geometry_msgs::msg::Twist command_at_disconnect;

        {
            std::lock_guard<std::mutex> lock(command_mutex_);

            command_at_disconnect = command_;
            set_command_to_zero_locked();
        }

        RCLCPP_ERROR(
            this->get_logger(),
            "Command at disconnect: vx=%.3f, vy=%.3f, wz=%.3f",
            command_at_disconnect.linear.x,
            command_at_disconnect.linear.y,
            command_at_disconnect.angular.z);

        /*
         * Repeatedly publish an actual zero command.
         *
         * No decayed sequence is generated.
         */
        geometry_msgs::msg::Twist zero_command;

        for (int cycle = 0; cycle < zero_publish_cycles_; ++cycle) {
            if (!rclcpp::ok()) {
                break;
            }

            velocity_publisher_->publish(zero_command);

            std::this_thread::sleep_for(50ms);
        }

        terminate_semantic_poisson();

        RCLCPP_ERROR(
            this->get_logger(),
            "Emergency shutdown sequence complete; stopping teleOp");

        stop_heartbeat_thread_.store(true);
        rclcpp::shutdown();
    }

    void terminate_semantic_poisson()
    {
        const std::vector<pid_t> pids =
            find_semantic_poisson_pids();

        if (pids.empty()) {
            RCLCPP_WARN(
                this->get_logger(),
                "No semantic_poisson process was found");

            return;
        }

        RCLCPP_ERROR(
            this->get_logger(),
            "Found %zu semantic_poisson process(es)",
            pids.size());

        /*
         * First request a clean ROS shutdown.
         */
        for (const pid_t pid : pids) {
            send_signal_to_process(pid, SIGINT, "SIGINT");
        }

        std::this_thread::sleep_for(750ms);

        /*
         * Escalate to SIGTERM if necessary.
         */
        for (const pid_t pid : pids) {
            if (process_is_alive(pid)) {
                send_signal_to_process(pid, SIGTERM, "SIGTERM");
            }
        }

        std::this_thread::sleep_for(750ms);

        /*
         * Final escalation prevents the node from remaining alive.
         */
        for (const pid_t pid : pids) {
            if (process_is_alive(pid)) {
                send_signal_to_process(pid, SIGKILL, "SIGKILL");
            }
        }

        std::this_thread::sleep_for(100ms);

        for (const pid_t pid : pids) {
            if (process_is_alive(pid)) {
                RCLCPP_ERROR(
                    this->get_logger(),
                    "semantic_poisson PID %d is still alive",
                    static_cast<int>(pid));
            } else {
                RCLCPP_WARN(
                    this->get_logger(),
                    "semantic_poisson PID %d terminated",
                    static_cast<int>(pid));
            }
        }
    }

    std::vector<pid_t> find_semantic_poisson_pids() const
    {
        std::vector<pid_t> pids;

        /*
         * The bracketed first character prevents pgrep from matching its own
         * command line while still matching semantic_poisson.
         */
        FILE *pipe = popen(
            "pgrep -f '[s]emantic_poisson'",
            "r");

        if (pipe == nullptr) {
            RCLCPP_ERROR(
                this->get_logger(),
                "Could not execute pgrep: %s",
                std::strerror(errno));

            return pids;
        }

        char buffer[128];

        while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            char *end_pointer = nullptr;

            errno = 0;

            const long parsed_pid =
                std::strtol(
                    buffer,
                    &end_pointer,
                    10);

            if (
                errno == 0 &&
                end_pointer != buffer &&
                parsed_pid > 1)
            {
                const pid_t pid =
                    static_cast<pid_t>(parsed_pid);

                /*
                 * Never signal teleOp itself.
                 */
                if (pid != ::getpid()) {
                    pids.push_back(pid);
                }
            }
        }

        const int pclose_result = pclose(pipe);

        if (pclose_result == -1) {
            RCLCPP_WARN(
                this->get_logger(),
                "pclose failed after pgrep: %s",
                std::strerror(errno));
        }

        return pids;
    }

    bool process_is_alive(const pid_t pid) const
    {
        if (::kill(pid, 0) == 0) {
            return true;
        }

        /*
         * EPERM means the process exists but this user cannot signal it.
         */
        return errno == EPERM;
    }

    void send_signal_to_process(
        const pid_t pid,
        const int signal_number,
        const char *signal_name)
    {
        RCLCPP_WARN(
            this->get_logger(),
            "Sending %s to semantic_poisson PID %d",
            signal_name,
            static_cast<int>(pid));

        if (::kill(pid, signal_number) != 0) {
            RCLCPP_ERROR(
                this->get_logger(),
                "Failed to send %s to PID %d: %s",
                signal_name,
                static_cast<int>(pid),
                std::strerror(errno));
        }
    }

    /*
     * ROS interfaces
     */
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr
        velocity_publisher_;

    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr
        key_publisher_;

    rclcpp::TimerBase::SharedPtr keyboard_timer_;

    /*
     * Current teleoperation command
     */
    geometry_msgs::msg::Twist command_;
    std::mutex command_mutex_;

    int idle_counter_{0};
    const int idle_limit_{20};

    /*
     * Velocity parameters
     */
    double max_forward_velocity_{0.90};
    double max_backward_velocity_{0.90};
    double max_lateral_velocity_{0.90};
    double max_yaw_velocity_{0.80};

    double linear_step_{0.10};
    double yaw_step_{0.10};

    /*
     * Heartbeat parameters
     */
    std::string ethernet_interface_{"eth0"};
    std::string heartbeat_host_{"192.168.123.222"};

    int heartbeat_period_ms_{500};
    int heartbeat_timeout_seconds_{1};
    int heartbeat_failure_limit_{2};
    int zero_publish_cycles_{10};

    /*
     * Heartbeat state
     */
    std::thread heartbeat_thread_;

    std::atomic<bool> stop_heartbeat_thread_{false};
    std::atomic<bool> emergency_shutdown_started_{false};
};


int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    const auto node =
        std::make_shared<TeleopNode>();

    rclcpp::spin(node);

    if (rclcpp::ok()) {
        rclcpp::shutdown();
    }

    return 0;
}