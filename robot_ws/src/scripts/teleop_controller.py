#!/usr/bin/env python3
"""
Xbox Controller Teleop Node using 'inputs' library.
Works headlessly without display server.

Controller mapping (Xbox):
- Left Stick Y: Forward/Backward (vx)
- Left Stick X: Strafe Left/Right (vy)  
- Right Stick X: Rotation (vyaw)
- A Button: Space (start/save/stop)
- B Button: 'r' (toggle realtime safety filter)
- X Button: 'p' (toggle predictive safety filter)
- Y Button: 'd' (deal parameter)
- Start Button: 'q' (quit)
- D-Pad: Number keys 1-4 for wn parameter
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32
import threading
import os
import signal
import subprocess
import time
import inputs


class TeleopControllerNode(Node):
    def __init__(self):
        super().__init__('teleop_controller')
        
        # Parameters (Independent execution, no ROS params)
        self.vel_max_x_fwd = 1.1
        self.vel_max_x_bwd = 0.75
        self.vel_max_y = 0.75
        self.vel_max_yaw = 0.75
        self.deadzone = 0.1
        publish_rate = 100.0
        
        self.get_logger().info(
            f'Velocity bounds: x_fwd={self.vel_max_x_fwd:.2f}, x_bwd={self.vel_max_x_bwd:.2f}, '
            f'y={self.vel_max_y:.2f}, yaw={self.vel_max_yaw:.2f}'
        )
        
        # Publishers
        self.twist_pub = self.create_publisher(Twist, 'u_des', 1)
        self.key_pub = self.create_publisher(Int32, 'key_press', 1)
        
        # Controller state (normalized -1 to 1)
        self.axis_lx = 0.0  # Left stick X
        self.axis_ly = 0.0  # Left stick Y
        self.axis_rx = 0.0  # Right stick X
        self.axis_ry = 0.0  # Right stick Y
        self.lock = threading.Lock()
        
        # Button state for edge detection
        self.prev_buttons = {}
        
        # Pending key for next publish cycle (0 = no key)
        self.pending_key = 0

        # Track button states for display
        self.btn_states = {
            'A': 0, 'B': 0, 'X': 0, 'Y': 0,
            'LB': 0, 'RB': 0,
            'DU': 0, 'DD': 0, 'DL': 0, 'DR': 0
        }

        # --- Ethernet link-loss safety watchdog ---------------------------
        # If the monitored ethernet interface drops, the robot must not keep
        # running on the last commanded velocity ("going rogue"). Instead we
        # smoothly decay u_des to 0, then tell semantic_poisson to stop and
        # shut down. ROS/DDS traffic between these co-located nodes goes over
        # loopback, so it is unaffected by the external NIC dropping.
        self.eth_interface = 'eth0'      # SSH Ethernet interface on the Jetson
        self.link_poll_hz = 5.0          # link state poll rate
        self.link_down_debounce = 2      # 2 polls at 5 Hz = about 0.4 s
        self.zero_publish_cycles = 50   # publish zero for 0.5 s before termination
        self.semantic_process_pattern = 'semantic_poisson'

        # Safety state machine: 'active' -> 'zeroing' -> 'terminating' -> 'done'
        self.safety_state = 'active'
        self.link_lost = False
        self._link_seen_up = False       # only arm after link has been up once
        self._zero_publish_count = 0
        self._semantic_kill_sent = False
        # Snapshot of the single most recent command at disconnect time.
        self.disconnect_command = None
        # Last published velocity command. No command history is retained.
        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_vyaw = 0.0

        # Timer for publishing at constant rate (matches C++ 100Hz)
        self.timer = self.create_timer(1.0 / publish_rate, self.publish_callback)

        # Start controller input thread
        self.running = True
        self.input_thread = threading.Thread(target=self.read_controller, daemon=True)
        self.input_thread.start()

        # Start ethernet link-loss watchdog thread
        self.eth_monitor_thread = threading.Thread(target=self.monitor_ethernet, daemon=True)
        self.eth_monitor_thread.start()

        self.get_logger().info(
            f'Xbox controller teleop started. Monitoring ethernet link on '
            f"'{self.eth_interface}'. Waiting for controller...")
    
    def normalize_axis(self, value: int, max_val: int = 32768) -> float:
        """Normalize axis value to -1 to 1 range with deadzone."""
        normalized = value / max_val
        if abs(normalized) < self.deadzone:
            return 0.0
        return normalized
    
    def read_controller(self):
        """Background thread to read controller events."""
        while self.running:
            try:
                events = inputs.get_gamepad()
                for event in events:
                    self.handle_event(event)
            except inputs.UnpluggedError:
                self.get_logger().warn('Controller disconnected. Waiting...')
                import time
                time.sleep(1.0)
            except Exception as e:
                self.get_logger().error(f'Controller error: {e}')
                import time
                time.sleep(0.1)
    
    def handle_event(self, event):
        """Handle a single controller event."""
        code = event.code
        state = event.state
        
        with self.lock:
            # Axis events (sticks)
            if code == 'ABS_X':  # Left stick X (negated for correct direction)
                self.axis_lx = -self.normalize_axis(state)
            elif code == 'ABS_Y':  # Left stick Y (inverted)
                self.axis_ly = -self.normalize_axis(state)
            elif code == 'ABS_RX':  # Right stick X (negated)
                self.axis_rx = -self.normalize_axis(state)
            elif code == 'ABS_RY':  # Right stick Y
                self.axis_ry = -self.normalize_axis(state)
            
            # Button events (only trigger on press, not release)
            # Queue key for next publish cycle instead of publishing directly
            
            # Update button states
            if code == 'BTN_SOUTH': self.btn_states['A'] = state
            elif code == 'BTN_EAST': self.btn_states['B'] = state
            elif code == 'BTN_NORTH': self.btn_states['X'] = state  # Physical X = BTN_NORTH
            elif code == 'BTN_WEST': self.btn_states['Y'] = state   # Physical Y = BTN_WEST
            elif code == 'BTN_TL': self.btn_states['LB'] = state
            elif code == 'BTN_TR': self.btn_states['RB'] = state
            
            # Logic for commands (press only)
            # DEBUG: Show button codes when pressed
            if 'BTN' in code and state == 1:
                self.get_logger().info(f'Button pressed: {code}')
            if code == 'BTN_SOUTH' and state == 1:  # A button -> Space
                self.pending_key = ord(' ')
                # self.get_logger().info('A pressed -> Space')
            elif code == 'BTN_EAST' and state == 1:  # B button -> 'r'
                self.pending_key = ord('r')
                # self.get_logger().info('B pressed -> r (realtime SF toggle)')
            elif code == 'BTN_NORTH' and state == 1:  # Physical X button -> 'p'  
                self.pending_key = ord('p')
                # self.get_logger().info('X pressed -> p (predictive SF toggle)')
            # elif code == 'BTN_WEST' and state == 1:  # Physical Y button -> 'd'
            #     self.pending_key = ord('d')
            #     self.get_logger().info('Y pressed -> d (deal parameter)')
            elif code == 'BTN_START' and state == 1:  # Start button -> quit
                # self.get_logger().info('Start pressed -> Shutting down')
                self.running = False
                rclpy.shutdown()
            
            # # D-Pad for wn parameter (1-6)
            elif code == 'ABS_HAT0X':  # D-pad left/right
                if state == -1: self.btn_states['DL'] = 1; self.btn_states['DR'] = 0
                elif state == 1: self.btn_states['DL'] = 0; self.btn_states['DR'] = 1
                else: self.btn_states['DL'] = 0; self.btn_states['DR'] = 0

                # if state == -1:  # Left
                #     self.pending_key = ord('1')
                #     self.get_logger().info('D-Pad Left -> 1')
                # elif state == 1:  # Right
                #     self.pending_key = ord('2')
                #     self.get_logger().info('D-Pad Right -> 2')
            elif code == 'ABS_HAT0Y':  # D-pad up/down
                if state == -1: self.btn_states['DU'] = 1; self.btn_states['DD'] = 0
                elif state == 1: self.btn_states['DU'] = 0; self.btn_states['DD'] = 1
                else: self.btn_states['DU'] = 0; self.btn_states['DD'] = 0

                # if state == -1:  # Up
                #     self.pending_key = ord('3')
                #     self.get_logger().info('D-Pad Up -> 3')
                # elif state == 1:  # Down
                #     self.pending_key = ord('4')
                #     self.get_logger().info('D-Pad Down -> 4')
            
            # # Bumpers for 5-6
            # elif code == 'BTN_TL' and state == 1:  # Left bumper
            #     self.pending_key = ord('5')
            #     self.get_logger().info('LB pressed -> 5')
            # elif code == 'BTN_TR' and state == 1:  # Right bumper
            #     self.pending_key = ord('6')
            #     self.get_logger().info('RB pressed -> 6')
    
    def publish_key(self, key_code: int):
        """Publish a key press."""
        msg = Int32()
        msg.data = key_code
        self.key_pub.publish(msg)

    def _eth_link_up(self) -> bool:
        """Return True when the monitored Ethernet interface has carrier."""
        carrier_path = f'/sys/class/net/{self.eth_interface}/carrier'
        try:
            with open(carrier_path, 'r', encoding='utf-8') as file:
                return file.read().strip() == '1'
        except OSError as exc:
            self.get_logger().error(
                f"Cannot read Ethernet carrier from {carrier_path}: {exc}"
            )
            return False

    def monitor_ethernet(self):
        """Trip the watchdog after the previously-up Ethernet link drops."""
        period = 1.0 / self.link_poll_hz
        down_count = 0
        previous_link_up = None

        self.get_logger().warn(
            f"Ethernet watchdog thread started on '{self.eth_interface}'"
        )

        while self.running:
            link_up = self._eth_link_up()

            if link_up != previous_link_up:
                self.get_logger().warn(
                    f"Ethernet carrier changed: interface={self.eth_interface}, "
                    f"carrier={'UP' if link_up else 'DOWN'}, "
                    f"armed={self._link_seen_up}"
                )
                previous_link_up = link_up

            if link_up:
                if not self._link_seen_up:
                    self.get_logger().warn(
                        f"Ethernet watchdog ARMED on '{self.eth_interface}'"
                    )
                self._link_seen_up = True
                down_count = 0
            else:
                down_count += 1
                if (
                    self._link_seen_up
                    and not self.link_lost
                    and down_count >= self.link_down_debounce
                ):
                    with self.lock:
                        self.link_lost = True
                    self.get_logger().error(
                        f"Ethernet link '{self.eth_interface}' lost after "
                        f"{down_count} consecutive checks. Safety shutdown triggered."
                    )

            time.sleep(period)

    def _find_semantic_pids(self):
        """Return semantic_poisson process IDs, excluding this teleop process."""
        try:
            result = subprocess.run(
                ['pgrep', '-f', self.semantic_process_pattern],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 1:
                return []
            if result.returncode != 0:
                self.get_logger().error(
                    f'pgrep failed with code {result.returncode}: '
                    f'{result.stderr.strip()}'
                )
                return []

            own_pid = os.getpid()
            return [
                int(value) for value in result.stdout.split()
                if value.isdigit() and int(value) != own_pid
            ]
        except Exception as exc:
            self.get_logger().error(f'Failed to locate semantic_poisson: {exc}')
            return []

    @staticmethod
    def _pid_alive(pid):
        """Return True while a process exists, including during signal handling."""
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _signal_and_wait(self, pids, sig, timeout_sec):
        """Signal all supplied PIDs and wait up to timeout_sec for exit."""
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
            except PermissionError as exc:
                self.get_logger().error(
                    f'Permission denied signaling semantic_poisson PID {pid}: {exc}'
                )

        deadline = time.monotonic() + timeout_sec
        remaining = list(pids)
        while remaining and time.monotonic() < deadline:
            time.sleep(0.05)
            remaining = [pid for pid in remaining if self._pid_alive(pid)]
        return remaining

    def _terminate_semantic_poisson(self):
        """Synchronously stop semantic_poisson before teleop exits.

        SIGINT permits ROS cleanup, SIGTERM handles a stuck executor, and
        SIGKILL is the final safety fallback. The function verifies process
        exit after every stage rather than launching a daemon fallback thread.
        """
        if self._semantic_kill_sent:
            return
        self._semantic_kill_sent = True

        pids = self._find_semantic_pids()
        if not pids:
            self.get_logger().error(
                f'No process matched pattern {self.semantic_process_pattern!r}. '
                'Check the executable command with: pgrep -af semantic_poisson'
            )
            return

        self.get_logger().error(f'Terminating semantic_poisson PID(s): {pids}')

        remaining = self._signal_and_wait(pids, signal.SIGINT, 0.75)
        if remaining:
            self.get_logger().warn(
                f'PID(s) {remaining} ignored SIGINT; sending SIGTERM'
            )
            remaining = self._signal_and_wait(remaining, signal.SIGTERM, 0.50)

        if remaining:
            self.get_logger().error(
                f'PID(s) {remaining} ignored SIGTERM; sending SIGKILL'
            )
            remaining = self._signal_and_wait(remaining, signal.SIGKILL, 0.25)

        if remaining:
            self.get_logger().error(
                f'Unable to terminate semantic_poisson PID(s): {remaining}'
            )
        else:
            self.get_logger().error('semantic_poisson process terminated.')

    def _safety_publish(self):
        """Latch zero velocity after Ethernet loss and terminate both nodes.

        The command present at disconnect is captured exactly once in
        ``disconnect_command``. No decayed/intermediate teleop commands are
        generated or replayed.
        """
        zero_twist = Twist()

        if self.safety_state == 'active':
            self.disconnect_command = (
                self.last_vx,
                self.last_vy,
                self.last_vyaw,
            )
            self.pending_key = 0
            self.axis_lx = 0.0
            self.axis_ly = 0.0
            self.axis_rx = 0.0
            self.axis_ry = 0.0
            self.safety_state = 'zeroing'
            self.get_logger().error(
                'Ethernet lost. Latched latest command '
                f'{self.disconnect_command}; commanding zero immediately.'
            )

        if self.safety_state == 'zeroing':
            self.last_vx = 0.0
            self.last_vy = 0.0
            self.last_vyaw = 0.0
            self.twist_pub.publish(zero_twist)
            self._zero_publish_count += 1

            if self._zero_publish_count >= self.zero_publish_cycles:
                self.safety_state = 'terminating'
            return

        if self.safety_state == 'terminating':
            self.twist_pub.publish(zero_twist)
            self._terminate_semantic_poisson()
            self.safety_state = 'done'
            self.running = False
            self.get_logger().error(
                'semantic_poisson termination requested; shutting down teleop.'
            )
            if rclpy.ok():
                rclpy.shutdown()
            return

        # Keep the command latched at zero if shutdown takes another callback.
        self.twist_pub.publish(zero_twist)

    def publish_callback(self):
        """Timer callback to publish velocity commands and key_press (for timing)."""
        with self.lock:
            # Link-loss safety override: ignore controller input, latch zero,
            # terminate semantic_poisson, and shut down this teleop node.
            if self.link_lost:
                self._safety_publish()
                return

            twist = Twist()

            # Left stick Y -> forward/backward
            if self.axis_ly > 0:
                twist.linear.x = self.axis_ly * self.vel_max_x_fwd
            else:
                twist.linear.x = self.axis_ly * self.vel_max_x_bwd

            # Left stick X -> strafe
            twist.linear.y = self.axis_lx * self.vel_max_y

            # Right stick X -> rotation
            twist.angular.z = self.axis_rx * self.vel_max_yaw

            # Remember only the latest command for the disconnect snapshot
            self.last_vx = twist.linear.x
            self.last_vy = twist.linear.y
            self.last_vyaw = twist.angular.z

            self.twist_pub.publish(twist)

            # Throttle print to 10Hz (every 10th callback) to avoid I/O blocking lag
            if not hasattr(self, '_print_counter'):
                self._print_counter = 0
            self._print_counter += 1
            if self._print_counter % 10 == 0:
                # Format button string with descriptions
                desc_map = {
                    'A': 'Stp', 'B': 'RT', 'X': 'Prd', 'Y': 'Dl',
                    'LB': 'W', 'RB': 'W',
                    'DU': 'W', 'DD': 'W', 'DL': 'W', 'DR': 'W'
                }
                btn_str = " ".join([f"{k}({desc_map.get(k, '')}):{v}" for k, v in self.btn_states.items()])
                print(f"\rv:{twist.linear.x:.2f},{twist.linear.y:.2f} w:{twist.angular.z:.2f}|{btn_str}   ", end="", flush=True)

            # Publish pending key or ERR (-1) to maintain constant callback rate
            # This matches C++ teleop behavior for data logging timing
            key_msg = Int32()
            if self.pending_key != 0:
                key_msg.data = self.pending_key
                self.pending_key = 0  # Clear after publishing
            else:
                key_msg.data = -1  # ERR from ncurses
            self.key_pub.publish(key_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TeleopControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()