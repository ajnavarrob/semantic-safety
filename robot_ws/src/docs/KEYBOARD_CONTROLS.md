# Keyboard Controls Reference

This document describes all keyboard inputs for the teleoperation and semantic safety system.

---

## Teleoperation Controls (teleop.cpp)

These keys control robot motion via the `teleop_node`:

| Key | Action | Value |
|-----|--------|-------|
| `↑` (Up Arrow) | Forward velocity | +0.50 m/s |
| `↓` (Down Arrow) | Backward velocity | -0.50 m/s |
| `←` (Left Arrow) | Strafe left | +0.50 m/s |
| `→` (Right Arrow) | Strafe right | -0.50 m/s |
| `,` | Rotate left (counter-clockwise) | +1.0 rad/s |
| `.` | Rotate right (clockwise) | -1.0 rad/s |
| `q` | Quit teleop node | — |

> [!NOTE]
> Velocity commands auto-reset to zero after 20 idle cycles (~200ms) when no keys are pressed.

---

## Safety System Controls (semantic_poisson.cpp)

These keys control the Poisson safety filter and experiment flow:

### Experiment State Machine

| Key | Action | Effect |
|-----|--------|--------|
| `Space` (×1) | Enable data saving | Starts logging to CSV |
| `Space` (×3) | Start robot | Enables motion commands |
| `Space` (×6) | Stop robot | Stops motion and shuts down |

### Safety Filter Togglesr1

| Key | Action | Description |
|-----|--------|-------------|
| `r` | Toggle realtime safety filter | Enables/disables analytical CBF-based safety filter |
| `p` | Toggle predictive safety filter | Enables/disables MPC-based predictive safety |

### CBF Gain Selection (ωₙ)

| Key | Gain Value (ωₙ) |
|-----|-----------------|
| `1` | 0.5 |
| `2` | 1.0 |
| `3` | 1.5 |
| `4` | 2.0 |
| `5` | 4.0 |
| `6` | 8.0 |

> [!TIP]
> Lower ωₙ values result in more conservative (safer) behavior, while higher values allow closer approaches to obstacles.

### Experiment Parameter Deck

| Key | Action |
|-----|--------|
| `d` | Deal next randomized parameter set |

The `d` key cycles through a shuffled deck of parameter configurations (`0`-`6`), each setting different combinations of safety filter modes and ωₙ values for experiments.

---

## Architecture

```
┌─────────────────┐     key_press      ┌─────────────────────┐
│   teleop_node   │ ──────────────────▶│ poisson_control     │
│  (teleop.cpp)   │                    │ (semantic_poisson)  │
│                 │      u_des         │                     │
│                 │ ──────────────────▶│                     │
└─────────────────┘   Twist command    └─────────────────────┘
```

- **key_press**: `std_msgs/Int32` — raw key code for flags/toggles
- **u_des**: `geometry_msgs/Twist` — desired velocity command
