# %%
import matplotlib.pyplot as plt


def calculate_cyclical_lr(iteration, total_steps, num_cycles, hold_fraction=0.5):
    step_size, hold_steps = calculate_step_size_and_hold_steps(
        total_steps, num_cycles, hold_fraction
    )

    cycle = iteration // (step_size + hold_steps)

    cycle_pos = iteration - (cycle * (step_size + hold_steps))

    if cycle_pos < step_size:
        return cycle_pos / step_size
    elif cycle_pos < step_size + hold_steps:
        return 1.0
    else:
        return 0.0


def calculate_step_size_and_hold_steps(total_steps, num_cycles, hold_fraction=0.5):
    hold_fraction = min(max(hold_fraction, 0), 1)
    steps_per_cycle = total_steps / num_cycles

    hold_steps = int(steps_per_cycle * hold_fraction)
    step_size = steps_per_cycle - hold_steps

    return step_size, hold_steps


# Example usage and plot
total_training_steps = 40000
num_cycles = 1
hold_fraction = 0.00

# Calculate learning rates for each iteration
learning_rates = [
    calculate_cyclical_lr(i, total_training_steps, num_cycles, hold_fraction)
    for i in range(total_training_steps)
]

# Plotting the learning rates
plt.figure(figsize=(10, 5))
plt.plot(learning_rates, label=r"$\beta_{KL}$")
plt.xlabel("Training Iteration")
plt.ylabel("Learning Rate")
plt.legend()
plt.grid(True)
plt.show()

# %%
