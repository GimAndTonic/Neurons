import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Set up the initial plot
fig, ax = plt.subplots()
x = np.linspace(0, 2 * np.pi, 100)
line, = ax.plot(x, np.sin(x))

def update_plot(frame):
    """Update the plot with a new phase."""
    phase = 2 * np.pi * frame / 20
    line.set_ydata(np.sin(x + phase))

# Create an animation
ani = FuncAnimation(fig, update_plot, frames=20, interval=500, repeat=False)

plt.show()
