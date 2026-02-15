import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

print("Loading trajectory data...")
# Load the CSV data exported by our C++ simulation
df = pd.read_csv("trajectory.csv")

# Get unique steps to know how many frames we have
steps = df['Step'].unique()

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_facecolor('black') # Space is black!
fig.patch.set_facecolor('black')

# Set the axis limits based on the solar system size
ax.set_xlim(-500, 500)
ax.set_ylim(-500, 500)
ax.set_title("N-Body Solar System Simulation", color='white')
ax.axis('off') # Hide the grid lines for a cleaner look

# Initialize the scatter plot
scatter = ax.scatter([], [], s=[], c=[])

def update(frame):
    # Filter data for the current time step
    current_data = df[df['Step'] == frame]
    
    x = current_data['X'].values
    y = current_data['Y'].values
    ids = current_data['BodyID'].values
    
    # Set colors: Sun (ID 0) is Yellow, Comet (ID 1) is Red, Planets are Blue
    colors = ['yellow' if i == 0 else 'red' if i == 1 else 'cyan' for i in ids]
    
    # Set sizes: Sun is big, others are small
    sizes = [100 if i == 0 else 10 for i in ids]
    
    # Update the scatter plot data
    scatter.set_offsets(list(zip(x, y)))
    scatter.set_color(colors)
    scatter.set_sizes(sizes)
    
    return scatter,

print("Generating animation...")
# Create the animation object
ani = animation.FuncAnimation(fig, update, frames=steps, interval=20, blit=True)

plt.show()