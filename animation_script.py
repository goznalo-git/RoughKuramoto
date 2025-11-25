import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# --- CONFIGURATION ---
# Choose: 'ballistic', 'eden', or 'random'
SIMULATION_MODE = 'eden' 

# Saving Settings
SAVE_ANIMATION = True          # Set to True to save a file, False to just show it
OUTPUT_FILENAME = "Gifs/" + SIMULATION_MODE + '_deposition.gif' # .gif is safest; use .mp4 if you have ffmpeg
SAVE_FRAMES = 600              # Total number of frames to record
FPS = 30                       # Frames per second

# Grid settings
WIDTH = 200        
HEIGHT = 300       
INTERVAL = 1                   # Display speed (ms) - irrelevant if saving

class GrowthSimulator:
    def __init__(self, mode='ballistic', width=100, height=200):
        self.mode = mode
        self.width = width
        self.height = height
        # Unified grid and height map for all deposition models
        self.grid = np.zeros((height, width), dtype=int)
        self.heights = np.zeros(width, dtype=int)
        self.cmap = 'magma' 

    def step(self):
        # 1. Pick a random column to drop a block
        i = random.randint(0, self.width - 1)
        
        # 2. Identify neighbor heights (Periodic/Wrap-around boundaries)
        left_h = self.heights[(i - 1) % self.width]
        right_h = self.heights[(i + 1) % self.width]
        current_h = self.heights[i]
        
        # 3. Determine New Height based on the rules
        if self.mode == 'random':
            # Rule: Independent Columns (Tetris with no friction)
            # The block just lands on top of the current column.
            new_h = current_h + 1

        elif self.mode == 'ballistic':
            # Rule: Sticky Sides (Standard Ballistic Deposition)
            # The block sticks if it touches the TOP of the current column
            # OR the SIDE of a neighbor.
            # If neighbor is height H, we stick at H.
            new_h = max(current_h + 1, left_h, right_h)

        elif self.mode == 'eden':
            # Rule: Sticky Corners (Your requested variant)
            # The block sticks if it touches the TOP of the current column
            # OR the CORNER of a neighbor.
            # If neighbor is height H, sticking to its corner puts us at H + 1.
            new_h = max(current_h + 1, left_h + 1, right_h + 1)
        
        # 4. Update the grid if we are within bounds
        if new_h < self.height:
            self.heights[i] = new_h
            # Draw the block (inverted y-axis for matrix indexing)
            self.grid[self.height - new_h, i] = 1

    def get_data(self):
        return self.grid

# --- VISUALIZATION SETUP ---
sim = GrowthSimulator(mode=SIMULATION_MODE, width=WIDTH, height=HEIGHT)
fig, ax = plt.subplots(figsize=(8, 10))
ax.set_title(f"{SIMULATION_MODE.capitalize()} Model Simulation")
ax.axis('off')

# Initialize image
im = ax.imshow(sim.get_data(), cmap=sim.cmap, origin='upper', vmin=0, vmax=1)

def update(frame):
    # Perform multiple simulation steps per animation frame for speed
    steps_per_frame = 50
    for _ in range(steps_per_frame):
        sim.step()
    
    im.set_data(sim.get_data())
    
    # Optional: Print progress if saving
    if SAVE_ANIMATION and frame % 10 == 0:
        print(f"Rendering frame {frame}/{SAVE_FRAMES}...", end='\r')
        
    return [im]

# Create Animation
ani = animation.FuncAnimation(
    fig, update, interval=INTERVAL, blit=False, cache_frame_data=False,
    frames=SAVE_FRAMES if SAVE_ANIMATION else None
)

if SAVE_ANIMATION:
    print(f"Starting save to {OUTPUT_FILENAME}...")
    if OUTPUT_FILENAME.endswith('.mp4'):
        # Requires ffmpeg installed on system
        ani.save(OUTPUT_FILENAME, writer='ffmpeg', fps=FPS)
    else:
        # Works with default matplotlib install (uses Pillow)
        ani.save(OUTPUT_FILENAME, writer='pillow', fps=FPS)
    print(f"\nSaved {OUTPUT_FILENAME}!")
else:
    plt.show()