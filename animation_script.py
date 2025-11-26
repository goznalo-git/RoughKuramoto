import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# --- CONFIGURATION ---
# Choose: 'ballistic', 'eden', 'relaxation' or 'random'
SIMULATION_MODE = 'relaxation'

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
        # Unified grid and height map
        self.grid = np.zeros((height, width), dtype=int)
        self.heights = np.zeros(width, dtype=int)
        self.cmap = 'magma' 

    def step(self):
        # 1. Pick a random column
        i = random.randint(0, self.width - 1)
        
        # 2. Identify neighbors (Periodic boundaries)
        l_idx = (i - 1) % self.width
        r_idx = (i + 1) % self.width
        
        l_h = self.heights[l_idx]
        r_h = self.heights[r_idx]
        c_h = self.heights[i] # Current height
        
        target_idx = i
        new_h = 0

        # --- LOGIC SELECTION ---

        if self.mode == 'random':
            # Pure pile-up
            target_idx = i
            new_h = c_h + 1

        elif self.mode == 'ballistic':
            # Sticky Sides (KPZ)
            target_idx = i
            new_h = max(c_h + 1, l_h, r_h)

        elif self.mode == 'eden':
            # Sticky Corners (Fastest Growth)
            target_idx = i
            new_h = max(c_h + 1, l_h + 1, r_h + 1)

        elif self.mode == 'relaxation':
            # Family Model (Surface Diffusion / Edwards-Wilkinson)
            # The particle lands at i, but slides to the lowest neighbor.
            
            min_h = min(c_h, l_h, r_h)
            
            # Logic: If a neighbor is strictly lower, move there.
            # (Prioritize neighbors over current column to maximize smoothing)
            if l_h == min_h:
                target_idx = l_idx
            elif r_h == min_h:
                target_idx = r_idx
            else:
                target_idx = i
            
            # In relaxation, we just pile up on the chosen spot (no side sticking)
            new_h = self.heights[target_idx] + 1
        
        # 3. Update the Grid
        if new_h < self.height:
            self.heights[target_idx] = new_h
            self.grid[self.height - new_h, target_idx] = 1

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