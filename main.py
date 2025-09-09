"""
Main entry point for the RC car simulation.
Run this file to start the simulation with manual control.
"""

import pygame
import sys
import argparse
from simulation.environment import RCCarEnvironment
from control.manual_control import ManualController

def build_env(map_index: int) -> RCCarEnvironment:
    start_map = f"maps/map_start{map_index}.png"
    display_map = f"maps/map{map_index}.png"
    return RCCarEnvironment(
        map_path=start_map,
        display_map_path=display_map,
        show_collision_box=True,  # Toggle collision box visibility
        show_sensors=True,        # Toggle sensor ray visibility
        show_racing_line=True     # Toggle racing line visibility
    )

def parse_args():
    parser = argparse.ArgumentParser(description="RC Car Simulation")
    parser.add_argument("--map", "-m", type=int, default=1, help="Map index to load (e.g., 1 for map1)")
    return parser.parse_args()

def main():
    # Initialize pygame
    pygame.init()

    args = parse_args()
    current_map = 5

    # Create the environment for the selected map
    env = build_env(current_map)
    
    # Create manual controller
    controller = ManualController()
    
    # Main game loop
    clock = pygame.time.Clock()
    running = True
    
    print("Manual Control:")
    print("W/S - Forward/Backward")
    print("A/D - Turn Left/Right")
    print("Arrow Keys - Direct wheel control")
    print("R - Reset car position")
    print("1-9 - Switch map set (mapN/map_startN)")
    print("ESC - Exit")
    
    while running:
        dt = clock.tick(60) / 1000.0  # 60 FPS, convert to seconds
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset()
                elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9]:
                    # Switch map on number key
                    key_to_num = {
                        pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3,
                        pygame.K_4: 4, pygame.K_5: 5, pygame.K_6: 6,
                        pygame.K_7: 7, pygame.K_8: 8, pygame.K_9: 9,
                    }
                    new_map = key_to_num.get(event.key, current_map)
                    if new_map != current_map:
                        current_map = new_map
                        env = build_env(current_map)
                        print(f"Switched to map {current_map}")
        
        # Get control input
        action = controller.get_action()
        
        # Step the environment
        observation, reward, done, info = env.step(action)
        
        # Reset if episode is done
        if done:
            env.reset()
        
        # Render the environment
        env.render()
        
        # Update display
        pygame.display.flip()
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
