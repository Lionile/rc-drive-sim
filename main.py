import pygame
import sys
import argparse
from simulation.environment import Environment
from controllers.manual_controller import ManualController

def build_env(map_index: int) -> Environment:
    start_map = f"maps/map_start{map_index}.png"
    display_map = f"maps/map{map_index}.png"
    return Environment(
        map_path=start_map,
        display_map_path=display_map,
        show_collision_box=True,  # collision box visibility
        show_sensors=True,        # sensor ray visibility
        show_racing_line=True,    # racing line visibility
        show_track_edges=False    # track boundary edges visibility
    )

def parse_args():
    parser = argparse.ArgumentParser(description="RC Car Simulation")
    parser.add_argument("--map", "-m", type=int, default=1, help="Map index to load (e.g., 1 for map1)")
    return parser.parse_args()

def main():
    pygame.init()

    args = parse_args()
    current_map = 5

    # create the environment
    env = build_env(current_map)
    
    # manual controller
    controller = ManualController()
    
    # main game loop
    clock = pygame.time.Clock()
    running = True
    
    while running:
        dt = clock.tick(60) / 1000.0  # 60 FPS, convert to seconds
        fps = clock.get_fps()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset()
                elif event.key == pygame.K_t:
                    env.show_track_edges = not env.show_track_edges
                    print(f"Track edges: {'ON' if env.show_track_edges else 'OFF'}")
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
        
        # control input
        observation = env.get_observation()
        action = controller.act(observation)
        
        # step
        observation, reward, done, info = env.step(action)
        
        # reset if episode is done
        if done:
            env.reset()
        
        env.render()
        # Draw FPS on top of everything
        if hasattr(env, 'screen'):
            font = pygame.font.Font(None, 24)
            fps_text = font.render(f"FPS: {fps:.1f}", True, (255, 255, 0))
            env.screen.blit(fps_text, (10, 10))
        
        pygame.display.flip()
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
