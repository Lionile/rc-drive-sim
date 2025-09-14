"""
RC Car Simulation Runner
Run with different controllers via command line arguments.

Usage:
    python run.py --controller manual --map 5
    python run.py --controller random --map 1 --render
    python run.py --controller ppo --model path/to/model.zip --map 3
    python run.py --controller td3 --model path/to/model.zip --no-render
"""

import pygame
import sys
import argparse
from simulation.environment import Environment
from controllers.manual_controller import ManualController
from controllers.random_controller import RandomController
from controllers.ppo_controller import PPOController
from controllers.td3_controller import TD3Controller
from controllers.pid_controller import PIDController

def create_controller(controller_type, model_path=None, seed=None):
    """
    Factory function to create the appropriate controller.
    
    Args:
        controller_type: Type of controller ('manual', 'random', 'pid', 'ppo', 'td3')
        model_path: Path to trained model (for RL controllers)
        seed: Random seed (for random controller)
        
    Returns:
        Controller instance
    """
    if controller_type == 'manual':
        return ManualController()
    elif controller_type == 'random':
        return RandomController(seed=seed)
    elif controller_type == 'pid':
        return PIDController()
    elif controller_type == 'ppo':
        return PPOController(model_path=model_path)
    elif controller_type == 'td3':
        return TD3Controller(model_path=model_path)
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")

def build_env(map_index: int) -> Environment:
    """Build environment with specified map."""
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

def setup_display(env, render_enabled):
    """Setup or teardown display based on render mode."""
    if render_enabled:
        if not pygame.get_init():
            pygame.init()
        if not hasattr(env, 'screen') or env.screen is None:
            env.screen = pygame.display.set_mode(env.window_size)
            pygame.display.set_caption("RC Car Simulation")
    # Note: We keep pygame initialized even in headless mode to allow toggling

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RC Car Simulation Runner")
    
    # Controller selection
    parser.add_argument("--controller", "-c", type=str, default="manual",
                        choices=["manual", "random", "pid", "ppo", "td3"],
                        help="Controller type to use")
    
    # Model path for RL controllers
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Path to trained model (for PPO/TD3 controllers)")
    
    # Map selection
    parser.add_argument("--map", type=int, default=5,
                        help="Map index to load (e.g., 1 for map1)")
    
    # Rendering options
    parser.add_argument("--render", action="store_true", default=True,
                        help="Enable rendering (default: enabled)")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering for headless mode")
    
    # Random seed
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (for random controller)")
    
    # Physics frequency
    parser.add_argument("--fps", type=int, default=60,
                        help="Physics update frequency (default: 60 Hz)")
    
    return parser.parse_args()

def main():
    """Main simulation loop with configurable controller."""
    args = parse_args()
    
    # Determine rendering mode
    render_enabled = args.render and not args.no_render
    
    # Initialize pygame if rendering
    if render_enabled:
        pygame.init()
    
    # Print configuration
    print(f"RC Car Simulation")
    print(f"Controller: {args.controller}")
    print(f"Map: {args.map}")
    print(f"Rendering: {'ON' if render_enabled else 'OFF'}")
    print(f"Physics FPS: {args.fps}")
    if args.model:
        print(f"Model: {args.model}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print("-" * 40)
    
    # Always initialize pygame (needed for event handling and toggling)
    pygame.init()
    
    # Create environment
    env = build_env(args.map)
    
    # Setup initial display state
    setup_display(env, render_enabled)
    
    # Create controller
    try:
        controller = create_controller(args.controller, args.model, args.seed)
        print(f"✓ {args.controller.capitalize()} controller created")
    except Exception as e:
        print(f"✗ Error creating controller: {e}")
        sys.exit(1)
    
    # Print controls for manual controller
    if args.controller == 'manual':
        print("\nManual Controls:")
        print("W/S - Forward/Backward")
        print("A/D - Turn Left/Right")
        print("Arrow Keys - Direct wheel control")
        print("R - Reset car position")
        print("T - Toggle track edge visibility")
        print("D - Toggle distance heatmap")
        print("H - Toggle rendering (headless/visual)")
        print("1-9 - Switch map set")
        print("ESC - Exit")
        print()
    
    # Main simulation loop
    clock = pygame.time.Clock() if render_enabled else None
    running = True
    current_map = args.map
    
    while running:
        # Calculate dt and fps
        if render_enabled:
            if clock is None:
                clock = pygame.time.Clock()
            dt = clock.tick(args.fps) / 1000.0
            fps = clock.get_fps()
        else:
            dt = 1.0 / args.fps
            fps = args.fps
        
        # Handle pygame events (process events even in headless mode to allow toggling)
        if pygame.get_init():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        env.reset()
                        controller.reset()
                    elif event.key == pygame.K_t:
                        env.show_track_edges = not env.show_track_edges
                        print(f"Track edges: {'ON' if env.show_track_edges else 'OFF'}")
                    elif event.key == pygame.K_d:
                        env.show_distance_heatmap = not env.show_distance_heatmap
                        print(f"Distance heatmap: {'ON' if env.show_distance_heatmap else 'OFF'}")
                    elif event.key == pygame.K_h:
                        render_enabled = not render_enabled
                        print(f"Rendering: {'ON' if render_enabled else 'OFF'}")
                        setup_display(env, render_enabled)
                    elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, 
                                      pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9]:
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
                            setup_display(env, render_enabled)
                            print(f"Switched to map {current_map}")
        
        # Get control input
        observation = env.get_observation()
        action = controller.act(observation, dt)
        
        # Step environment
        observation, reward, terminated, truncated, info = env.step(action, dt)
        
        # Reset if episode is done (either terminated or truncated)
        if terminated or truncated:
            # Print episode end reason for debugging
            if terminated:
                reason = "collision" if info.get('collision', False) else "terminated"
                print(f"Episode ended: {reason} (step {info.get('step', 0)})")
            elif truncated:
                print(f"Episode truncated: time limit reached ({info.get('max_steps', 0)} steps)")
            
            env.reset()
            controller.reset()
        
        # Render if enabled
        if render_enabled:
            env.render()
            # Draw FPS counter
            if hasattr(env, 'screen'):
                font = pygame.font.Font(None, 24)
                fps_text = font.render(f"FPS: {fps:.1f}", True, (255, 255, 0))
                env.screen.blit(fps_text, (10, 10))
            
            pygame.display.flip()
    
    # Cleanup
    if render_enabled:
        pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
