import numpy as np
import time
import random

# Isaac Sim imports
# Note: These require Isaac Sim environment to be active
try:
    from omni.isaac.kit import SimulationApp

    # Initialize the SimulationApp before any other Isaac Sim imports
    simulation_app = SimulationApp({"headless": False})

    from omni.isaac.core import World
    from omni.isaac.core.objects import VisualCuboid, FixedCuboid
    from omni.isaac.sensor import Camera
    import omni.isaac.core.utils.prims as prim_utils
    import omni.kit.viewport.utility as viewport_utils
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.viewports import set_camera_view
except ImportError:
    print("Isaac Sim libraries not found. This script must be run within an Isaac Sim environment.")
    simulation_app = None

def create_random_room(world, room_size=(10, 10), wall_height=2.5):
    """
    Creates a room with a floor and random walls.
    """
    # Create Floor
    world.scene.add(
        FixedCuboid(
            prim_path="/World/Floor",
            name="floor",
            position=np.array([0, 0, -0.05]),
            scale=np.array([room_size[0], room_size[1], 0.1]),
            color=np.array([0.5, 0.5, 0.5])
        )
    )

    # Create Outer Walls
    half_width = room_size[0] / 2
    half_depth = room_size[1] / 2
    
    wall_configs = [
        ([half_width, 0, wall_height/2], [0.1, room_size[1], wall_height], "wall_north"),
        ([-half_width, 0, wall_height/2], [0.1, room_size[1], wall_height], "wall_south"),
        ([0, half_depth, wall_height/2], [room_size[0], 0.1, wall_height], "wall_east"),
        ([0, -half_depth, wall_height/2], [room_size[0], 0.1, wall_height], "wall_west"),
    ]

    for pos, scale, name in wall_configs:
        world.scene.add(
            FixedCuboid(
                prim_path=f"/World/{name}",
                name=name,
                position=np.array(pos),
                scale=np.array(scale),
                color=np.array([0.7, 0.7, 0.8])
            )
        )

    # Add some random internal obstacles (cuboids)
    num_obstacles = random.randint(3, 7)
    for i in range(num_obstacles):
        obs_size = np.array([random.uniform(0.5, 2.0), random.uniform(0.5, 2.0), random.uniform(0.5, 2.0)])
        obs_pos = np.array([
            random.uniform(-half_width + 1, half_width - 1),
            random.uniform(-half_depth + 1, half_depth - 1),
            obs_size[2] / 2
        ])
        world.scene.add(
            FixedCuboid(
                prim_path=f"/World/Obstacle_{i}",
                name=f"obstacle_{i}",
                position=obs_pos,
                scale=obs_size,
                color=np.array([random.random(), random.random(), random.random()])
            )
        )

def main():
    if simulation_app is None:
        return

    world = World(stage_units_in_meters=1.0)
    
    # 1. Generate Room
    create_random_room(world)
    
    # 2. Add Depth Camera
    # Create a camera prim
    camera_path = "/World/Camera"
    camera = Camera(
        prim_path=camera_path,
        name="depth_camera",
        resolution=(640, 480),
        translation=np.array([0.0, 0.0, 1.5]), # 1.5 meters high
        orientation=np.array([1.0, 0.0, 0.0, 0.0]) # Looking forward
    )
    
    world.scene.add(camera)
    
    # Initialize the sensors
    camera.initialize()
    camera.add_motion_vectors_to_frame()
    
    # Reset world to apply changes
    world.reset()

    print("Starting simulation loop...")
    
    try:
        while simulation_app.is_running():
            # Step the simulation
            world.step(render=True)
            
            if world.is_playing():
                # Get images from the camera
                rgba_data = camera.get_rgba()
                depth_data = camera.get_depth()
                
                if rgba_data is not None and depth_data is not None:
                    # rgba_data is (H, W, 4) numpy array
                    # depth_data is (H, W) numpy array (float32, distance in meters)
                    
                    print(f"Captured Frame - RGB Shape: {rgba_data.shape}, Depth Shape: {depth_data.shape}")
                    print(f"Mean Depth: {np.mean(depth_data):.2f} meters")
                    
                    # Example: Accessing a specific pixel
                    center_depth = depth_data[240, 320]
                    print(f"Depth at center: {center_depth:.2f} meters")
                    
                # To move the camera around (random walk)
                # new_pos = camera.get_world_pose()[0] + np.array([0.01, 0, 0])
                # camera.set_world_pose(position=new_pos)

    except KeyboardInterrupt:
        print("Simulation stopped by user.")
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
