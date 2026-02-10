import torch
import random
from pathlib import Path
from objectnav.constants import *
from objectnav.sim.agent import init_agent
from objectnav.sim.simulator import make_sim
from objectnav.sim.navmesh import compute_navmesh
from objectnav.sim.maps import build_grid_map_from_navmesh
from objectnav.utils.spatial.rotations import rotation_to_yaw
from objectnav.utils.visualization.maps import save_map_with_agent
from objectnav.utils.visualization.observations import save_rgbd_observations
from objectnav.utils.visualization.detections import save_yolo_detections_plot
from objectnav.perception.config import YoloConfig
from objectnav.perception.pipeline import build_yolo_detector, run_yolo_detections


def main():
    # Pick a fixed seed (or read from CLI/env)
    # seed = 1234 # deterministic seed for testing
    seed = random.randint(0, 2**32 - 1) # random seed for non-deterministic runs

    # Load YOLO11x model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    yolo_config = YoloConfig(
        weights_path=Path("datasets/models/yolo11x.pt"),
        device=device,
        use_softmax_patch=True
    )
    yolo_detector = build_yolo_detector(yolo_config)

    # Launch simulator
    simulator = make_sim(scene_dataset_config=Path("datasets/ai2thor-hab/ai2thor-hab/ai2thor-hab.scene_dataset_config.json"), scene_id="FloorPlan2_physics")
        
    # Setting a seed for reproducibility    
    random.seed(seed) # This will make random sampling reproducible (eg. yaw degree)
    simulator.sim.seed(seed)
    simulator.sim.pathfinder.seed(seed)

    # Get navigation mesh
    if compute_navmesh(simulator.sim):
        grid_map = build_grid_map_from_navmesh(simulator.sim)
    else:
        raise RuntimeError("Failed to compute navigation mesh.")

    # Initialize the (only) agent: configuration was made in simulator creation
    agent = init_agent(simulator.sim)
    
    # Select action space
    action_names = list(simulator.cfg.agents[0].action_space.keys())

    # Random short rollout to test
    T = 10
    for i in range(T):

        print(f"\nStep {i+1} / {T}")
        action = random.choice(action_names)
        observations = simulator.sim.step(action)
        rgb, depth, collided = observations["color_sensor"], observations["depth_sensor"], observations["collided"]
        agent_state = agent.get_state()

        print("Action:", action)
        print("Collided:", collided)
        print("Agent_state: position", agent_state.position, "yaw", rotation_to_yaw(agent_state.rotation)) 
        save_rgbd_observations(rgb, depth, save_path=f"outputs/observations_step_{i+1}.png")
        save_map_with_agent(
            grid_map,
            agent_state.position,
            agent_state.rotation,
            save_path=f"outputs/grid_map_with_agent_step_{i+1}.png",
            sim=simulator.sim,
            title="Grid Map + Agent",
            agent_radius_px=20,
        )
        
        # Run YOLO. Our images are in RGB format, but YOLO expects BGR for numpy input
        detections, yolo_result = run_yolo_detections(yolo_detector,rgb,input_color="rgb")

        save_yolo_detections_plot(
            yolo_result,
            save_path=f"outputs/detections_step_{i+1}.png",
            show_conf=True,
            show_labels=True,
            show_boxes=True,
        )

        print("Detections:")
        for det in detections:
            xyxy_fmt = "(" + ", ".join(f"{v:.2f}" for v in det.xyxy) + ")"
            probs_fmt = (
                "(" + ", ".join(f"{v:.3f}" for v in det.probs) + ")"
                if det.probs is not None
                else "N/A"
            )
            print(
                f"Class: {det.cls_name}, Confidence: {det.conf:.2f}, Box: {xyxy_fmt}, Scale: {det.scale:.2f},"
                f"\n Probs: {probs_fmt}"
            )


    # Close simulator
    print("Closing simulator...")
    simulator.close()


if __name__ == "__main__":
    main()