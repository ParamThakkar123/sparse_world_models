from __future__ import annotations

import json

from models.envs import TabletopPushEnv


def main() -> None:
    env = TabletopPushEnv()
    obs = env.reset()
    print("reset observation shapes:")
    print(json.dumps({key: list(value.shape) for key, value in obs.items()}, indent=2))

    for step in range(5):
        obs, reward, done, info = env.step(env.sample_random_action())
        print(
            json.dumps(
                {
                    "step": step,
                    "reward": round(reward, 4),
                    "done": done,
                    "target_distance": round(float(info["target_distance"]), 4),
                }
            )
        )

    state = env.get_state()
    print("state keys:", sorted(state.keys()))
    print("planar pose shape:", state["object_pose"].shape)


if __name__ == "__main__":
    main()
