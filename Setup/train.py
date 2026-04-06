import wandb
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from humanoid_env import HumanoidLifterEnv
from utils import load_wandb_config, resolve_device


TRAINING_CONFIG = {
    "algorithm": "SAC",
    "policy": "MlpPolicy",
    "total_timesteps": 50_000,
    "learning_rate": 3e-4,
    "batch_size": 256,
    "buffer_size": 1_000_000,
    "learning_starts": 1_000,
    "gamma": 0.99,
    "tau": 0.005,
    "train_freq": 1,
    "gradient_steps": 1,
}


def main():
    wandb_config = load_wandb_config()
    requested_device = wandb_config["device"]
    resolved_device = resolve_device(requested_device)

    print("Logging into Weights & Biases...")
    wandb.login()

    run = wandb.init(
        entity=wandb_config["entity"],
        project=wandb_config["project"],
        config={
            **TRAINING_CONFIG,
            "requested_device": requested_device,
            "resolved_device": resolved_device,
        },
        sync_tensorboard=True,
        save_code=True,
    )

    print(f"Using device: {resolved_device} (requested: {requested_device})")
    print("Initializing environment...")
    env = Monitor(HumanoidLifterEnv())
    try:
        print("Initializing SAC agent...")
        model = SAC(
            TRAINING_CONFIG["policy"],
            env,
            learning_rate=TRAINING_CONFIG["learning_rate"],
            batch_size=TRAINING_CONFIG["batch_size"],
            buffer_size=TRAINING_CONFIG["buffer_size"],
            learning_starts=TRAINING_CONFIG["learning_starts"],
            gamma=TRAINING_CONFIG["gamma"],
            tau=TRAINING_CONFIG["tau"],
            train_freq=TRAINING_CONFIG["train_freq"],
            gradient_steps=TRAINING_CONFIG["gradient_steps"],
            device=resolved_device,
            verbose=1,
            tensorboard_log="runs",
        )

        print("Starting training...")
        model.learn(
            total_timesteps=TRAINING_CONFIG["total_timesteps"],
            log_interval=1,
            progress_bar=True,
        )

        print("Saving trained model...")
        model.save("sac_humanoid_lifter")
    finally:
        run.finish()
        env.close()

    print("Done!")


if __name__ == "__main__":
    main()
