import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from humanoid_env import HumanoidLifterEnv
from utils import load_wandb_config, resolve_device

# Keeping your original stable hyperparameters [cite: 22, 23]
TRAINING_CONFIG = {
    "algorithm": "SAC",
    "policy": "MlpPolicy",
    "total_timesteps": 100_000,
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
    # KEEP: Your utility device resolution [cite: 32]
    wandb_config = load_wandb_config()
    requested_device = wandb_config["device"]
    resolved_device = resolve_device(requested_device)

    print("Logging into Weights & Biases...")
    wandb.login()

    # Initializing integrated W&B run
    run = wandb.init(
        entity=wandb_config.get("entity"),
        project=wandb_config.get("project"),
        config={
            **TRAINING_CONFIG,
            "resolved_device": resolved_device,
            "contact_aware_pipeline": True,
        },
        sync_tensorboard=True,
        save_code=True,
    )

    print(f"Using device: {resolved_device}")
    env = Monitor(HumanoidLifterEnv())
    
    try:
        print("Initializing Contact-Aware SAC agent...")
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
            tensorboard_log=f"runs/{run.id}",
        )

        print("Starting training...")
        # NEW: Using WandbCallback for empirical visualization [cite: 25]
        model.learn(
            total_timesteps=TRAINING_CONFIG["total_timesteps"],
            log_interval=1,
            progress_bar=True,
            callback=WandbCallback(
                model_save_path=f"models/{run.id}",
                verbose=2,
            ),
        )

        print("Saving trained model...")
        model.save("sac_humanoid_lifter_integrated")
        
    finally:
        run.finish()
        env.close()

    print("Pipeline Complete.")

if __name__ == "__main__":
    main()