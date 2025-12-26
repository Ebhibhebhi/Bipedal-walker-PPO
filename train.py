import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor

# --- 1. EXPERIMENT CONFIGURATIONS ---
# Added 'gae_lambda' to the dictionaries. 
# Default for PPO in SB3 is 0.95.
experiments = [
    {"name": "normalized_baseline", "ent_coef": 0.01, "norm_obs": True, "gamma": 0.99, "clip_range": 1.0, "gae_lambda": 0.95},
    {"name": "high_exploration", "ent_coef": 0.05, "norm_obs": True, "gamma": 0.99, "clip_range": 1.0, "gae_lambda": 0.95},
    {"name": "short_sighted", "ent_coef": 0.01, "norm_obs": True, "gamma": 0.70, "clip_range": 1.0, "gae_lambda": 0.95},
    {"name": "strict_clipping", "ent_coef": 0.01, "norm_obs": True, "gamma": 0.99, "clip_range": 0.5, "gae_lambda": 0.95},
    
    # --- NEW LAMBDA EXPERIMENTS ---
    # High Lambda = Lower Bias, Higher Variance (trusts actual rewards more)
    {"name": "lambda_high_variance", "ent_coef": 0.01, "norm_obs": True, "gamma": 0.99, "clip_range": 1.0, "gae_lambda": 1.0},
    # Low Lambda = Higher Bias, Lower Variance (trusts the Critic's 'guess' more)
    {"name": "lambda_low_variance", "ent_coef": 0.01, "norm_obs": True, "gamma": 0.99, "clip_range": 1.0, "gae_lambda": 0.80},
]

TOTAL_TIMESTEPS = 1_000_000 

# --- 2. CUSTOM ACTION CLIPPING WRAPPER ---
class ActionClipWrapper(gym.ActionWrapper):
    def __init__(self, env, limit=1.0):
        super().__init__(env)
        self.limit = limit
    def action(self, action):
        return np.clip(action, -self.limit, self.limit)

def make_env(clip_range=1.0):
    def _init():
        env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
        env = ActionClipWrapper(env, limit=clip_range)
        env = Monitor(env)
        return env
    return _init

# --- 3. MASTER TRAINING LOOP ---
if __name__ == "__main__":
    for exp in experiments:
        print(f"\n🚀 STARTING EXPERIMENT: {exp['name']}")
        
        model_dir = f"./models/{exp['name']}"
        os.makedirs(model_dir, exist_ok=True)

        env = DummyVecEnv([make_env(clip_range=exp['clip_range'])])
        
        if exp['norm_obs']:
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
        
        # Initialize PPO Model with the new gae_lambda parameter
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./logs/",
            ent_coef=exp['ent_coef'],
            gamma=exp['gamma'],
            gae_lambda=exp['gae_lambda'], # <--- ADDED THIS LINE
            device="auto"
        )

        model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=exp['name'])
        
        model.save(f"{model_dir}/ppo_model")
        if exp['norm_obs']:
            env.save(f"{model_dir}/vec_normalize.pkl")

        # --- 4. VIDEO EVALUATION ---
        print(f"📹 Generating final video for: {exp['name']}")
        eval_env = DummyVecEnv([make_env(clip_range=exp['clip_range'])])
        
        if exp['norm_obs']:
            eval_env = VecNormalize.load(f"{model_dir}/vec_normalize.pkl", eval_env)
            eval_env.training = False 
            eval_env.norm_reward = False 
            
        eval_env = VecVideoRecorder(
            eval_env, 
            f"./videos/{exp['name']}",
            record_video_trigger=lambda x: x == 0, 
            video_length=1600,
            name_prefix=f"eval_{exp['name']}"
        )

        obs = eval_env.reset()
        for _ in range(1600):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = eval_env.step(action)
            if done:
                break
        
        eval_env.close()

    print("\n✅ All experiments complete. Use 'tensorboard --logdir ./logs/' to view graphs.")