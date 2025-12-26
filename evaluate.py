import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder

# 1. Setup Environment
env_id = "BipedalWalker-v3"
# Use rgb_array for video recording
eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])

# 2. Load Stats and Model
eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)
eval_env.training = False
eval_env.norm_reward = False 

model = PPO.load("ppo_bipedal_final", env=eval_env)

# 3. Wrap for Final Video
eval_env = VecVideoRecorder(
    eval_env, 
    "videos/final_evaluation/",
    record_video_trigger=lambda x: x == 0, 
    video_length=1600,
    name_prefix="final_walker"
)

# 4. Run Evaluation
print("Recording final post-training video...")
obs = eval_env.reset()
for i in range(1600):
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = eval_env.step(action)
    if dones:
        break

print("Final video saved in videos/final_evaluation/")
eval_env.close()