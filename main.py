from flask import Flask, render_template, redirect, url_for
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from src import RobloxBedwars
import threading
import json
import time
import logging
from datetime import datetime

app = Flask(__name__)

logging.getLogger('werkzeug').disabled = True

training = False
stop_training_flag = False
model = None
envs = None
start_model = "ppo_roblox_bedwars_raycasts.zip"

def create_env(i):
    def _init():
        env = Monitor(RobloxBedwars(), filename=f'./logs/{datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}rb_env{str(i)}')
        return env
    return _init

def make_envs(num_envs=4):
    return SubprocVecEnv([create_env(i) for i in range(num_envs)], start_method='spawn')

@app.get("/poll/<player>")
def poll(player):
    if not envs:
        print("The environments have not been initialized yet")
        return {
            'reset_character': False,
            'calculate_reward_poll': False,
            'take_action_poll': None,
            'get_observation': False
        }  # do nothing until envs initialized

    env_idx = int(player) - 1
    print( envs.get_attr('reset_character', env_idx))
    requesting = {
        'reset_character': envs.get_attr('reset_character', env_idx)[0],
        'calculate_reward_poll': not envs.get_attr('calculate_reward_poll', env_idx)[0],
        'take_action_poll': envs.get_attr('take_action_poll', env_idx)[0],
        'get_observation': not envs.get_attr('get_observation', env_idx)[0]
    }
    return requesting

@app.get("/respond_to_poll/<data>")
def respond_to_poll(data):
    if not envs:
        return 'The environments have not been initialized yet'

    data = json.loads(data)
    envs.env_method('poll_callback', data, indices=data['player'] - 1)
    return "success"

@app.route("/")
def home():
    global training
    return render_template("index.html", training=training)

def training_loop():
    global model, stop_training_flag
    i = 0
    while not stop_training_flag:
        i += 1
        start_time = time.time()
        num_timesteps = 500
        model.learn(total_timesteps=num_timesteps, tb_log_name="run" + str(i), log_interval=1, progress_bar=True)  # Use a smaller number of timesteps for more frequent interactions
        end_time = time.time()
        total_time = end_time - start_time
        average_time_per_timestep = total_time / num_timesteps
        print("Average seconds taken per timestep", average_time_per_timestep)
        model.save('ppo_roblox_bedwars_raycasts.zip')

@app.route("/start_training")
def start_training():
    global training, stop_training_flag, model, envs
    training = True
    stop_training_flag = False

    envs = make_envs()
    try:
        model = PPO.load(start_model, env=envs, tensorboard_log="./roblox_bedwars_tensorboard/")
        print("Model loaded successfully.")
    except FileNotFoundError:
        model = PPO('MultiInputPolicy', env=envs, verbose=1, tensorboard_log="./roblox_bedwars_tensorboard/")
        print("No pre-trained model found. Starting fresh.")

    thread = threading.Thread(target=training_loop)
    thread.start()

    return redirect(url_for("home"))

@app.route("/stop_training")
def stop_training():
    global stop_training_flag
    stop_training_flag = True
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=False)
