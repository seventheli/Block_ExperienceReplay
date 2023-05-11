import os
import tqdm
import numpy as np
from mlflow.exceptions import MlflowException
from utils import log_with_timeout
from func_timeout import FunctionTimedOut

def train(algorithm, settings, checkout_path, mlflow_client, mlflow_run, yml_path):
    # list to save data
    episode_rewards = []
    episode_lengths = []
    eval_episode_rewards = []
    eval_episode_lengths = []
    gpu_util = []
    for i in tqdm.tqdm(range(1, 10000)):
        train_results = algorithm.train()
        total_time = train_results["time_total_s"]
        if total_time > settings.control.max_time:
            break
        # pref
        time_usage = train_results["info"]["learner"]["time_usage"]
        avg_gpu = float(np.mean(gpu_util))

        if train_results["info"]["learner"].get("default_policy", None) is not None:
            episode_rewards.extend(train_results["hist_stats"]["episode_reward"])
            episode_lengths.extend(train_results["hist_stats"]["episode_lengths"])
            sampled = train_results["num_agent_steps_sampled"]
            trained = train_results["num_agent_steps_trained"]
            try:
                gpu_util.append(train_results["perf"]["gpu_util_percent0"])
            except:
                pass

            def log_pred():
                # pref data
                log_with_timeout(client=mlflow_client, run_id=mlflow_run.info.run_id,
                                 key="GPU_Usage", value=avg_gpu, step=i)
                log_with_timeout(client=mlflow_client, run_id=mlflow_run.info.run_id,
                                 key="Num_Trained", value=trained, step=i)
                log_with_timeout(client=mlflow_client, run_id=mlflow_run.info.run_id,
                                 key="Num_Sampled", value=sampled, step=i)
                log_with_timeout(client=mlflow_client, run_id=mlflow_run.info.run_id,
                                 key="Store", value=time_usage["store"], step=i)
                log_with_timeout(client=mlflow_client, run_id=mlflow_run.info.run_id,
                                 key="Train", value=time_usage["train"], step=i)
                log_with_timeout(client=mlflow_client, run_id=mlflow_run.info.run_id,
                                 key="Sample", value=time_usage["sample"], step=i)
                log_with_timeout(client=mlflow_client, run_id=mlflow_run.info.run_id,
                                 key="Update", value=time_usage["update"], step=i)
                log_with_timeout(client=mlflow_client, run_id=mlflow_run.info.run_id,
                                 key="Train_Step", value=time_usage["all"], step=i)

            def log_reward():
                mean_reward = train_results["episode_reward_mean"]
                episode_reward = np.array(train_results["hist_stats"]["episode_reward"])
                episode_length = np.array(train_results["hist_stats"]["episode_lengths"])
                mean_reward_by_step = (episode_reward / episode_length).mean()
                tqdm.tqdm.write("episode %d, step %d, r=%.1f , logging" % (i, sampled, mean_reward))
                # reward
                log_with_timeout(client=mlflow_client, run_id=mlflow_run.info.run_id,
                                 key="Reward", value=mean_reward, step=i)
                log_with_timeout(client=mlflow_client, run_id=mlflow_run.info.run_id,
                                 key="Reward_By_Step", value=mean_reward_by_step, step=i)

            def log_eval():
                # eval data
                eval_data = train_results["evaluation"]
                eval_episode_rewards.extend(eval_data["hist_stats"]["episode_reward"])
                eval_episode_lengths.extend(eval_data["hist_stats"]["episode_lengths"])
                eval_mean_reward = eval_data["episode_reward_mean"]
                episode_reward = np.array(eval_data["hist_stats"]["episode_reward"])
                episode_length = np.array(eval_data["hist_stats"]["episode_lengths"])
                eval_mean_reward_by_step = (episode_reward / episode_length).mean()
                log_with_timeout(client=mlflow_client, run_id=mlflow_run.info.run_id,
                                 key="Eval_Reward", value=eval_mean_reward, step=i)
                log_with_timeout(client=mlflow_client, run_id=mlflow_run.info.run_id,
                                 key="Eval_Reward_By_Step", value=eval_mean_reward_by_step, step=i)

            if i == 10 or i % settings.control.log == 0:
                try:
                    log_reward()
                    log_pred()
                except FunctionTimedOut:
                    tqdm.tqdm.write("logging failed")
                except MlflowException:
                    tqdm.tqdm.write("logging failed")
            if train_results.get("evaluation", None) is not None:
                try:
                    log_reward()
                    log_eval()
                    log_pred()
                except FunctionTimedOut:
                    tqdm.tqdm.write("logging failed")
                except MlflowException:
                    tqdm.tqdm.write("logging failed")
                eval_score = train_results["evaluation"]["episode_reward_mean"]
                if total_time > settings.control.min_time and eval_score > settings.control.score:
                    break
            if i > 0 and i % settings.control.save == 0:
                try:
                    np.save(os.path.join(checkout_path, "episode_reward.npy"),
                            np.array([episode_rewards, episode_lengths]))
                    np.save(os.path.join(checkout_path, "eval_episode_reward.npy"),
                            np.array([eval_episode_rewards, eval_episode_lengths]))
                    mlflow_client.log_artifact(mlflow_run.info.run_id,
                                               os.path.join(checkout_path, "episode_reward.npy"))
                    mlflow_client.log_artifact(mlflow_run.info.run_id,
                                               os.path.join(checkout_path, "eval_episode_reward.npy"))
                    # save model
                    checkpoint = algorithm.save()
                    mlflow_client.log_artifact(mlflow_run.info.run_id, checkpoint)
                    mlflow_client.log_artifact(mlflow_run.info.run_id, yml_path)
                except:
                    pass
            if total_time > settings.control.min_time and episode_rewards > settings.control.score:
                break
