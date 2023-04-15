import numpy as np
import time
import matplotlib.pyplot as plt
import math

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import mdptoolbox
import mdptoolbox.example



def colors_lake():
    return {
        b'S': 'green',
        b'F': 'skyblue',
        b'H': 'black',
        b'G': 'gold',
    }


def directions_lake():
    return {
        3: '⬆',
        2: '➡',
        1: '⬇',
        0: '⬅'
    }


def run_episode(env, policy, gamma, render=True):
    obs = env.reset()[0]

    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done, _, _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma, n=100):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)


def extract_policy(env, v, gamma):
    policy = np.zeros(env.observation_space.n)
    for s in range(env.observation_space.n):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy


def compute_policy_v(env, policy, gamma):
    v = np.zeros(env.observation_space.n)
    eps = 1e-5
    while True:
        prev_v = np.copy(v)
        for s in range(env.observation_space.n):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, is_done in env.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            break
    return v


def policy_iteration(env, gamma):
    policy = np.random.choice(env.action_space.n, size=(env.observation_space.n))
    max_iters = 50000
    policy_diff = [len(policy)]
    reward_list = [0]
    for i in range(max_iters):
        v = np.zeros(env.observation_space.n)
        eps = 1e-5
        while True:
            prev_v = np.copy(v)
            for s in range(env.observation_space.n):
                policy_a = policy[s]
                v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, is_done in env.P[s][policy_a]])
            if (np.sum((np.fabs(prev_v - v))) <= eps):
                break

        new_policy = extract_policy(env, v, gamma)
        test = policy == new_policy
        diff = len(test) - np.sum(test)
        policy_diff.append(diff)
        mean_score = evaluate_policy(env, new_policy, gamma)
        reward_list.append(mean_score)
        if (np.all(test)):
            k = i + 1
            break
        policy = new_policy

    return policy, k, policy_diff, reward_list


def value_iteration(env, gamma):
    v = np.zeros(env.observation_space.n)
    max_iters = 50000
    eps = 1e-10
    desc = env.unwrapped.desc
    value_diff = []
    score_list = []
    for i in range(max_iters):
        prev_v = np.copy(v)
        for s in range(env.observation_space.n):
            q_sa = [sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.action_space.n)]
            v[s] = max(q_sa)
        diff = np.sum(np.fabs(prev_v - v))
        value_diff.append(diff)
        score_list.append(np.mean(v))
        if (np.sum(diff) <= eps):
            k = i + 1
            break
    return v, k, value_diff, score_list


def plot_policy_map(title, policy, map_desc, color_map, direction_map):
    M = policy.shape[1]
    N = policy.shape[0]

    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(0, M), ylim=(0, N))
    font_size = 'x-large'
    if M > 16:
        font_size = 'small'
    for i in range(N):
        for j in range(M):
            y = N - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1)
            p.set_facecolor(color_map[map_desc[i, j]])
            ax.add_patch(p)

            text = ax.text(x + 0.5, y + 0.5, direction_map[policy[i, j]], weight='bold', size=font_size,
                           horizontalalignment='center', verticalalignment='center', color='w')

    plt.axis('off')
    plt.xlim((0, M))
    plt.ylim((0, N))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'{title}.png')
    plt.close()

    return (plt)


def Frozen_Lake(size):
    env = gym.make('FrozenLake-v1', desc=generate_random_map(size=size))
    env = env.unwrapped
    desc = env.unwrapped.desc

    gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

    time_array = [0] * len(gammas)
    iters = [0] * len(gammas)
    list_scores = [0] * len(gammas)
    policy_diff_list = []
    reward_list = []

    print(f'Frozen Lake {size}: PI')
    for i in range(len(gammas)):
        gamma = gammas[i]
        print(gamma)
        st = time.time()
        best_policy, k, policy_diff, score_list = policy_iteration(env, gamma=gamma)
        scores = evaluate_policy(env, best_policy, gamma=gamma)
        end = time.time()
        list_scores[i] = np.mean(scores)
        iters[i] = k
        time_array[i] = end - st
        policy_diff_list.append(policy_diff)
        reward_list.append(score_list)

    plt.plot(gammas, time_array)
    plt.xlabel('Gammas')
    title = f'PI-Execution Time vs Gammas'
    plt.title(title)
    plt.ylabel('Execution Time (s)')
    plt.grid()
    plt.savefig(f'plots/frozen_{size}/{title.replace(" ", "_")}.png')
    plt.clf()

    plt.plot(gammas, list_scores)
    plt.xlabel('Gammas')
    plt.ylabel('Average Rewards')
    title = f'PI-Reward vs Gammas'
    plt.title(title)
    plt.grid()
    plt.savefig(f'plots/frozen_{size}/{title.replace(" ", "_")}.png')
    plt.clf()

    plt.plot(gammas, iters)
    plt.xlabel('Gammas')
    plt.ylabel('Iterations to Converge')
    title = f'PI-Convergence vs Gammas'
    plt.title(title)
    plt.grid()
    plt.savefig(f'plots/frozen_{size}/{title.replace(" ", "_")}.png')
    plt.clf()

    for g in range(len(gammas)):
        plt.plot(reward_list[g], label=f'Gamma={gammas[g]}')
    plt.legend(loc="best")
    plt.xlabel('Number of Iterations')
    plt.ylabel('Average Rewards')
    title = f'PI-Reward vs Iterations'
    plt.title(title)
    plt.grid()
    plt.savefig(f'plots/frozen_{size}/{title.replace(" ", "_")}.png')
    plt.clf()

    for g in range(len(gammas)):
        plt.plot(policy_diff_list[g], label=f'Gamma={gammas[g]}')
    plt.legend(loc="best")
    plt.xlabel('Number of Iterations')
    plt.ylabel('Policy Difference')
    title = f'PI-Policy Difference vs Iterations'
    plt.title(title)
    plt.grid()
    plt.savefig(f'plots/frozen_{size}/{title.replace(" ", "_")}.png')
    plt.clf()

    print(f'Frozen Lake {size}: VI')
    time_array = [0] * len(gammas)
    iters = [0] * len(gammas)
    list_scores = [0] * len(gammas)

    best_vals = [0] * len(gammas)
    rewards_list = []
    value_diff_list = []
    for i in range(len(gammas)):
        st = time.time()
        gamma = gammas[i]
        best_value, k, value_diff, score_list = value_iteration(env, gamma=gamma)
        policy = extract_policy(env, best_value, gamma=gamma)
        policy_score = evaluate_policy(env, policy, gamma=gamma, n=100)
        plot = plot_policy_map(f'plots/frozen_{size}/Policy_Map_Gamma:{gamma}',
                               policy.reshape(size, size), desc, colors_lake(),
                               directions_lake())
        end = time.time()
        iters[i] = k
        best_vals[i] = best_value
        list_scores[i] = np.mean(policy_score)
        time_array[i] = end - st
        rewards_list.append(np.array(score_list))
        value_diff_list.append(value_diff)

    plt.plot(gammas, time_array)
    plt.xlabel('Gammas')
    title = f'VI-Execution Time'
    plt.title(title)
    plt.ylabel('Execution Time (s)')
    plt.grid()
    plt.savefig(f'plots/frozen_{size}/{title.replace(" ", "_")}.png')
    plt.clf()

    plt.plot(gammas, list_scores)
    plt.xlabel('Gammas')
    plt.ylabel('Average Rewards')
    title = f'VI-Reward vs Gammas'
    plt.title(title)
    plt.grid()
    plt.savefig(f'plots/frozen_{size}/{title.replace(" ", "_")}.png')
    plt.clf()

    plt.plot(gammas, iters)
    plt.xlabel('Gammas')
    plt.ylabel('Iterations to Converge')
    title = f'VI-Convergence vs Gammas'
    plt.title(title)
    plt.grid()
    plt.savefig(f'plots/frozen_{size}/{title.replace(" ", "_")}.png')
    plt.clf()

    plt.plot(gammas, best_vals)
    plt.legend(loc='best')
    plt.xlabel('Gammas')
    plt.ylabel('Optimal Value')
    title = f'VI-Best Value'
    plt.title(title)
    plt.grid()
    plt.savefig(f'plots/frozen_{size}/{title.replace(" ", "_")}.png')
    plt.clf()

    for g in range(len(gammas)):
        plt.plot(value_diff_list[g], label=f'Gamma={gammas[g]}')
    plt.legend(loc='best')
    plt.xlabel('Iterations')
    plt.ylabel('Value Difference')
    title = f'VI-Value Difference vs Iterations'
    plt.title(title)
    plt.grid()
    plt.savefig(f'plots/frozen_{size}/{title.replace(" ", "_")}.png')
    plt.clf()

    for g in range(len(gammas)):
        plt.plot(rewards_list[g], label=f'Gamma={gammas[g]}')
    plt.legend(loc='best')
    plt.xlabel('Iterations')
    plt.ylabel('Average Rewards')
    title = f'VI-Rewards vs Iterations'
    plt.title(title)
    plt.grid()
    plt.savefig(f'plots/frozen_{size}/{title.replace(" ", "_")}.png')
    plt.clf()

    print(f'Frozen Lake {size}: QL')
    epsilon_list = [0.8, 0.85, 0.9, 0.95, 0.99]

    st = time.time()
    reward_array = []
    iter_array = []
    size_array = []
    averages_array = []
    time_array = []
    Q_array = []
    for epsilon in epsilon_list:
        Q = np.zeros((env.observation_space.n, env.action_space.n))
        rewards = []
        iters = []
        optimal = [0] * env.observation_space.n
        alpha = 0.85
        gamma = 0.95
        env = gym.make('FrozenLake-v1', desc=generate_random_map(size=size))
        env = env.unwrapped
        desc = env.unwrapped.desc
        for episode in range(10000):
            state = env.reset()[0]
            done = False
            t_reward = 0
            max_steps = 50000
            for i in range(max_steps):
                if done:
                    break
                current = state
                if np.random.rand() < (epsilon):
                    action = np.argmax(Q[current, :])
                else:
                    action = env.action_space.sample()

                state, reward, done, _, _ = env.step(action)
                t_reward += reward
                Q[current, action] += alpha * (reward + gamma * np.max(Q[state, :]) - Q[current, action])
            epsilon = (1 - math.e ** (-episode / 1000))
            rewards.append(t_reward)
            iters.append(i)

        for k in range(env.observation_space.n):
            optimal[k] = np.argmax(Q[k, :])

        reward_array.append(rewards)
        iter_array.append(iters)
        Q_array.append(Q)

        env.close()
        end = time.time()
        time_array.append(end - st)

        def chunk_list(l, n):
            for i in range(0, len(l), n):
                yield l[i:i + n]

        size = int(10000 / 50)
        chunks = list(chunk_list(rewards, size))
        averages = [sum(chunk) / len(chunk) for chunk in chunks]
        size_array.append(size)
        averages_array.append(averages)

    for i in range(len(epsilon_list)):
        plt.plot(range(0, len(reward_array[i]), size_array[i]), averages_array[i], label=f'epsilon={epsilon_list[i]}')

    plt.legend(loc='best')
    plt.xlabel('Iterations')
    plt.grid()
    title = f'Q Learning-Constant Epsilon'
    plt.title(title)
    plt.ylabel('Average Reward')
    plt.savefig(f'plots/frozen_{size}/{title.replace(" ", "_")}.png')
    plt.clf()

    plt.plot(epsilon_list, time_array)
    plt.legend(loc='best')
    plt.xlabel('Epsilon Values')
    plt.grid()
    title = f'Q Learning'
    plt.title(title)
    plt.ylabel('Execution Time (s)')
    plt.savefig(f'plots/frozen_{size}/{title.replace(" ", "_")}.png')
    plt.clf()


def Forest(size):
    print(f'Forest Management {size}: PI')
    gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

    exec_time = []
    reward = []
    iter_list = []

    for states in size:
        P, R = mdptoolbox.example.forest(S=states)
        mean_value = []
        policy = []
        iters = []
        time_array = []
        best_policy = None
        best_score = 0
        for i in range(len(gammas)):
            gamma = gammas[i]
            pi = mdptoolbox.mdp.PolicyIteration(P, R, gamma)
            pi.run()
            mean_value.append(np.mean(pi.V))
            if np.mean(pi.V) > best_score:
                best_policy = np.array(pi.policy)
                best_score = np.mean(pi.V)
            policy.append(pi.policy)
            iters.append(pi.iter)
            time_array.append(pi.time)
        exec_time.append(time_array)
        reward.append(mean_value)
        iter_list.append(iters)
        best_policy = best_policy.astype(str)
        best_policy[best_policy == '0'] = 'W'
        best_policy[best_policy == '1'] = 'C'

    plt.plot(gammas, exec_time[0], label='State - 100')
    plt.plot(gammas, exec_time[1], label='State - 200')
    plt.plot(gammas, exec_time[2], label='State - 500')
    plt.plot(gammas, exec_time[3], label='State - 1000')
    plt.xlabel('Gamma')
    plt.title('PI-Gamma vs Execution Time')
    plt.ylabel('Execution Time (s)')
    plt.grid()
    plt.legend()
    plt.savefig("plots/forest/PI_ExecutionTime.png")
    plt.clf()

    plt.plot(gammas, reward[0], label='State - 100')
    plt.plot(gammas, reward[1], label='State - 200')
    plt.plot(gammas, reward[2], label='State - 500')
    plt.plot(gammas, reward[3], label='State - 1000')
    plt.xlabel('Gamma')
    plt.ylabel('Average Rewards')
    plt.title('PI-Gamma vs Reward')
    plt.grid()
    plt.legend()
    plt.savefig("plots/forest/PI_AverageRewards.png")
    plt.clf()

    plt.plot(gammas, iter_list[0], label='State - 100')
    plt.plot(gammas, iter_list[1], label='State - 200')
    plt.plot(gammas, iter_list[2], label='State - 500')
    plt.plot(gammas, iter_list[3], label='State - 1000')
    plt.xlabel('Gamma')
    plt.ylabel('Iterations')
    plt.title('PI-Gamma vs Convergence')
    plt.grid()
    plt.legend()
    plt.savefig("plots/forest/PI_Iterations.png")
    plt.clf()

    print(f'Forest Management {size}: VI')
    exec_time = []
    reward = []
    iter_list = []

    for states in size:
        P, R = mdptoolbox.example.forest(S=states)
        mean_value = []
        policy = []
        iters = []
        time_array = []
        best_policy = None
        best_score = 0
        for i in range(len(gammas)):
            pi = mdptoolbox.mdp.ValueIteration(P, R, gammas[i])
            pi.run()
            mean_value.append(np.mean(pi.V))
            policy.append(pi.policy)
            if np.mean(pi.V) > best_score:
                best_policy = np.array(pi.policy)
                best_score = np.mean(pi.V)
            iters.append(pi.iter)
            time_array.append(pi.time)
        exec_time.append(time_array)
        reward.append(mean_value)
        iter_list.append(iters)
        best_policy = best_policy.astype(str)
        best_policy[best_policy == '0'] = 'W'
        best_policy[best_policy == '1'] = 'C'

    plt.plot(gammas, exec_time[0], label='State - 100')
    plt.plot(gammas, exec_time[1], label='State - 200')
    plt.plot(gammas, exec_time[2], label='State - 500')
    plt.plot(gammas, exec_time[3], label='State - 1000')
    plt.xlabel('Gamma')
    plt.title('VI-Gamma vs Execution Time')
    plt.ylabel('Execution Time (s)')
    plt.grid()
    plt.legend()
    plt.savefig("plots/forest/VI_ExecutionTime.png")
    plt.clf()

    plt.plot(gammas, reward[0], label='State - 100')
    plt.plot(gammas, reward[1], label='State - 200')
    plt.plot(gammas, reward[2], label='State - 500')
    plt.plot(gammas, reward[3], label='State - 1000')
    plt.xlabel('Gamma')
    plt.ylabel('Average Rewards')
    plt.title('VI-Gamma vs Reward')
    plt.grid()
    plt.legend()
    plt.savefig("plots/forest/VI_AverageRewards.png")
    plt.clf()

    plt.plot(gammas, iter_list[0], label='State - 100')
    plt.plot(gammas, iter_list[1], label='State - 200')
    plt.plot(gammas, iter_list[2], label='State - 500')
    plt.plot(gammas, iter_list[3], label='State - 1000')
    plt.xlabel('Gamma')
    plt.ylabel('Iterations')
    plt.title('VI-Gamma vs Convergence')
    plt.grid()
    plt.legend()
    plt.savefig("plots/forest/VI_Iterations.png")
    plt.clf()


    print(f'Forest Management {size}: QL')
    exec_time = []
    reward = []
    iter_list = []

    for states in size:
        P, R = mdptoolbox.example.forest(S=states)
        mean_value = []
        policy = []
        iters = []
        time_array = []
        best_policy = None
        best_score = 0
        for i in range(0, 10):
            pi = mdptoolbox.mdp.QLearning(P, R, gammas[i])
            pi.run()
            mean_value.append(np.mean(pi.V))
            if np.mean(pi.V) > best_score:
                best_policy = np.array(pi.policy)
                best_score = np.mean(pi.V)
            policy.append(pi.policy)
            time_array.append(pi.time)
        exec_time.append(time_array)
        reward.append(mean_value)
        iter_list.append(iters)
        best_policy = best_policy.astype(str)
        best_policy[best_policy == '0'] = 'W'
        best_policy[best_policy == '1'] = 'C'

    plt.plot(gammas, exec_time[0], label='State - 100')
    plt.plot(gammas, exec_time[1], label='State - 200')
    plt.plot(gammas, exec_time[2], label='State - 500')
    plt.plot(gammas, exec_time[3], label='State - 1000')
    plt.xlabel('Gamma')
    plt.title('Q-Gamma vs Execution Time')
    plt.ylabel('Execution Time (s)')
    plt.grid()
    plt.legend()
    plt.savefig("plots/forest/Q_ExecutionTime.png")
    plt.clf()

    plt.plot(gammas, reward[0], label='State - 100')
    plt.plot(gammas, reward[1], label='State - 200')
    plt.plot(gammas, reward[2], label='State - 500')
    plt.plot(gammas, reward[3], label='State - 1000')
    plt.xlabel('Gamma')
    plt.ylabel('Average Rewards')
    plt.title('Q-Gamma vs Reward')
    plt.grid()
    plt.legend()
    plt.savefig("plots/forest/Q_AverageRewards.png")
    plt.clf()
    return


Frozen_Lake(6)
Frozen_Lake(16)
Forest([100, 200, 500, 1000])
