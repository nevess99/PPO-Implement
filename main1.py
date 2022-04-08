import gym
import pygame
import matplotlib.pyplot as plt
import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from CP_Env_v2 import CartPoleEnv


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []  # states encountered
        self.probs = []  # log probability
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    # list of integers that correspond to the indices of our memories and then
    # we're going to have batch size chunks of those memories
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)  # number of states in trajectory
        np.random.shuffle(indices)  # for stochastic gradient descent
        batches = [indices[i:i + self.batch_size] for i in batch_start]     # all possible starting points from batches, and go from indices all the way to i+batchsee

        # return entire array of each of these values,
        return np.array(self.states), \
               np.array(self.actions), \
               np.array(self.probs), \
               np.array(self.vals), \
               np.array(self.rewards), \
               np.array(self.dones), \
               batches  # iterate over batches

    # store in memory by appending each element to respective list
    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    # clear mem at every trajectory
    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


# derive from nn.module class
# learning rate alpha; number of fully connected layers 1st amd 2nd; checkout point directory
class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
                 fc1_dims=256, fc2_dims=256, chkpt_dir='ppo'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppoMC')
        # deep neural network
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),  # linear layer, takes input_dims and outputs fc1_dims
            nn.ReLU(),                          #
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)  # takes care of the use of probabilities, sum to 1
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)  # optimize learning rate
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')  # use GPU if possible
        self.to(self.device)

    # feed forward func
    def forward(self, state):
        dist = self.actor(state)    # pass state thru deep nn and get the distribution
        dist = Categorical(dist)

        return dist  # calc series of probabilities used to draw from a distribution to get action
        # used to get log of probabilities for calc of the ratio of 2 prob in our update for the learning function

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='ppo'):     # output of this network is single value: output value of single state regardless of n_actions
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppoMC')      # critic model file
        # self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)    # optimizer w learning rate alpha, same lr for actor and critic
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    # pass state through critic network
    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


# base agent class
class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0002, gae_lambda=0.95,
                 policy_clip=0.1, batch_size=5, n_epochs=4):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    # func handles interface between agent and memory
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)    # convert np array to t tensor. Add batch dimension for deep nn

        # pass through nn
        dist = self.actor(state)    # distribution for choosing an action
        value = self.critic(state)      # value of particular state
        action = dist.sample()      # to get action, sample distribution

        # squeeze to get rid of batch dimensions
        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    # learning function
    def learn(self):
        for _ in range(self.n_epochs):  # iterate over number of epochs i.e 3 epochs
            state_arr, action_arr, old_prob_arr, vals_arr, \
            reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)     # array of zeros

            for t in range(len(reward_arr) - 1):    # for each time step
                discount = 0.99
                a_t = 0     # advantage at each time step
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] *
                                       (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)

            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                # take states that we have already encountered, the old prob according to old actor parameters.
                # pass states through actor network, to get new distribution to calculate for new probabilities
                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)     # get new values according to the updated values of critic network
                # calculate new prob, and get probability ratio
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                # prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                                                 1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()  # clear memory at every epoch


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    #env = gym.make('MountainCar-v0')
    #env = CartPoleEnv()
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0002
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  input_dims=env.observation_space.shape)
    n_games = 1000
    UPDATE_EVERY = 20
    ep_rewards = []
    aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

    figure_file = 'plots/McData1.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()

        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)    # choose action based on current state of env
            observation_, reward, done, info = env.step(action)     # new state, reward, done info back from env
            # env.render()
            # pygame.event.get()
            n_steps += 1    # after every action number of steps goes up by 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_      # current state to new state of env

        ep_rewards.append(score)
        if (i % UPDATE_EVERY == 0):
            lastest_episodes = ep_rewards[-UPDATE_EVERY:]
            #average_reward = sum(lastest_episodes) / len(lastest_episodes)
            aggr_ep_rewards['ep'].append(i)
            aggr_ep_rewards['avg'].append(avg_score)
            aggr_ep_rewards['min'].append(min(lastest_episodes))
            aggr_ep_rewards['max'].append(max(lastest_episodes))
                #print(
                #    f"Episode: {n_games} avg: {avg_score}, min:{min(lastest_episodes)}, max: {max(lastest_episodes)}")
        score_history.append(score)     # end of every episode
        avg_score = np.mean(score_history[100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('Episode', i, 'Score %.1f' % score, 'Avg Score %.1f' % avg_score,
              'Time_steps', n_steps, 'Learning_steps', learn_iters)

    # plot
    x = [i + 1 for i in range(len(score_history))]
    #plot_learning_curve(x, score_history, figure_file)

    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
    plt.legend()
    #plt.show()
    plt.savefig('plots/graph1000eps.pdf')


