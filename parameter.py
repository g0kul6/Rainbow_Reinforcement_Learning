#creating a environment to test our agent
import gym
#creating the environment CartPole
env = gym.make('CartPole-v0')

#size of state and action
state_size=env.observation_space.shape[0]
action_size=env.action_space.n

#batch size for optimization
batch_size=32

#numer of episodes for training
n_episodes=20000

#experience replay size for replay buffer
replay_size=256

