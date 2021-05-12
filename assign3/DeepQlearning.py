import gym
from gym.wrappers import Monitor
import itertools
import numpy as np
import os
import random
import sys
import psutil
import tensorflow as tf
import matplotlib.pyplot as plt 

# if "../" not in sys.path:
#   sys.path.append("../")

import plotting
from collections import deque, namedtuple

env = gym.envs.make("Breakout-v0")

# Atari Actions: 0 (noop), 1 (fire), 2 (left) and 3 (right) are valid actions
valid_actions = [0, 1, 2, 3]

class StateProcessor():
    """
    Processes a raw Atari images. Resizes it and converts it to grayscale.
    """
    #  state: A [210, 160, 3] Atari RGB State to [84, 84] State representing grayscale values
    def __init__(self):
        # Build the Tensorflow graph
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(
                self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess, state):

        return sess.run(self.output, { self.input_state: state })


class Estimator():
    """Q-Value Estimator neural network.

    This network is used for both the Q-Network and the Target Network.
    """
    def __init__(self, scope="estimator"):
        self.scope = scope
        
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
   
    def _build_model(self):
        """
        Builds the Tensorflow graph.
        """
        # Placeholders for our input
        # Our input are 4 grayscale frames of shape 84, 84 each
        self.X_place = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        # The TD target value
        self.y_place = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        X = tf.to_float(self.X_place) / 255.0
        batch_size = tf.shape(self.X_place)[0]

        # Three convolutional layers
        conv1 = tf.contrib.layers.conv2d(
            X, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(
            conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(
            conv2, 64, 3, 1, activation_fn=tf.nn.relu)

        # Fully connected layers
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        self.predictions = tf.contrib.layers.fully_connected(fc1, len(valid_actions))
        
        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)
        

        ## [Your task1] ## Calculate the loss by using tf.squared_difference and tf.reduce_mean
        ############################################################################
        self.loss = tf.reduce_mean(tf.squared_difference(self.y_place, self.action_predictions))
        ############################################################################

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss,global_step=tf.contrib.framework.get_global_step())


    def predict(self, sess, s):
        """
        Predicts action values.
        Args:
          sess: Tensorflow session
          s: State input of shape [batch_size, 84, 84, 4]

        Returns:
          Tensor of shape [batch_size, NUM_valid_actions] containing the estimated 
          action values.
        """
        return sess.run(self.predictions, { self.X_place: s })

    def update(self, sess, s, a, y):
        """
        Updates the estimator towards the given targets.
        Args:
          sess: Tensorflow session object
          s: State input of shape [batch_size, 84, 84, 4]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        feed_dict = { self.X_place: s, self.y_place: y, self.actions_pl: a }
        _, loss = sess.run([self.train_op, self.loss],feed_dict)
       
        return loss

#########################################################################################
#########################################################################################
# For testing tensoflow model

tf.reset_default_graph() 
global_step = tf.Variable(0, name="global_step", trainable=False)

e = Estimator(scope="test")
sp = StateProcessor()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # Example observation batch
    observation = env.reset()
    observation_p = sp.process(sess, observation)
    observation = np.stack([observation_p] * 4, axis=2)
    observations = np.array([observation] * 2)
    
    # Test Prediction
    print(e.predict(sess, observations))

    # Test training step
    y = np.array([10.0, 10.0])
    a = np.array([1, 3])
    print(e.update(sess, observations, a, y))

#########################################################################################
#########################################################################################


class ModelParametersCopier():
    """
    Copy model parameters of one estimator to another.
    """
    def __init__(self, estimator1, estimator2):

        e1_params = [t for t in tf.trainable_variables() 
                     if t.name.startswith(estimator1.scope)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() 
                     if t.name.startswith(estimator2.scope)]
        e2_params = sorted(e2_params, key=lambda v: v.name)

        self.update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            self.update_ops.append(op)
            
    def make(self, sess):
        """
        Makes copy.
        Args:
            sess: Tensorflow session instance
        """
        sess.run(self.update_ops)

def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.

    Returns:
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(sess, observation, epsilon):
        ## [Your task2] ## Epsilon-greddy policy
        ############################################################################
        # (1) Define 1d array
        A = np.ones(nA, dtype=float) * epsilon / nA
        # (2) Returns q values for a given state by using estimator
        q_val = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        # (3) Find the action that maximizes input q_value
        max_act = np.argmax(q_val)
        # (4) 1-\epsilon+\epsilon/nA(in case of best action)
        A[max_act] = A[max_act] + 1 - epsilon
        ############################################################################

        return A
    return policy_fn

def deep_q_learning(sess,
                    env,
                    q_estimator,
                    target_estimator,
                    state_processor,
                    num_episodes,
                    experiment_dir,
                    replay_memory_size=500000,
                    replay_memory_init_size=50000,
                    update_target_estimator=10000,
                    discount_factor=0.99,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=500000,
                    batch_size=32,
                    record_video_every=50):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        sess: Tensorflow Session object
        env: OpenAI environment
        q_estimator: Estimator object used for the q values
        target_estimator: Estimator object used for the targets
        state_processor: A StateProcessor object
        num_episodes: Number of episodes to run for
        experiment_dir: Directory to save Tensorflow summaries in
        replay_memory_size: Size of the replay memory
        replay_memory_init_size: Number of random experiences to sampel when initializing the reply memory.
        update_target_estimator: Copy parameters from the Q estimator to the target estimator every N steps
        discount_factor: Gamma discount factor
        epsilon_start: Chance to sample a random action when taking an action.
          Epsilon is decayed over time and this is the start value
        epsilon_end: The final minimum value of epsilon after decaying is done
        epsilon_decay_steps: Number of steps to decay epsilon over
        batch_size: Size of batches to sample from the replay memory
        record_video_every: Record a video every N episodes

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    # The replay memory
    replay_memory = []
    
    # Make model copier object
    estimator_copy = ModelParametersCopier(q_estimator, 
                                           target_estimator)

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    
    # For 'system/' summaries, usefull to check if currrent process looks healthy
    current_process = psutil.Process()
  
    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
   
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver = tf.train.Saver()
    # Load a previous checkpoint if we find one
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)
    
    # Get the current time step
    total_t = sess.run(tf.contrib.framework.get_global_step())
    
    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # The policy we're following
    policy = make_epsilon_greedy_policy(q_estimator, len(valid_actions))

    # Populate the replay memory with initial experience
    print("Populating replay memory...")
    state = env.reset()
    state = state_processor.process(sess, state)
    state = np.stack([state] * 4, axis=2)
    for i in range(replay_memory_init_size):

        action_probs = policy(sess, state, epsilons[min(total_t, epsilon_decay_steps-1)])
        ## [Your task3] ## 
        ############################################################################
        # (1) # Pick the first action (action sampling based on probability vector) by using np.random.choice
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        # (2) Take a step
        next_state, reward, done, _ = env.step(valid_actions[action])
        ############################################################################
        next_state = state_processor.process(sess, next_state)
        next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
        

        replay_memory.append(Transition(state, action, reward, next_state, done))
        if done:
            state = env.reset()
            state = state_processor.process(sess, state)
            state = np.stack([state] * 4, axis=2)
        else:
            state = next_state


    for i_episode in range(num_episodes):

        # Save the current checkpoint
        saver.save(tf.get_default_session(), checkpoint_path)

        # Reset the environment
        state = env.reset()
        state = state_processor.process(sess, state)
        state = np.stack([state] * 4, axis=2)
        loss = None

        # One step in the environment
        for t in itertools.count():

            # Epsilon for this time step
            epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]

            ## [Your task4] ## Update the target estimator and print "Copied model parameters to target network."
            ############################################################################
            if (total_t % update_target_estimator == 0):
                estimator_copy.make(sess)
                print("\nCopied model parameters to target network.")
            ############################################################################
        

            # Print out which step we're on, useful for debugging.
            print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
                    t, total_t, i_episode + 1, num_episodes, loss), end="")
            sys.stdout.flush()


            ## [Your task5] ## Take a step (REFER: Task3)
            ############################################################################
            action_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(valid_actions[action])
            next_state = state_processor.process(sess, next_state)
            next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)
            ############################################################################
            
            # If our replay memory is full, pop the first element
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            # Save transition to replay memory
            replay_memory.append(Transition(state, action, reward, next_state, done))   

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            

            
            ## [Your task6] ## 
            ############################################################################
            # (1) Sample a minibatch from the replay memory by using random.sample
            samples = random.sample(replay_memory, batch_size)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

            # (2) Calculate q values and targets
            next_q_val = q_estimator.predict(sess, next_states_batch)
            max_acts = np.argmax(next_q_val, axis=1)
            next_q_val_target = target_estimator.predict(sess, next_states_batch)
            targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                        discount_factor * next_q_val_target[np.arange(batch_size), max_acts]
        
            # (3) Perform gradient descent update
            states_batch = np.array(states_batch)
            loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)
            
            ############################################################################
            
            if done:
                break

            state = next_state
            total_t += 1
        
        yield total_t, plotting.EpisodeStats(
            episode_lengths=stats.episode_lengths[:i_episode+1],
            episode_rewards=stats.episode_rewards[:i_episode+1])

    return stats

tf.reset_default_graph()

# Where we save our checkpoints and graphs
experiment_dir = os.path.abspath("./experiments/{}".format(env.spec.id))

# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)
    
# Create estimators
q_estimator = Estimator(scope="q_estimator")
target_estimator = Estimator(scope="target_q")

# State processor
state_processor = StateProcessor()

# Run it!
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for t, stats in deep_q_learning(sess,
                                    env,
                                    q_estimator=q_estimator,
                                    target_estimator=target_estimator,
                                    state_processor=state_processor,
                                    experiment_dir=experiment_dir,
                                    num_episodes=10000,
                                    replay_memory_size=500000,
                                    replay_memory_init_size=50000,
                                    update_target_estimator=10000,
                                    epsilon_start=1.0,
                                    epsilon_end=0.1,
                                    epsilon_decay_steps=500000,
                                    discount_factor=0.99,
                                    batch_size=32):

        print("\nEpisode Reward: {}".format(stats.episode_rewards[-1]))

############################################################################
# Write your answer for problem 1. (c) in below:
############################################################################
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
############################################################################