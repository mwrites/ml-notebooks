"""
Utilities module containing helper functions for the Deep Q-Learning - Lunar Lander
Jupyter notebook (C3_W3_A1_Assignment) from DeepLearning.AI's "Unsupervised Learning,
Recommenders, Reinforcement Learning" course on Coursera.
"""

import base64
import random
from itertools import zip_longest

import imageio
import IPython
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 0  # Seed for the pseudo-random number generator.
MINIBATCH_SIZE = 64  # Mini-batch size.
TAU = 1e-3  # Soft update parameter.
E_DECAY = 0.995  # ε-decay rate for the ε-greedy policy.
E_MIN = 0.01  # Minimum ε value for the ε-greedy policy.


random.seed(SEED)


def get_experiences(memory_buffer):
    """
    Returns a random sample of experience tuples drawn from the memory buffer.

    Retrieves a random sample of experience tuples from the given memory_buffer and
    returns them as TensorFlow Tensors. The size of the random sample is determined by
    the mini-batch size (MINIBATCH_SIZE). 
    
    Args:
        memory_buffer (deque):
            A deque containing experiences. The experiences are stored in the memory
            buffer as namedtuples: namedtuple("Experience", field_names=["state",
            "action", "reward", "next_state", "done"]).

    Returns:
        A tuple (states, actions, rewards, next_states, done_vals) where:

            - states are the starting states of the agent.
            - actions are the actions taken by the agent from the starting states.
            - rewards are the rewards received by the agent after taking the actions.
            - next_states are the new states of the agent after taking the actions.
            - done_vals are the boolean values indicating if the episode ended.

        All tuple elements are TensorFlow Tensors whose shape is determined by the
        mini-batch size and the given Gym environment. For the Lunar Lander environment
        the states and next_states will have a shape of [MINIBATCH_SIZE, 8] while the
        actions, rewards, and done_vals will have a shape of [MINIBATCH_SIZE]. All
        TensorFlow Tensors have elements with dtype=tf.float32.
    """

    experiences = random.sample(memory_buffer, k=MINIBATCH_SIZE)
    states = tf.convert_to_tensor(
        np.array([e.state for e in experiences if e is not None]), dtype=tf.float32
    )
    actions = tf.convert_to_tensor(
        np.array([e.action for e in experiences if e is not None]), dtype=tf.float32
    )
    rewards = tf.convert_to_tensor(
        np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32
    )
    next_states = tf.convert_to_tensor(
        np.array([e.next_state for e in experiences if e is not None]), dtype=tf.float32
    )
    done_vals = tf.convert_to_tensor(
        np.array([e.done for e in experiences if e is not None]).astype(np.uint8),
        dtype=tf.float32,
    )
    return (states, actions, rewards, next_states, done_vals)


def check_update_conditions(t, num_steps_upd, memory_buffer):
    """
    Determines if the conditions are met to perform a learning update.

    Checks if the current time step t is a multiple of num_steps_upd and if the
    memory_buffer has enough experience tuples to fill a mini-batch (for example, if the
    mini-batch size is 64, then the memory buffer should have more than 64 experience
    tuples in order to perform a learning update).
    
    Args:
        t (int):
            The current time step.
        num_steps_upd (int):
            The number of time steps used to determine how often to perform a learning
            update. A learning update is only performed every num_steps_upd time steps.
        memory_buffer (deque):
            A deque containing experiences. The experiences are stored in the memory
            buffer as namedtuples: namedtuple("Experience", field_names=["state",
            "action", "reward", "next_state", "done"]).

    Returns:
       A boolean that will be True if conditions are met and False otherwise. 
    """

    if (t + 1) % num_steps_upd == 0 and len(memory_buffer) > MINIBATCH_SIZE:
        return True
    else:
        return False


def get_new_eps(epsilon):
    """
    Updates the epsilon value for the ε-greedy policy.
    
    Gradually decreases the value of epsilon towards a minimum value (E_MIN) using the
    given ε-decay rate (E_DECAY).

    Args:
        epsilon (float):
            The current value of epsilon.

    Returns:
       A float with the updated value of epsilon.
    """

    return max(E_MIN, E_DECAY * epsilon)


def get_action(q_values, epsilon=0.0):
    """
    Returns an action using an ε-greedy policy.

    This function will return an action according to the following rules:
        - With probability epsilon, it will return an action chosen at random.
        - With probability (1 - epsilon), it will return the action that yields the
        maximum Q value in q_values.
    
    Args:
        q_values (tf.Tensor):
            The Q values returned by the Q-Network. For the Lunar Lander environment
            this TensorFlow Tensor should have a shape of [1, 4] and its elements should
            have dtype=tf.float32. 
        epsilon (float):
            The current value of epsilon.

    Returns:
       An action (numpy.int64). For the Lunar Lander environment, actions are
       represented by integers in the closed interval [0,3].
    """

    if random.random() > epsilon:
        if isinstance(q_values, tf.Tensor):
            return int(tf.argmax(q_values[0]).numpy())
        else:
            return int(np.argmax(q_values[0]))
    else:
        return random.randint(0, 3)


def update_target_network(q_network, target_q_network):
    """
    Updates the weights of the target Q-Network using a soft update.
    
    The weights of the target_q_network are updated using the soft update rule:
    
                    w_target = (TAU * w) + (1 - TAU) * w_target
    
    where w_target are the weights of the target_q_network, TAU is the soft update
    parameter, and w are the weights of the q_network.
    
    Args:
        q_network (tf.keras.Sequential): 
            The Q-Network. 
        target_q_network (tf.keras.Sequential):
            The Target Q-Network.
    """

    for target_weights, q_net_weights in zip(
        target_q_network.weights, q_network.weights
    ):
        target_weights.assign(TAU * q_net_weights + (1.0 - TAU) * target_weights)


def plot_history(point_history, **kwargs):
    """
    Plots the total number of points received by the agent after each episode together
    with the moving average (rolling mean). 

    Args:
        point_history (list):
            A list containing the total number of points the agent received after each
            episode.
        **kwargs: optional
            window_size (int):
                Size of the window used to calculate the moving average (rolling mean).
                This integer determines the fixed number of data points used for each
                window. The default window size is set to 10% of the total number of
                data points in point_history, i.e. if point_history has 200 data points
                the default window size will be 20.
            lower_limit (int):
                The lower limit of the x-axis in data coordinates. Default value is 0.
            upper_limit (int):
                The upper limit of the x-axis in data coordinates. Default value is
                len(point_history).
            plot_rolling_mean_only (bool):
                If True, only plots the moving average (rolling mean) without the point
                history. Default value is False.
            plot_data_only (bool):
                If True, only plots the point history without the moving average.
                Default value is False.
    """

    lower_limit = 0
    upper_limit = len(point_history)

    window_size = (upper_limit * 10) // 100

    plot_rolling_mean_only = False
    plot_data_only = False

    if kwargs:
        if "window_size" in kwargs:
            window_size = kwargs["window_size"]

        if "lower_limit" in kwargs:
            lower_limit = kwargs["lower_limit"]

        if "upper_limit" in kwargs:
            upper_limit = kwargs["upper_limit"]

        if "plot_rolling_mean_only" in kwargs:
            plot_rolling_mean_only = kwargs["plot_rolling_mean_only"]

        if "plot_data_only" in kwargs:
            plot_data_only = kwargs["plot_data_only"]

    points = point_history[lower_limit:upper_limit]

    # Generate x-axis for plotting.
    episode_num = [x for x in range(lower_limit, upper_limit)]

    # Use Pandas to calculate the rolling mean (moving average).
    rolling_mean = pd.DataFrame(points).rolling(window_size).mean()

    plt.figure(figsize=(10, 7), facecolor="white")

    if plot_data_only:
        plt.plot(episode_num, points, linewidth=1, color="cyan")
    elif plot_rolling_mean_only:
        plt.plot(episode_num, rolling_mean, linewidth=2, color="magenta")
    else:
        plt.plot(episode_num, points, linewidth=1, color="cyan")
        plt.plot(episode_num, rolling_mean, linewidth=2, color="magenta")

    text_color = "black"

    ax = plt.gca()
    ax.set_facecolor("black")
    plt.grid()
    plt.xlabel("Episode", color=text_color, fontsize=30)
    plt.ylabel("Total Points", color=text_color, fontsize=30)
    yNumFmt = mticker.StrMethodFormatter("{x:,}")
    ax.yaxis.set_major_formatter(yNumFmt)
    ax.tick_params(axis="x", colors=text_color)
    ax.tick_params(axis="y", colors=text_color)
    plt.show()


def display_table(current_state, action, next_state, reward, done):
    """
    Displays a table containing the current state, action, next state, reward, and done
    values from Gym's Lunar Lander environment.

    All floating point numbers in the table are displayed rounded to 3 decimal places
    and actions are displayed using their labels instead of their numerical value (i.e
    if action = 0, the action will be printed as "Do nothing" instead of "0").

    Args:
        current_state (numpy.ndarray):
            The current state vector returned by the Lunar Lander environment 
            before an action is taken
        action (int):
            The action taken by the agent. In the Lunar Lander environment, actions are
            represented by integers in the closed interval [0,3] corresponding to:
                - Do nothing = 0
                - Fire right engine = 1
                - Fire main engine = 2
                - Fire left engine = 3
        next_state (numpy.ndarray):
            The state vector returned by the Lunar Lander environment after the agent
            takes an action, i.e the observation returned after running a single time
            step of the environment's dynamics using env.step(action).
        reward (numpy.float64):
            The reward returned by the Lunar Lander environment after the agent takes an
            action, i.e the reward returned after running a single time step of the
            environment's dynamics using env.step(action).
        done (bool):
            The done value returned by the Lunar Lander environment after the agent
            takes an action, i.e the done value returned after running a single time
            step of the environment's dynamics using env.step(action).
    
    Returns:
        table (Pandas Dataframe):
            A dataframe containing the current_state, action, next_state, reward,
            and done values. This will result in the table being displayed in the
            Jupyter Notebook.
    """
    
    STATE_VECTOR_COL_NAME = 'State Vector'
    DERIVED_COL_NAME = 'Derived from the State Vector (the closer to zero, the better)'
    
    # States
    add_derived_info = lambda state: np.hstack([
        state, 
        [(state[0]**2 + state[1]**2)**.5],
        [(state[2]**2 + state[3]**2)**.5],
        [np.abs(state[4])]
    ])
    
    modified_current_state = add_derived_info(current_state)
    modified_next_state = add_derived_info(next_state)
    
    states = np.vstack([
        modified_current_state, 
        modified_next_state,
        modified_next_state - modified_current_state,        
    ]).T
    
    get_state = lambda idx, type=np.float32: dict(zip(
        ['Current State', 'Next State'], 
        states[idx].astype(type)
    ))

    # Actions
    action_labels = [
        "Do nothing",
        "Fire right engine",
        "Fire main engine",
        "Fire left engine",
    ]

    display(
        pd.DataFrame({
            ('', '', ''): {'Action': action_labels[action], 'Reward': reward, 'Episode Terminated': done},
            (STATE_VECTOR_COL_NAME, 'Coordinate', 'X (Horizontal)'): get_state(0),
            (STATE_VECTOR_COL_NAME, 'Coordinate', 'Y (Vertical)'): get_state(1),
            (STATE_VECTOR_COL_NAME, 'Velocity', 'X (Horizontal)'): get_state(2),
            (STATE_VECTOR_COL_NAME, 'Velocity', 'Y (Vertical)'): get_state(3),
            (STATE_VECTOR_COL_NAME, 'Tilting', 'Angle'): get_state(4),
            (STATE_VECTOR_COL_NAME, 'Tilting', 'Angular Velocity'): get_state(5),
            (STATE_VECTOR_COL_NAME, 'Ground contact', 'Left Leg?'): get_state(6, np.bool),
            (STATE_VECTOR_COL_NAME, 'Ground contact', 'Right Leg?'): get_state(7, np.bool),
            (DERIVED_COL_NAME, 'Distance from landing pad', ''): get_state(8),
            (DERIVED_COL_NAME, 'Velocity', ''): get_state(9),
            (DERIVED_COL_NAME, 'Tilting Angle (absolute value)', ''): get_state(10),
        })\
            .fillna('')\
            .reindex(['Current State', 'Action', 'Next State', 'Reward', 'Episode Terminated'])\
            .style\
            .applymap(lambda x: 'background-color : grey' if x == '' else '')\
            .set_table_styles(
                [
                    {"selector": "th", "props": [("border", "1px solid grey"), ('text-align', 'center')]},
                    {"selector": "tbody td", "props": [("border", "1px solid grey"), ('text-align', 'center')]},
                ]
            )
    )


def embed_mp4(filename):
    """
    Embeds an MP4 video file in a Jupyter notebook.
    
    Args:
        filename (string):
            The path to the the MP4 video file that will be embedded (i.e.
            "./videos/lunar_lander.mp4").
    
    Returns:
        Returns a display object from the given video file. This will result in the
        video being displayed in the Jupyter Notebook.
    """

    video = open(filename, "rb").read()
    b64 = base64.b64encode(video)
    tag = """
    <video width="840" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>""".format(
        b64.decode()
    )

    return IPython.display.HTML(tag)


"""
Utilities module containing helper functions for the Deep Q-Learning - Lunar Lander
Jupyter notebook (C3_W3_A1_Assignment) from DeepLearning.AI's "Unsupervised Learning,
Recommenders, Reinforcement Learning" course on Coursera.
"""

import base64
import random
from itertools import zip_longest

import imageio
import IPython
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image, ImageDraw, ImageFont

SEED = 0  # Seed for the pseudo-random number generator.
MINIBATCH_SIZE = 64  # Mini-batch size.
TAU = 1e-3  # Soft update parameter.
E_DECAY = 0.995  # ε-decay rate for the ε-greedy policy.
E_MIN = 0.01  # Minimum ε value for the ε-greedy policy.


random.seed(SEED)


def get_experiences(memory_buffer):
    """
    Returns a random sample of experience tuples drawn from the memory buffer.

    Retrieves a random sample of experience tuples from the given memory_buffer and
    returns them as TensorFlow Tensors. The size of the random sample is determined by
    the mini-batch size (MINIBATCH_SIZE). 
    
    Args:
        memory_buffer (deque):
            A deque containing experiences. The experiences are stored in the memory
            buffer as namedtuples: namedtuple("Experience", field_names=["state",
            "action", "reward", "next_state", "done"]).

    Returns:
        A tuple (states, actions, rewards, next_states, done_vals) where:

            - states are the starting states of the agent.
            - actions are the actions taken by the agent from the starting states.
            - rewards are the rewards received by the agent after taking the actions.
            - next_states are the new states of the agent after taking the actions.
            - done_vals are the boolean values indicating if the episode ended.

        All tuple elements are TensorFlow Tensors whose shape is determined by the
        mini-batch size and the given Gym environment. For the Lunar Lander environment
        the states and next_states will have a shape of [MINIBATCH_SIZE, 8] while the
        actions, rewards, and done_vals will have a shape of [MINIBATCH_SIZE]. All
        TensorFlow Tensors have elements with dtype=tf.float32.
    """

    experiences = random.sample(memory_buffer, k=MINIBATCH_SIZE)
    states = tf.convert_to_tensor(
        np.array([e.state for e in experiences if e is not None]), dtype=tf.float32
    )
    actions = tf.convert_to_tensor(
        np.array([e.action for e in experiences if e is not None]), dtype=tf.float32
    )
    rewards = tf.convert_to_tensor(
        np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32
    )
    next_states = tf.convert_to_tensor(
        np.array([e.next_state for e in experiences if e is not None]), dtype=tf.float32
    )
    done_vals = tf.convert_to_tensor(
        np.array([e.done for e in experiences if e is not None]).astype(np.uint8),
        dtype=tf.float32,
    )
    return (states, actions, rewards, next_states, done_vals)


def check_update_conditions(t, num_steps_upd, memory_buffer):
    """
    Determines if the conditions are met to perform a learning update.

    Checks if the current time step t is a multiple of num_steps_upd and if the
    memory_buffer has enough experience tuples to fill a mini-batch (for example, if the
    mini-batch size is 64, then the memory buffer should have more than 64 experience
    tuples in order to perform a learning update).
    
    Args:
        t (int):
            The current time step.
        num_steps_upd (int):
            The number of time steps used to determine how often to perform a learning
            update. A learning update is only performed every num_steps_upd time steps.
        memory_buffer (deque):
            A deque containing experiences. The experiences are stored in the memory
            buffer as namedtuples: namedtuple("Experience", field_names=["state",
            "action", "reward", "next_state", "done"]).

    Returns:
       A boolean that will be True if conditions are met and False otherwise. 
    """

    if (t + 1) % num_steps_upd == 0 and len(memory_buffer) > MINIBATCH_SIZE:
        return True
    else:
        return False


def get_new_eps(epsilon):
    """
    Updates the epsilon value for the ε-greedy policy.
    
    Gradually decreases the value of epsilon towards a minimum value (E_MIN) using the
    given ε-decay rate (E_DECAY).

    Args:
        epsilon (float):
            The current value of epsilon.

    Returns:
       A float with the updated value of epsilon.
    """

    return max(E_MIN, E_DECAY * epsilon)


def get_action(q_values, epsilon=0.0):
    """
    Returns an action using an ε-greedy policy.

    This function will return an action according to the following rules:
        - With probability epsilon, it will return an action chosen at random.
        - With probability (1 - epsilon), it will return the action that yields the
        maximum Q value in q_values.
    
    Args:
        q_values (tf.Tensor):
            The Q values returned by the Q-Network. For the Lunar Lander environment
            this TensorFlow Tensor should have a shape of [1, 4] and its elements should
            have dtype=tf.float32. 
        epsilon (float):
            The current value of epsilon.

    Returns:
       An action (numpy.int64). For the Lunar Lander environment, actions are
       represented by integers in the closed interval [0,3].
    """

    if random.random() > epsilon:
        if isinstance(q_values, tf.Tensor):
            return int(tf.argmax(q_values[0]).numpy())
        else:
            return int(np.argmax(q_values[0]))
    else:
        return random.randint(0, 3)


def update_target_network(q_network, target_q_network):
    """
    Updates the weights of the target Q-Network using a soft update.
    
    The weights of the target_q_network are updated using the soft update rule:
    
                    w_target = (TAU * w) + (1 - TAU) * w_target
    
    where w_target are the weights of the target_q_network, TAU is the soft update
    parameter, and w are the weights of the q_network.
    
    Args:
        q_network (tf.keras.Sequential): 
            The Q-Network. 
        target_q_network (tf.keras.Sequential):
            The Target Q-Network.
    """

    for target_weights, q_net_weights in zip(
        target_q_network.weights, q_network.weights
    ):
        target_weights.assign(TAU * q_net_weights + (1.0 - TAU) * target_weights)


def plot_history(point_history, **kwargs):
    """
    Plots the total number of points received by the agent after each episode together
    with the moving average (rolling mean). 

    Args:
        point_history (list):
            A list containing the total number of points the agent received after each
            episode.
        **kwargs: optional
            window_size (int):
                Size of the window used to calculate the moving average (rolling mean).
                This integer determines the fixed number of data points used for each
                window. The default window size is set to 10% of the total number of
                data points in point_history, i.e. if point_history has 200 data points
                the default window size will be 20.
            lower_limit (int):
                The lower limit of the x-axis in data coordinates. Default value is 0.
            upper_limit (int):
                The upper limit of the x-axis in data coordinates. Default value is
                len(point_history).
            plot_rolling_mean_only (bool):
                If True, only plots the moving average (rolling mean) without the point
                history. Default value is False.
            plot_data_only (bool):
                If True, only plots the point history without the moving average.
                Default value is False.
    """

    lower_limit = 0
    upper_limit = len(point_history)

    window_size = (upper_limit * 10) // 100

    plot_rolling_mean_only = False
    plot_data_only = False

    if kwargs:
        if "window_size" in kwargs:
            window_size = kwargs["window_size"]

        if "lower_limit" in kwargs:
            lower_limit = kwargs["lower_limit"]

        if "upper_limit" in kwargs:
            upper_limit = kwargs["upper_limit"]

        if "plot_rolling_mean_only" in kwargs:
            plot_rolling_mean_only = kwargs["plot_rolling_mean_only"]

        if "plot_data_only" in kwargs:
            plot_data_only = kwargs["plot_data_only"]

    points = point_history[lower_limit:upper_limit]

    # Generate x-axis for plotting.
    episode_num = [x for x in range(lower_limit, upper_limit)]

    # Use Pandas to calculate the rolling mean (moving average).
    rolling_mean = pd.DataFrame(points).rolling(window_size).mean()

    plt.figure(figsize=(10, 7), facecolor="white")

    if plot_data_only:
        plt.plot(episode_num, points, linewidth=1, color="cyan")
    elif plot_rolling_mean_only:
        plt.plot(episode_num, rolling_mean, linewidth=2, color="magenta")
    else:
        plt.plot(episode_num, points, linewidth=1, color="cyan")
        plt.plot(episode_num, rolling_mean, linewidth=2, color="magenta")

    text_color = "black"

    ax = plt.gca()
    ax.set_facecolor("black")
    plt.grid()
    plt.xlabel("Episode", color=text_color, fontsize=30)
    plt.ylabel("Total Points", color=text_color, fontsize=30)
    yNumFmt = mticker.StrMethodFormatter("{x:,}")
    ax.yaxis.set_major_formatter(yNumFmt)
    ax.tick_params(axis="x", colors=text_color)
    ax.tick_params(axis="y", colors=text_color)
    plt.show()


def display_table(current_state, action, next_state, reward, done):
    """
    Displays a table containing the current state, action, next state, reward, and done
    values from Gym's Lunar Lander environment.

    All floating point numbers in the table are displayed rounded to 3 decimal places
    and actions are displayed using their labels instead of their numerical value (i.e
    if action = 0, the action will be printed as "Do nothing" instead of "0").

    Args:
        current_state (numpy.ndarray):
            The current state vector returned by the Lunar Lander environment 
            before an action is taken
        action (int):
            The action taken by the agent. In the Lunar Lander environment, actions are
            represented by integers in the closed interval [0,3] corresponding to:
                - Do nothing = 0
                - Fire right engine = 1
                - Fire main engine = 2
                - Fire left engine = 3
        next_state (numpy.ndarray):
            The state vector returned by the Lunar Lander environment after the agent
            takes an action, i.e the observation returned after running a single time
            step of the environment's dynamics using env.step(action).
        reward (numpy.float64):
            The reward returned by the Lunar Lander environment after the agent takes an
            action, i.e the reward returned after running a single time step of the
            environment's dynamics using env.step(action).
        done (bool):
            The done value returned by the Lunar Lander environment after the agent
            takes an action, i.e the done value returned after running a single time
            step of the environment's dynamics using env.step(action).
    
    Returns:
        table (Pandas Dataframe):
            A dataframe containing the current_state, action, next_state, reward,
            and done values. This will result in the table being displayed in the
            Jupyter Notebook.
    """
    
    STATE_VECTOR_COL_NAME = 'State Vector'
    DERIVED_COL_NAME = 'Derived from the State Vector (the closer to zero, the better)'
    
    # States
    add_derived_info = lambda state: np.hstack([
        state, 
        [(state[0]**2 + state[1]**2)**.5],
        [(state[2]**2 + state[3]**2)**.5],
        [np.abs(state[4])]
    ])
    
    modified_current_state = add_derived_info(current_state)
    modified_next_state = add_derived_info(next_state)
    
    states = np.vstack([
        modified_current_state, 
        modified_next_state,
        modified_next_state - modified_current_state,        
    ]).T
    
    get_state = lambda idx, type=np.float32: dict(zip(
        ['Current State', 'Next State'], 
        states[idx].astype(type)
    ))

    # Actions
    action_labels = [
        "Do nothing",
        "Fire right engine",
        "Fire main engine",
        "Fire left engine",
    ]

    display(
        pd.DataFrame({
            ('', '', ''): {'Action': action_labels[action], 'Reward': reward, 'Episode Terminated': done},
            (STATE_VECTOR_COL_NAME, 'Coordinate', 'X (Horizontal)'): get_state(0),
            (STATE_VECTOR_COL_NAME, 'Coordinate', 'Y (Vertical)'): get_state(1),
            (STATE_VECTOR_COL_NAME, 'Velocity', 'X (Horizontal)'): get_state(2),
            (STATE_VECTOR_COL_NAME, 'Velocity', 'Y (Vertical)'): get_state(3),
            (STATE_VECTOR_COL_NAME, 'Tilting', 'Angle'): get_state(4),
            (STATE_VECTOR_COL_NAME, 'Tilting', 'Angular Velocity'): get_state(5),
            (STATE_VECTOR_COL_NAME, 'Ground contact', 'Left Leg?'): get_state(6, np.bool),
            (STATE_VECTOR_COL_NAME, 'Ground contact', 'Right Leg?'): get_state(7, np.bool),
            (DERIVED_COL_NAME, 'Distance from landing pad', ''): get_state(8),
            (DERIVED_COL_NAME, 'Velocity', ''): get_state(9),
            (DERIVED_COL_NAME, 'Tilting Angle (absolute value)', ''): get_state(10),
        })\
            .fillna('')\
            .reindex(['Current State', 'Action', 'Next State', 'Reward', 'Episode Terminated'])\
            .style\
            .applymap(lambda x: 'background-color : grey' if x == '' else '')\
            .set_table_styles(
                [
                    {"selector": "th", "props": [("border", "1px solid grey"), ('text-align', 'center')]},
                    {"selector": "tbody td", "props": [("border", "1px solid grey"), ('text-align', 'center')]},
                ]
            )
    )


def embed_mp4(filename):
    """
    Embeds an MP4 video file in a Jupyter notebook.
    
    Args:
        filename (string):
            The path to the the MP4 video file that will be embedded (i.e.
            "./videos/lunar_lander.mp4").
    
    Returns:
        Returns a display object from the given video file. This will result in the
        video being displayed in the Jupyter Notebook.
    """

    video = open(filename, "rb").read()
    b64 = base64.b64encode(video)
    tag = """
    <video width="840" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>""".format(
        b64.decode()
    )

    return IPython.display.HTML(tag)


def create_video(filename, env, q_network, episode, text_color, fps=60, frame_skip=4):
    # Get the video dimensions
    temp_state, _ = env.reset()
    temp_frame = env.render()
    video_height, video_width = temp_frame.shape[:2]

    # Calculate padding
    pad_width = (16 - (video_width % 16)) % 16
    pad_height = (16 - (video_height % 16)) % 16

    new_width = video_width + pad_width
    new_height = video_height + pad_height

    video = imageio.get_writer(filename, fps=fps)
    done = False
    state, _ = env.reset()
    frame_count = 0
    
    while not done:
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = q_network(state_tensor)
        action = tf.argmax(action_probs[0]).numpy()
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        if frame_count % frame_skip == 0:
            frame = env.render()
            pil_image = Image.fromarray(frame)
            
            # Create a new padded image
            padded_image = Image.new('RGB', (new_width, new_height), (0, 0, 0))
            padded_image.paste(pil_image, (0, 0))
            
            # Create a transparent image for text overlay
            overlay_width = new_width // 2
            overlay_height = new_height // 2
            text_image = Image.new('RGBA', (overlay_width, overlay_height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(text_image)
            
            # Use a large font size
            font_size = overlay_height // 5  # Adjust this value to change text size
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            except IOError:
                font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', font_size)
                
            text = f"Attempt: {episode}"
            
            # Get text size
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Position text in center of the overlay
            position = ((overlay_width - text_width) // 2, (overlay_height - text_height) // 3)
            
            # Draw text with a black outline for better visibility
            outline_width = max(1, font_size // 15)
            for offset in [(x, y) for x in range(-outline_width, outline_width+1) 
                                   for y in range(-outline_width, outline_width+1)]:
                draw.text((position[0]+offset[0], position[1]+offset[1]), text, font=font, fill=(0,0,0))
            
            # Draw the main text in the specified color
            draw.text(position, text, font=font, fill=text_color)
            
            # Create a new image with the same size as the padded frame
            final_image = Image.new('RGBA', (new_width, new_height), (0, 0, 0, 0))
            
            # Paste the text image in the top-right corner of the final image
            final_image.paste(text_image, (new_width - overlay_width, 0))
            
            # Overlay the final image onto the padded frame
            padded_image = Image.alpha_composite(padded_image.convert('RGBA'), final_image)
            
            frame_with_text = np.ascontiguousarray(padded_image.convert('RGB'))
            
            video.append_data(frame_with_text)
        
        state = next_state
        frame_count += 1
    
    video.close()
    env.close()