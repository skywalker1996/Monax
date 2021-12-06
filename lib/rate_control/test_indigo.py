import numpy as np
from lib.rate_control.models_indigo import DaggerLSTM
import operator
import tensorflow as tf
from os import path
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import warnings

warnings.filterwarnings("ignore")

model_path = "lib/rate_control/model"
math_ops = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv,
}


def apply_op(op, op1, op2):
    return math_ops[op](op1, op2)

def normalize(state):
    return [state[0] / 200, state[1] / 200,
            state[2] / 200, state[3] / 5000]

def one_hot(action, action_cnt):
    ret = [0.0] * action_cnt
    ret[action] = 1.0
    return ret

def format_actions(action_list):
    """ Returns the action list, initially a list with elements "[op][val]"
    like /2.0, -3.0, +1.0, formatted as a dictionary.

    The dictionary keys are the unique indices (to retrieve the action) and
    the values are lists ['op', val], such as ['+', '2.0'].
    """
    return {idx: [action[0], float(action[1:])]
                  for idx, action in enumerate(action_list)}

def get_best_action(actions, cwnd, target):
    """ Returns the best action by finding the action that leads to the
    closest resulting cwnd to target.
    """
    return min(actions,
               key=lambda idx: action_error(actions, idx, cwnd, target))

def action_error(actions, idx, cwnd, target):
    """ Returns the absolute difference between the target and an action
    applied to the cwnd.
    The action is [op, val] located at actions[idx].
    """
    op = actions[idx][0]
    val = actions[idx][1]
    return abs(apply_op(op, cwnd, val) - target)

class Test(object):
    def __init__(self,restore_vars):
        self.state_dim = 4 #Sender.state_dim
        self.action_mapping = format_actions(["/2.0", "-10.0", "+0.0", "+10.0", "*2.0"])
        action_cnt = len(self.action_mapping)
        self.action_cnt = action_cnt #Sender.action_cnt
        self.best_cwnd=10 #define by yourself
        self.aug_state_dim = self.state_dim + self.action_cnt
        self.prev_action = self.action_cnt - 1
        self.state_buf = []
        self.action_buf = []
        self.curr_ep = 0
        with tf.variable_scope('global'):
            self.model = DaggerLSTM(
                        state_dim=self.aug_state_dim, action_cnt=self.action_cnt)
        self.lstm_state = self.model.zero_init_state(1)

        self.sess = tf.Session()

        # restore saved variables
        saver = tf.train.Saver(self.model.trainable_vars)
        saver.restore(self.sess, restore_vars)

        # init the remaining vars, especially those created by optimizer
        uninit_vars = set(tf.global_variables())
        uninit_vars -= set(self.model.trainable_vars)
        self.sess.run(tf.variables_initializer(uninit_vars))


    def sample_action(self, state):
        """ Given a state buffer in the past step, returns an action
        to perform.

        Appends to the state/action buffers the state and the
        "correct" action to take according to the expert.
        """

        self.cwnd = state[self.state_dim - 1]
        #expert_action = get_best_action(self.action_mapping, cwnd, self.best_cwnd)

        # For decision-making, normalize.
        norm_state = normalize(state)

        one_hot_action = one_hot(self.prev_action, self.action_cnt)
        aug_state = norm_state + one_hot_action

        # Fill in state_buf, action_buf
        #self.state_buf.append(aug_state)
        #self.action_buf.append(expert_action)

        # Always use the expert on the first episode to get our bearings.
        '''if self.curr_ep == 0:
            self.prev_action = expert_action
            return expert_action'''

        # Get probability of each action from the local network.
        pi = self.model
        feed_dict = {
            pi.input: [[aug_state]],
            pi.state_in: self.lstm_state,
        }
        ops_to_run = [pi.action_probs, pi.state_out]
        action_probs, self.lstm_state = self.sess.run(ops_to_run, feed_dict)

        # Choose an action to take and update current LSTM state
        # action = np.argmax(np.random.multinomial(1, action_probs[0][0] - 1e-5))
        action = np.argmax(action_probs[0][0])
        self.prev_action = action

        return action

    def take_action(self, action_idx):
        old_cwnd = self.cwnd
        op, val = self.action_mapping[action_idx]

        self.cwnd = apply_op(op, self.cwnd, val)
        self.cwnd = max(2.0, self.cwnd)

        return int(self.cwnd)

if __name__ == '__main__':

    test=Test(restore_vars=model_path)
    state = [34,24,45,10]
    action = test.sample_action(state)
    cwnd=test.take_action(action)