import numpy as np
import pandas as pd
from qap.environment import TradingEnv
from qap.agent import DDPGAgent
from qap.test import test

def main():
    df = pd.read_csv('IBM_5min_year1month2_with_rsi.csv')
    df = df.dropna()
    data = df[['open', 'close', 'high', 'low', 'volume', 'rsi']].values
    technical = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    behavior = np.zeros(20)

    env = TradingEnv(technical, behavior)
    agent = DDPGAgent(6, 20, 5, 5, 3, 128)

    for episode in range(50):
        state = env.reset()
        total_reward = 0.0
        while not env.done:
            action = agent.get_action(state)
            next_state, reward, done, action_success = env.step(action.argmax())
            agent.remember(state, action, reward, next_state, done)
            agent.update(batch_size=16)
            state = next_state
            total_reward += reward
        print("Episode {}: Total profit = {}".format(episode, env.profit))

    # test
    test(agent)

if __name__ == "__main__":
    main()
