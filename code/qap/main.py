import numpy as np
import pandas as pd
import torch
from qap.environment import TradingEnv
from qap.agent import DDPGAgent

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
            # if episode > 0:
            #     print(episode)
            action = agent.get_action(state)
            next_state, reward, done, action_success = env.step(action.argmax())
            agent.remember(state, action, reward, next_state, done)
            agent.update(batch_size=16)
            state = next_state
            total_reward += reward
        print("Episode {}: Total profit = {}".format(episode, env.profit))

    state = env.reset()
    total_reward = 0.0
    # 记录买入和卖出点
    prices = []
    buy_points = []
    sell_points = []
    i = 0
    while not env.done:
        action = agent.get_action(state)
        next_state, reward, done, action_success = env.step(action.argmax())
        # 标记买入和卖出点
        prices.append(data[i][1])
        if action_success == True and action.argmax() == 1:
            buy_points.append(i)
            print('1')
        elif action_success == True and action.argmax() == 2:
            sell_points.append(i)
            print('2')
        i += 1
        state = next_state
        total_reward += reward

    print("Total profit = {}".format(env.profit))

    # 绘制价格曲线和买入/卖出点
    import matplotlib.pyplot as plt
    plt.plot(prices)
    plt.plot(buy_points, [prices[i] for i in buy_points], 'go')
    plt.plot(sell_points, [prices[i] for i in sell_points], 'ro')
    plt.show()


if __name__ == "__main__":
    main()
