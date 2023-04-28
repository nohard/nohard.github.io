import numpy as np
import pandas as pd
from qap.environment import TradingEnv

def test(agent):
    df = pd.read_csv('IBM_5min_year1month1_with_rsi.csv')
    df = df.dropna()
    data = df[['open', 'close', 'high', 'low', 'volume', 'rsi']].values
    technical = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    behavior = np.zeros(20)
    env = TradingEnv(technical, behavior)
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
