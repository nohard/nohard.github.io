import numpy as np

class TradingEnv:
    def __init__(self, technical, behavior):
        self.technical = technical
        self.behavior = behavior
        self.len = len(technical)
        self.reset()

    def reset(self):
        self.behavior = np.zeros(20)
        self.t = 0
        self.done = False
        self.profit = 0.0
        self.hold = False
        self.hold_price = 0
        self.buy_step = 0
        self.sell_step = 0
        return self.get_state()

    def step(self, action):
        reward = 0
        action_success = False
        if action == 1:
            # Buy
            # print(action)
            if self.hold == False:
                self.hold = True
                self.hold_price = self.technical[self.t][1]
                self.buy_step = self.t
                profit = 10 * (self.technical[self.t + 1][0] - self.hold_price)
                reward += profit
                action_success = True
        elif action == 2:
            # Sell
            # print(action)
            if self.hold == True:
                self.hold = False
                self.sell_step = self.t
                sell_price = self.technical[self.t][1]
                profit = 10 * (2*sell_price - self.hold_price - self.technical[self.t+1][1])
                self.profit += 10 * (sell_price - self.hold_price)
                reward += profit
                action_success = True

        # 惩罚频繁交易和长时间无交易
        if abs(self.sell_step - self.buy_step) < 10 or min(self.t - self.buy_step, self.t - self.sell_step) > 50:
            reward -= 1

        self.t += 1
        if self.t == self.len - 1:
            self.done = True

        if action > 0 and action_success == False:
            action = -1
            # 惩罚无效交易
            reward -= 0.5

        # 更新behavior
        self.behavior[:-1] = self.behavior[1:]
        self.behavior[-1] = action

        next_state = self.get_state()
        return next_state, reward, self.done, action_success,

    def get_state(self):
        if self.t >= self.len:
            raise Exception("End of data reached")
        state = np.concatenate([self.technical[self.t], self.behavior])
        return state
