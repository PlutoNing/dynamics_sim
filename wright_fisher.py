import numpy as np
from dynamics import StochasticDynamicsSimulator
""" Wright-Fisher 过程 是一种描述 基因漂变（genetic drift）和种群变化的模型，最早由 Sewall Wright 
和 Ronald Fisher 提出。它通常用于模拟在有限种群中，个体策略或基因在多代之间如何变化，且不考虑自然选择的作用。

Wright-Fisher 过程的基本特点：
固定种群大小：种群大小是常数，每一代都有相同数量的个体。
随机重组：在每一代中，每个个体都有可能从上一代的个体中随机选择一个副本进行繁殖。这意味着某些个体可能会消失，
某些个体会被多次复制。
基因漂变：在有限种群中，随机过程会导致某些基因或策略的消失或固定（即某个基因型完全占据整个种群）。
在进化博弈中，Wright-Fisher 过程通常用于模拟策略如何随机演化，特别是当选择压力较弱时，基因漂变和随机过程的
作用更加显著。 """

class WrightFisher(StochasticDynamicsSimulator):
    def __init__(self, mu=0.05, *args, **kwargs):
        # TODO: don't allow pop_size of 0, wright fisher only works with finite pop size
        super(WrightFisher, self).__init__(*args, **kwargs)
        self.mu = mu

    def next_generation(self, previous_state):
        state = []

        fitness = self.calculate_fitnesses(previous_state)

        # Generate offspring population probabilistically based on
        # fitness/avg_fitness, with some potential for each individual to be mutated
        for player_idx, (strategy_distribution, fitnesses, num_players) in enumerate(zip(previous_state, fitness, self.num_players)):
            num_strats = len(strategy_distribution)
            total_mutations = 0
            new_player_state = np.zeros(num_strats)
            for strategy_idx, n in enumerate(strategy_distribution):
                f = fitness[player_idx][strategy_idx]

                # sample from binomial distribution to get number of mutations for strategy
                if n == 0:
                    mutations = 0
                else:
                    mutations = np.random.binomial(n, self.mu)
                n -= mutations
                total_mutations += mutations
                new_player_state[strategy_idx] = n * f
                # distribute player strategies proportional n * f
                # don't use multinomial, because that adds randomness we don't want yet
            new_player_state *= float(num_players - total_mutations) / new_player_state.sum()
            new_player_state = np.array(self.round_individuals(new_player_state))

            new_player_state += np.random.multinomial(total_mutations, [1. / num_strats] * num_strats)
            state.append(new_player_state)

        return state
