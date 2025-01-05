from dynamics import StochasticDynamicsSimulator
import numpy
""" Moran 过程 是一个描述有限种群中个体演化的模型，尤其是基因频率变化的过程。它由 P.A. Moran 提出，
通常用于模拟有限种群中的个体如何通过竞争和繁殖传递其策略。

Moran 过程的基本特点：
固定种群大小：种群的大小是固定的，每一代中个体数目不变。
个体的死亡与繁殖：每一代随机选择一个个体进行死亡，并且该个体的复制会替代它的位置。复制个体的策略是随机
选择的，即可能是任何现有个体的策略。
策略传播：与 Wright-Fisher 过程不同，Moran 过程不仅考虑个体的随机选择，还考虑了策略的竞争和传播。
在每个更新步骤中，个体根据其 适应度（fitness） 来被选择繁殖，适应度较高的个体有更高的概率被选择进行繁殖。
应用：
Moran 过程 更适用于有 自然选择 的情况，因为繁殖是基于适应度的，即适应度较高的个体有更大的繁殖机会。
它常用于模拟博弈中的 策略竞争，如某一策略（例如“合作”）的个体会倾向于通过复制自身的策略在种群中传播，
尤其是在策略间有 不同的适应度 时。 """

class Moran(StochasticDynamicsSimulator):
    """
    A stochastic dynamics simulator that performs the Moran process on all player types in the population.
    See U{Moran Process<http://en.wikipedia.org/wiki/Moran_process#Selection>}
    """
    def __init__(self, num_iterations_per_time_step=1, *args, **kwargs):
        """
        The constructor for the Moran dynamics process, that the number of births/deaths to process per time step.

        @param num_iterations_per_time_step: the number of iterations of the Moran process we do per time step
        @type num_iterations_per_time_step: int
        """
        super(Moran, self).__init__(*args, **kwargs)
        assert num_iterations_per_time_step >= 1
        self.num_iterations_per_time_step = num_iterations_per_time_step

    def next_generation(self, previous_state):
        next_state = []

        # copy to new state
        for p in previous_state:
            next_state.append(p.copy())

        fitness = self.calculate_fitnesses(next_state)

        minimum_total = min(p.sum() for p in next_state)
        # make sure there are enough individuals of each type to take away 2 * num_iterations_per_time_step
        num_iterations = min(self.num_iterations_per_time_step * 2, minimum_total) / 2
        num_iterations = int(num_iterations)

        for p, f in zip(next_state, fitness):
            reproduce = numpy.zeros(len(p))

            for i in range(num_iterations):
                # sample from distribution to determine winner and loser (he who reproduces, he who dies)
                weighted_total = sum(n_i * f_i for n_i, f_i in zip(p, f))
                dist = numpy.array([n_i * f_i / weighted_total for n_i, f_i in zip(p, f)])
                sample = numpy.random.multinomial(1, dist)
                p -= sample
                reproduce += sample

            for i in range(num_iterations):
                # now determine who dies from what's left
                total = p.sum()
                dist = [n_i / float(total) for n_i in p]
                p -= numpy.random.multinomial(1, dist)
            print(type(p))
            print(type(reproduce))
            p = (p + reproduce *2 )


        return next_state
