A generic, extensible simulation library for evolutionary game theory simulations. DyPy provides [Moran](http://en.wikipedia.org/wiki/Moran_process), and [Wright-Fisher](http://en.wikipedia.org/wiki/Genetic_drift#Wright.E2.80.93Fisher_model) processes, as well as [Replicator Dynamics](http://en.wikipedia.org/wiki/Replicator_equation) and makes it simple to execute complex and robust simulations across a range of parameters and visualize the results with beautiful graphs.

See documentation [here](http://ecbtln.github.io).

####Requirements

DyPy depends on [matplotlib](http://matplotlib.org) for graphing, and [numpy](http://www.numpy.org) and [joblib](https://pythonhosted.org/joblib/). To install these dependencies, make sure you are in the root directory of the repo and run the following command, which may require sudo.

```bash
$ pip install -r requirements.txt
```

####Usage

开始使用 DyPy 的最简单方法是将类子类化，并通过将其收益矩阵适当地定义为各种参数的函数来定义要模拟的感兴趣游戏。您还可以定义一个函数，将 equlibria 分类为玩每种策略的玩家分布的函数。

定义游戏类后，选择一个动态过程并执行所需的模拟。一些选项包括:

- 模拟一个模拟的给定世代数，并绘制每个玩家策略随时间变化的动态图
- 多次重复给定的模拟，并返回每个结果均衡的频率。.
- 将一个或多个参数更改为 dynamics 或 game 构造函数，并在 2D 或 2D 图形中绘制此变化对结果均衡的影响.

The ```GameDynamicsWrapper``` and ```VariedGame``` classes take care of simplifying the simulation and graphing processes, and automatically parallelize the computations across all available cores.
GameDynamicsWrapper和VariedGame类负责简化仿真和绘图过程，并自动在所有可用内核之间并行计算。

To see an example, take a look at the [*Cooperate Without Looking*](https://github.com/ecbtln/cwol_sim/blob/master/cwol.py) subclass along with its associated [simulations](https://github.com/ecbtln/cwol_sim/blob/master/test.py).
要查看示例，请查看 Collaborate Without Looking 子类及其关联的模拟。

####Persistence (coming soon)

DyPy decouples the process of simulating with graphing. This encourages users to run long-running simulations and gather tons of data, and then insert and tweak the graph parameters afterwards.
DyPy 将仿真过程与绘图解耦。这鼓励用户运行长时间运行的模拟并收集大量数据，然后在之后插入和调整图形参数。