from game import SymmetricNPlayerGame


class HawkDoveBourgeois(SymmetricNPlayerGame):
    """
    A class used to to represent the 2 player hawk-dove-bourgeois game. See U{http://www.life.umd.edu/classroom/zool360/L18-ESS/ess.html}
    """
    DEFAULT_PARAMS = dict(v=30, c=60)
    STRATEGY_LABELS = ('Hawk', 'Dove', 'Bourgeois')

    def __init__(self, v, c):
        payoff_matrix = (((v - c) / 2.0, v, 3.0 * v / 4.0 - c / 4.0),
                         (0, v / 2.0, v / 4.0),
                         ((v - c) / 4.0, 3.0 * v / 4.0, v / 2.0))

        super(HawkDoveBourgeois, self).__init__(payoff_matrix, 2)
