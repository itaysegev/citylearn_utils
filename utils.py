from ipywidgets import IntProgress
from citylearn.citylearn import CityLearnEnv
from stable_baselines3.common.callbacks import BaseCallback


class CustomCallback(BaseCallback):
    def __init__(self, env: CityLearnEnv, loader: IntProgress):
        r"""Initialize CustomCallback.

        Parameters
        ----------
        env: Mapping[str, CityLearnEnv]
            CityLearn environment instance.
        loader: IntProgress
            Progress bar.
        """

        super().__init__(verbose=0)
        self.loader = loader
        self.env = env
        self.reward_history = [0]

    def _on_step(self) -> bool:
        r"""Called each time the env step function is called."""

        if self.env.time_step == 0:
            self.reward_history.append(0)

        else:
            self.reward_history[-1] += sum(self.env.rewards[-1])

        self.loader.value += 1

        return True


def get_loader(**kwargs):
    """Returns a progress bar"""

    kwargs = {
        'value': 0,
        'min': 0,
        'max': 10,
        'description': 'Simulating:',
        'bar_style': '',
        'style': {'bar_color': 'maroon'},
        'orientation': 'horizontal',
        **kwargs
    }
    return IntProgress(**kwargs)

class SACDCallback(BaseCallback):
    def __init__(self, env: CityLearnEnv, loader: IntProgress, weights_vector: List[float] = [1]):
        r"""Initialize CustomCallback.

        Parameters
        ----------
        env: Mapping[str, CityLearnEnv]
            CityLearn environment instance.
        loader: IntProgress
            Progress bar.
        """

        super().__init__(verbose=0)
        self.loader = loader
        self.env = env
        self.reward_history = [0] * len(weights_vector)
        self.weights_vector = weights_vector

    def _on_step(self) -> bool:
        r"""Called each time the env step function is called."""

        if self.env.time_step == 0:
            self.reward_history.append(0)

        else:
            print(self.weights_vector,"weights vector")
            print(self.env.rewards[-1],"reward")
            print(self.reward_history[-1], "reward history")
            self.reward_history[-1] += sum(self.env.rewards[-1])

        self.loader.value += 1

        return True