from pathlib import Path
from time import sleep
from abc import ABC, abstractmethod
from typing import Union, Callable, Any, Type

from flask_socketio import SocketIO
from tqdm import tqdm

from base.datasets_processing import GeneralDataset


class ProgressBar(tqdm):
    def __init__(
            self,
            socket: SocketIO,
            dst,
            *args,
            **kwargs
    ):
        super(ProgressBar, self).__init__(*args, **kwargs)
        self.dst = dst
        self.socket = socket
        self._kwargs = {}

    def _report(
            self,
            obligate: bool = False
    ) -> None:
        if self.socket is not None:
            msg = {}
            msg.update(self._kwargs)
            msg.update({
                "progress": {
                    "text": f'{self.n} of {self.total}',
                    "load": self.n / self.total if self.total > 0 else 1
                }})
            self.socket.send(block=self.dst, msg=msg, tag=self.dst + '_progress', obligate=obligate)

    def reset(
            self,
            total: Union[float, None] = None,
            **kwargs
    ):
        res = super().reset(total=total)
        self._kwargs = kwargs
        self._report(obligate=True)
        return res

    def update(
            self,
            n: int = 1
    ):
        res = super().update(n=n)
        self._report(obligate=True)
        return res


def finalize_decorator(
        func: Callable
) -> Callable:
    def wrapper(
            *args,
            **kwargs
    ) -> Any:
        # Before call
        self: Explainer = args[0]
        self._run_mode = args[1]
        self._run_kwargs = args[2]
        # Function call
        result = func(*args, **kwargs)
        # After call
        self._finalize()
        return result

    return wrapper


class Explainer(
    ABC
):
    """
    Superclass for supported explainers.
    """
    name = 'Explainer'

    @staticmethod
    def check_availability(
            gen_dataset: GeneralDataset,
            model_manager: Type
    ) -> bool:
        """ Availability check for the given dataset and model manager. """
        return False

    def __init__(
            self,
            gen_dataset: GeneralDataset,
            model: Type,
            **kwargs
    ):
        """
        :param gen_dataset: dataset
        :param model: GNN model
        """
        self.gen_dataset = gen_dataset
        self.model = model

        # self.socket = SocketConnect()
        self.pbar = None  # to be set when run from explainer manager
        self.raw_explanation = None  # result of external explainer algorithm
        self.explanation = None  # explanation in our format, ready to be saved
        self._run_kwargs = None  # cache for kwargs from run()
        self._run_mode = None  # cache for mode from run()

    @finalize_decorator
    @abstractmethod
    def run(
            self,
            mode: str,
            kwargs: dict,
            finalize: bool = True
    ):
        """
        Run explanation on a given element (node or graph).
        finalize_decorator handles finalize() call when run() is finished.

        :param mode: 'global' or 'local'
        :param kwargs: run args
        :param finalize: whether to convert raw explanation to our format
        :return:
        """
        pass

    @abstractmethod
    def _finalize(
            self
    ):
        """
        Convert current explanation into inner framework json-able format.

        :return:
        """
        pass

    def save(
            self,
            path: Union[str, Path]
    ) -> None:
        """
        Dump explanation in json format at a given path.

        :param path:
        """
        assert self.explanation is not None
        self.explanation.save(path)


class DummyExplainer(
    Explainer
):
    """ Dummy explainer for debugging
    """
    name = '_Dummy'

    @staticmethod
    def check_availability(
            gen_dataset: GeneralDataset,
            model_manager: Type
    ) -> bool:
        """ Fits for all """
        return True

    def __init__(
            self,
            gen_dataset: GeneralDataset,
            model: Type,
            init_arg=None,
            **kwargs
    ):
        Explainer.__init__(self, gen_dataset, model)
        self.init_arg = init_arg
        self._local_explanation = None
        self._global_explanation = None

    @finalize_decorator
    def run(
            self,
            mode: str,
            kwargs: dict,
            finalize: bool = True
    ) -> None:
        self.pbar.reset(total=10, mode=mode)
        if mode == "local":
            assert self._global_explanation is not None

            idx = kwargs.pop('element_idx')
            run_arg = kwargs.get('local_run_arg')
            if self.gen_dataset.is_multi():
                self._local_explanation = [idx]
            else:
                self._local_explanation = [idx]
            # Get 1st neighbor
            edge_index = self.gen_dataset.data.edge_index
            for i, j in zip(edge_index[0], edge_index[1]):
                if int(i) == idx:
                    self._local_explanation.append(int(j))
        else:
            run_arg = kwargs.get('global_run_arg')
            self._global_explanation = \
                'Global multi' if self.gen_dataset.is_multi() else 'Global single'
            self._global_explanation += f' args={self.init_arg, run_arg}'

        for _ in range(10):
            sleep(0.1)
            self.pbar.update(1)
        self.pbar.close()
        # Remove unpickable attributes
        self.pbar = None

    def _finalize(
            self
    ) -> None:
        mode = self._run_mode
        if mode == "local":
            assert self._global_explanation is not None

            from explainers.explanation import AttributionExplanation
            self.explanation = AttributionExplanation(local=True, nodes="binary")
            self.explanation.add_nodes({ix: 1 for ix in self._local_explanation})
        else:
            data = f"result: {self._global_explanation}"
            from explainers.explanation import Explanation
            self.explanation = Explanation(type='string', local=False, data=data)

        # Remove unpickable attributes
        self.pbar = None
