import sys
import types

class Mock(types.ModuleType):
    """ Returns itself.
    """
    def __init__(self, name):
        super().__init__(name)
        self.__file__ = f"<mocked module {name}>"
        self.__package__ = name.rpartition('.')[0]
        self.__spec__ = None
        self.__path__ = []  # important for Sphinx to see as a package

    def __getattr__(self, name):
        fullname = f"{self.__name__}.{name}"
        if fullname in sys.modules:
            return sys.modules[fullname]
        mock = Mock(fullname)
        sys.modules[fullname] = mock
        setattr(self, name, mock)
        return mock

    def __mro_entries__(self, bases):
        return (self.__class__,)

    def __call__(self, *args, **kwargs):
        return Mock(f"{self.__name__}()")

    def __iter__(self):
        return iter([])


# Add all libs that are used in documented modules and are long to install
MOCK_MODULES = [
    'torch_sparse',
    'torch_scatter',
    'torch_cluster.random_walk',
    'torch_spline_conv',
    'dig',
    'dig.version',
    'dig.xgraph',
    'dig.xgraph.models.utils',
    'dig.xgraph.models.models',
    'dig.xgraph.method',
    'dig.xgraph.method.base_explainer',
    'dig.xgraph.method.shapley',
    'dig.xgraph.method.subgraphx',
    'dgl',
    'mat73',
    'rdkit',
    'shap',
]

for mod in MOCK_MODULES:
    if mod not in sys.modules:
        sys.modules[mod] = Mock(mod)
