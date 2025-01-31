import types
import functools

from web_interface.back_front.utils import SocketConnect


class BlockConfig(dict):
    def __init__(
            self
    ):
        super().__init__()

    def init(
            self,
            *args
    ):
        pass

    def modify(
            self,
            **kwargs
    ):
        for key, value in kwargs.items():
            self[key] = value

    def finalize(
            self
    ):
        """ Check correctness """
        # TODO check correctness
        return True

    def toDefaults(
            self
    ):
        """ Set default values """
        self.clear()

    def breik(
            self,
            arg: str = None
    ):
        if arg is None:
            return
        if arg == "full":
            self.clear()
        elif arg == "default":
            self.toDefaults()
        else:
            raise ValueError(f"Unknown argument for breik(): {arg}")


class Block:
    """
        A logical block of a dependency diagram.
    """

    def __init__(self, name, socket: SocketConnect = None):
        # Properties
        self.diagram = None  # Diagram, to be bind after constructor
        self.name = name  # Unique understandable name of block
        self.condition = []  # Boolean function over required blocks
        self.requires = []  # List of Blocks that must be ready to unlock this Block
        self.influences = []  # List of Blocks that are locked until this Block is not ready
        self.socket = socket
        self.tag = 'block'

        # Variables
        self._is_set = False  # Indicator of whether block is defined
        self._config = BlockConfig()  # The config of this Block, result formed from frontend
        self._object = None  # Result of backend request, will be passed to dependent blocks
        self._result = None  # Info to be send to frontend at submit

    def is_set(
            self
    ) -> bool:
        """ Whether block is defined """
        return self._is_set

    def init(
            self,
            *args
    ) -> None:
        """
        Create the default version of config.

        Args:
            *args: the objects of blocks this block depends on
        """
        if self._is_set:
            print(f'Block[{self.name}] is set and cannot be inited ')
            return

        print("Block[" + self.name + "].init()")
        self._config.init(*args)
        init_params = self._init(*args)
        self._send('onInit', init_params)

    def _init(
            self,
            *args
    ) -> None:
        """ Returns jsonable info to be sent to front with onInit()
        """
        # To be overridden in subclass
        raise NotImplementedError

    def modify(
            self,
            **key_values
    ) -> None:
        """ Change some values of the config
        """
        if self._is_set:
            raise RuntimeError(f'Block[{self.name}] is set and cannot be modified!')
        else:
            print(f'Block[{self.name}].modify()')
            self._config.modify(**key_values)
            self._send('onModify')

    def finalize(
            self
    ) -> None:
        """ Check config correctness and make block to be defined """
        if self._is_set:
            print(f'Block[{self.name}] already set')
            return

        print(f'Block[{self.name}].finalize()')
        if self._finalize():
            self._is_set = True

        else:
            raise RuntimeError(f'Block[{self.name}] failed to finalize')

    def _finalize(
            self
    ) -> None:
        """ Checks whether the config is correct to create the object.
        Returns True if OK or False.
        # TODO can we send to front errors to be fixed?
        """
        raise NotImplementedError

    def submit(
            self
    ) -> None:
        """ Run diagram with this block value """
        self.finalize()

        if not self._is_set:
            raise RuntimeError(f'Block[{self.name}] is not set and cannot be submitted')

        print(f'Block[{self.name}].submit()')
        self._submit()
        # self._send('onSubmit')  # FIXME do we want result?
        self._send('onSubmit', self._result)
        if self.diagram:
            self.diagram.on_submit(self)

    def _submit(
            self
    ) -> None:
        """ Perform back request, ect """
        # To be overridden in subclass
        raise NotImplementedError

    def get_object(
            self
    ) -> object:
        """ Get contained backend object
        """
        return self._object

    def unlock(
            self,
            toDefault: bool = False
    ) -> None:
        """ Make block to be undefined
        """
        if self._is_set:
            print(f'Block[{self.name}].unlock()')
            self._is_set = False
            self._send('onUnlock', {"toDefault": toDefault})

            if toDefault:
                self._config.toDefaults()
            else:
                # Remove all values to avoid them staying when modify() will be called again
                self._config.clear()

            if self.diagram:
                self.diagram.on_drop(self)

    def breik(
            self,
            arg: str = None
    ) -> None:
        """ Break block logically
        """
        print(f'Block[{self.name}].break()')
        self.unlock()
        self._config.breik(arg)
        self._send('onBreak')

    def _send(
            self,
            func: str,
            kw_params: dict = None
    ) -> None:
        """ Send signal to frontend listeners. """
        kw_params_str = str(kw_params)
        if len(kw_params_str) > 30:
            kw_params_str = kw_params_str[:30] + f'... [of len={len(kw_params_str)}]'
        print(f'Block[{self.name}]._send(func={func},kw_params={kw_params_str})')
        if self.socket:
            self.socket.send(block=self.name, func=func, msg=kw_params, tag=self.tag)


class WrapperBlock(Block):
    def __init__(
            self,
            blocks: [Block],
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.blocks = blocks
        # Patch submit and unlock functions
        old_submits = {}
        old_unlocks = {}
        for b in self.blocks:
            old_submits[b] = copy_func(b.submit)

            def new_submit(slf):
                # FIXME what if finalize fails?
                # FIXME block submits the same as wrapper submits - do we need it?
                old_submits[slf](slf)
                self.onsubmit(slf)  # NOTE it uses old unlock functions

            b.submit = types.MethodType(new_submit, b)

            old_unlocks[b] = copy_func(b.unlock)

            def new_unlock(
                    slf,
                    *args,
                    **kwargs
            ):
                old_unlocks[slf](slf, *args, **kwargs)
                self.unlock()
            b.unlock = types.MethodType(new_unlock, b)

    def init(
            self,
            *args
    ) -> None:
        super().init(*args)
        for b in self.blocks:
            b.init(*args)

    def breik(
            self,
            arg: str = None
    ) -> None:
        for b in self.blocks:
            b.breik(arg)
        super().breik(arg)

    def onsubmit(
            self,
            block
    ) -> None:
        # # Break all but the given
        # for b in self.blocks:
        #     if b != block:
        #         b.breik(True)

        self._is_set = True
        self._object = block._object
        self._result = block._result
        print(f'Block[{self.name}].submit()')
        self._send('onSubmit',)  # No args to avoid duplication
        if self.diagram:
            self.diagram.on_submit(self)

    def modify(
            self,
            **key_values
    ) -> None:
        # Must not be called
        raise RuntimeError

    def _finalize(
            self
    ) -> None:
        # Must not be called
        raise RuntimeError

    def _submit(
            self
    ) -> None:
        # Must not be called
        raise RuntimeError


def copy_func(
        f
):
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""

    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g
