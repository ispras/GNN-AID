import types
import functools

from web_interface.back_front.utils import SocketConnect


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

    # Whether block is defined
    def is_set(self):
        return self._is_set

    # Get the config
    def get_config(self):
        return self._config.copy()

    def init(self, *args):
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

    def _init(self, *args):
        """ Returns jsonable info to be sent to front with onInit()
        """
        # To be overridden in subclass
        raise NotImplementedError

    # Change some values of the config
    def modify(self, **key_values):
        if self._is_set:
            raise RuntimeError(f'Block[{self.name}] is set and cannot be modified!')
        else:
            print(f'Block[{self.name}].modify()')
            self._config.modify(**key_values)
            self._send('onModify')

    # Check config correctness and make block to be defined
    def finalize(self):
        if self._is_set:
            print(f'Block[{self.name}] already set')
            return

        print(f'Block[{self.name}].finalize()')
        if self._finalize():
            self._is_set = True

        else:
            raise RuntimeError(f'Block[{self.name}] failed to finalize')

    def _finalize(self):
        """ Returns True or False
        # TODO can we send to front errors to be fixed?
        """
        raise NotImplementedError

    # Run diagram with this block value
    def submit(self):
        self.finalize()

        if not self._is_set:
            raise RuntimeError(f'Block[{self.name}] is not set and cannot be submitted')

        print(f'Block[{self.name}].submit()')
        self._submit()
        # self._send('onSubmit')  # FIXME do we want result?
        self._send('onSubmit', self._result)
        if self.diagram:
            self.diagram.on_submit(self)

    # Perform back request, ect
    def _submit(self):
        # To be overridden in subclass
        raise NotImplementedError

    def get_object(self):
        """ Get contained backend object
        """
        return self._object

    def unlock(self, toDefault=False):
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

    def breik(self, arg=None):
        """ Break block logically
        """
        print(f'Block[{self.name}].break()')
        self.unlock()
        self._config.breik(arg)
        self._send('onBreak')

    def _send(self, func, kw_params=None):
        """ Send signal to frontend listeners. """
        kw_params_str = str(kw_params)
        if len(kw_params_str) > 30:
            kw_params_str = kw_params_str[:30] + f'... [of len={len(kw_params_str)}]'
        print(f'Block[{self.name}]._send(func={func},kw_params={kw_params_str})')
        if self.socket:
            self.socket.send(block=self.name, func=func, msg=kw_params, tag=self.tag)


class BlockConfig(dict):
    def __init__(self):
        super().__init__()

    def init(self, *args):
        pass

    def modify(self, **kwargs):
        for key, value in kwargs.items():
            self[key] = value

    # Check correctness
    def finalize(self):
        # TODO check correctness
        return True

    # Set default values
    def toDefaults(self):
        self.clear()

    def breik(self, arg=None):
        if arg is None:
            return
        if arg == "full":
            self.clear()
        elif arg == "default":
            self.toDefaults()
        else:
            raise ValueError(f"Unknown argument for breik(): {arg}")


class WrapperBlock(Block):
    def __init__(self, blocks, *args, **kwargs):
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

            def new_unlock(slf, *args, **kwargs):
                old_unlocks[slf](slf, *args, **kwargs)
                self.unlock()
            b.unlock = types.MethodType(new_unlock, b)

    def init(self, *args):
        super().init(*args)
        for b in self.blocks:
            b.init(*args)

    def breik(self, arg=None):
        for b in self.blocks:
            b.breik(arg)
        super().breik(arg)

    def onsubmit(self, block):
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

    def modify(self, **key_values):
        # Must not be called
        raise RuntimeError

    def _finalize(self):
        # Must not be called
        raise RuntimeError

    def _submit(self):
        # Must not be called
        raise RuntimeError


def copy_func(f):
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""

    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


# if __name__ == '__main__':
#     class A:
#         def __init__(self, x):
#             self.x = x
#
#         def f(self):
#             print(self.x)
#
#     a_list = [A(1), A(2), A(3)]
#
#     def pf(a):
#         print('pf', a.x)
#
#     # Patching
#     for a in a_list:
#         old_f = copy_func(a.f)
#
#         def new_f():
#             pf(a)
#             old_f(a)
#         a.f = new_f
#
#     for a in a_list:
#         a.f()
