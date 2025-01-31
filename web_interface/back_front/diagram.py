from typing import Union

from web_interface.back_front.block import WrapperBlock, Block


class Diagram:
    """Diagram of frontend states and transitions between them.
    """

    def __init__(
            self
    ):
        self.blocks = {}  # {name -> Block}

    def get(
            self,
            name: str
    ) -> Block:
        """ Get block by its name """
        return self.blocks[name]

    def add_block(
            self,
            block: Block
    ) -> None:
        if block.name in self.blocks:
            return

        self.blocks[block.name] = block
        block.diagram = self
        if isinstance(block, WrapperBlock):
            for b in block.blocks:
                self.blocks[b.name] = b
                b.diagram = self

    def add_dependency(
            self,
            _from: Union[list, Block],
            to: Union[list, Block],
            condition=all
    ) -> None:
        # assert condition in [all, any]
        if not isinstance(_from, list):
            _from = [_from]

        for b in _from:
            self.add_block(b)
            b.influences.append(to)
            to.requires.append(b)

        self.add_block(to)
        to.condition = condition

    def on_submit(
            self,
            block: Block
    ) -> None:
        """ Init all blocks possible after the block submission
        """
        print('Diagram.onSubmit(' + block.name + ')')
        for b_after in block.influences:
            go = b_after.condition([b_before.is_set() for b_before in b_after.requires])
            if go:
                # IMP add params names as blocks names
                b_after.init(*[x.get_object() for x in b_after.requires if x.is_set()])

    def on_drop(
            self,
            block: Block
    ) -> None:
        """ Recursively break all block that critically depend on the given one
        """
        print('Diagram.onBreak(' + block.name + ')')
        # Check as each depends on all - TODO make custom
        for b in block.influences:
            b.breik()

    def drop(
            self
    ) -> None:
        """ Drop all blocks. """
        # FIXME many blocks will break many times
        for block in self.blocks.values():
            block.breik()
