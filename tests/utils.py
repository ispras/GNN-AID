import shutil
from time import time
from unittest import mock

__patched_dirs = []


def monkey_patch_dirs() -> None:
    """
    Patcher of directories imported in other modules. Useful for tests.
    """
    from aux.utils import GRAPHS_DIR, DATASETS_DIR, MODELS_DIR, EXPLANATIONS_DIR
    dirs = [GRAPHS_DIR, DATASETS_DIR, MODELS_DIR, EXPLANATIONS_DIR]

    modules = ['aux.utils', 'aux.data_info', 'aux.declaration']

    l = locals()
    g = globals()
    for a_dir in dirs:
        # Find a name of the dir variable
        a_name = None
        for name, value in list(locals().items()):
            if value is a_dir:
                a_name = name
                break
        if a_name is None:
            raise RuntimeError(f"Cannot find object for dir '{a_dir}'")

        # Create a tmp version
        tmp_dir = a_dir.parent / (a_dir.name + "__tmp_dir_" + str(time()))
        __patched_dirs.append(tmp_dir)

        # Create patches and run them
        for module in modules:
            case = f"{module}.{a_name}"
            patcher = mock.patch(case, tmp_dir)
            patcher.start()


def cleanup_patches():
    """ Stop patches, remove patched dirs and all their contents. Use after tests are finished.
    """
    # Stop patches
    mock.patch.stopall()

    for a_dir in __patched_dirs:
        if a_dir.exists():
            shutil.rmtree(a_dir)


if __name__ == '__main__':
    from aux.utils import GRAPHS_DIR

    monkey_patch_dirs([GRAPHS_DIR])

    from aux.declaration import GRAPHS_DIR
    print(GRAPHS_DIR)