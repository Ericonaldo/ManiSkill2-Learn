# https://github.com/alexmojaki/sorcery
from sorcery import (
    assigned_names,
    call_with_name,
    delegate_to_attr,
    dict_of,
    maybe,
    print_args,
    select_from,
    unpack_attrs,
    unpack_keys,
)

# https://github.com/gruns/icecream
# from icecream import ic


def colored_print(output_string, level="info", logger=print):
    import sys

    from termcolor import colored

    if level.lower() in ["warning", "error"]:
        level = colored(level.upper(), "red")
        output_string = colored(output_string, "cyan")
        logger(f"{level}: {output_string}")
    else:
        logger(output_string)


def empty_print(*args, **kwargs):
    pass


def custom_assert(pause, output_string, logger=None):
    if logger is not None:
        logger = logger.log
    else:
        logger = print
    import sys

    if not pause:
        from termcolor import colored

        file_name = colored(sys._getframe().f_code.co_filename, "red")
        line_number = colored(sys._getframe().f_back.f_lineno, "cyan")
        output_string = colored(output_string, "red")
        logger(f"Assert Error at {file_name}, line {line_number}")
        logger(f"Output: {output_string}")


class SlicePrinter:
    def __getitem__(self, index):
        print(index)


slice_printer = SlicePrinter()
