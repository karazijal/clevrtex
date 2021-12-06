import sys
from fire import Fire, parser, decorators
from fire.core import _MakeParseFn
from functools import partial

import inspect

def maybe_handle_self(component):
    if inspect.isclass(component) or inspect.isroutine(component):
        argspec = inspect.getfullargspec(component)
        if 'self' in argspec.args:
            return partial(component, self=object())
    return component


def StrictFire(component, name=None):
    args = sys.argv[1:]
    if args[-1] == '--no-strict':
        args = args[:-1]
        return Fire(component=component, command=args, name=name)

    original_component = component
    remaining_args, flag_args = parser.SeparateFlagArgs(args)
    parsed_flag_args, _ = parser.CreateParser().parse_known_args(flag_args)
    v = parsed_flag_args.trace
    fn = component
    (varargs, kwargs), consumed_args, remaining_args, capacity = _MakeParseFn(fn, decorators.GetMetadata(fn))(remaining_args)
    if v:
        print(component, fn, consumed_args, remaining_args)
    target = remaining_args[0]
    remaining_args = remaining_args[1:]
    fn = maybe_handle_self(getattr(component, target))
    (varargs, kwargs), consumed_args, remaining_args, capacity = _MakeParseFn(fn, decorators.GetMetadata(fn))(remaining_args)
    if v:
        print(component, fn, consumed_args, remaining_args)
    if remaining_args:
        raise ValueError(f"StrictFire: would probably not consume {remaining_args};\nIf this is intended consider reruning with --no-strict at the end")
    return Fire(component=original_component, name=name)


class Foo:
    def __init__(self, a=1):
        self.a = a
    def run(self, b=2):
        print("Still running")
        return self.a + b

if __name__ =='__main__':
    StrictFire(Foo)
