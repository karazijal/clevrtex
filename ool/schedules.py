import numpy as np


class ScheduleParsingError(Exception):
    pass


class Schedule:
    @staticmethod
    def parse_tuple_or_list(s):
        s = s.strip()
        if not (s.startswith('(') and s.endswith(')')) or (s.startswith('[') and s.endswith(']')):
            raise ValueError(f"{s} not a tuple")
        s = s[1:-1].strip()
        return tuple(Schedule.int_or_float(p.strip()) for p in s.split(','))

    @staticmethod
    def int_or_float(v):
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        try:
            return Schedule.parse_tuple_or_list(v)
        except ValueError:
            pass
        return None

    def value(self, step, total_steps):
        raise NotImplementedError()

    @property
    def init(self):
        return self.value(0, 100000000)

    @staticmethod
    def maybe_try(s, cls):
        try:
            if not s.startswith('itr_'):
                return cls(s)
            else:
                return ItrSchedule(cls(s[4:]))
        except ScheduleParsingError:
            return None

    @staticmethod
    def build(s):
        if s is None:
            return None
        s = str(s).strip()
        r = Schedule.maybe_try(s, LinearSchedule) or \
            Schedule.maybe_try(s, SwitchSchedule) or \
            Schedule.maybe_try(s, SetSchedule) or \
            Schedule.maybe_try(s, ConstSchedule)
        if r is None:
            raise ValueError(f"Unknown schedule: {s}")
        return r


class ItrSchedule(Schedule):
    def __init__(self, schedule):
        self.s = schedule
        if isinstance(schedule, ConstSchedule):
            raise ValueError(f"No need to form a Const Schedule for iterations")

    def value(self, step, total_steps):
        return self.s.value(step, total_steps)

    def __str__(self):
        return 'itr_' + str(self.s)


class ConstSchedule(Schedule):
    def __init__(self, s):
        self.v = self.int_or_float(s)
        if self.v is None:
            raise ScheduleParsingError()

    def value(self, step, total_steps):
        return self.v

    def __str__(self):
        return f"const({self.v})"


class SetSchedule(ConstSchedule):
    """Like Const but for strings"""

    def __init__(self, s):
        if not s.startswith('set('):
            raise ScheduleParsingError()
        s = s.lstrip('set(').rstrip(')')
        self.v = s

    def __str__(self):
        return f"set({self.v})"


class LinearSchedule(Schedule):
    def __init__(self, s):
        if not (s.startswith('lin(') or s.startswith('linspace(')):
            raise ScheduleParsingError()
        s = s.lstrip('lin(').lstrip('linspace(').rstrip(')')
        try:
            dur, start, stop = [x.strip() for x in s.split(',')]
            self.start = float(start)
            self.stop = float(stop)
            self.dur = self.int_or_float(dur)
        except ValueError:
            raise ScheduleParsingError()

        if self.dur is None:
            raise ScheduleParsingError()

    def value(self, step, total_steps):
        if self.dur > 0:
            if isinstance(self.dur, int):
                sch = np.linspace(self.start, self.stop, self.dur)
            else:
                sch = np.linspace(self.start, self.stop, int(self.dur * total_steps))
            ind = min(step, len(sch) - 1)
            return sch[ind]
        else:
            return self.start

    def __str__(self):
        return f"lin({self.dur}, {self.start}, {self.stop})"


class SwitchSchedule(Schedule):
    def __init__(self, s):
        if not (s.startswith('to(') or s.startswith('switch(')):
            raise ScheduleParsingError()
        s = s.lstrip('to(').lstrip('switch(').rstrip(')')
        try:
            dur, start, target = [x.strip() for x in s.split(':')]
            self.dur = self.int_or_float(dur)
            self.target = self.build(target)
            self.start = self.build(start)
        except ValueError:
            raise ScheduleParsingError()

        if self.dur is None:
            raise ScheduleParsingError()
        if self.target is None:
            self.target = target
        if self.start is None:
            self.start = start

    def value(self, step, total_steps):
        if self.dur > 0:
            if isinstance(self.dur, int):
                dur = self.dur
            else:
                dur = int(self.dur * total_steps)
            if step > dur:
                return self.target.value(step - dur, total_steps - dur)
            else:
                return self.start.value(step, dur)
        return None

    def __str__(self):
        return f"switch({self.dur}, {self.start} {self.target})"


class ScheduledValue:
    def __init__(self, object, name, schedule):
        self.object = object
        self.name = name
        if not hasattr(self.object, self.name):
            raise ValueError(f"{self.object} does not have attribute {self.name}")
        if not isinstance(schedule, Schedule):
            schedule = Schedule.build(schedule)
        self.schedule = schedule

    @property
    def target(self):
        return getattr(self.object, self.name)

    @target.setter
    def target(self, value):
        setattr(self.object, self.name, value)

    def update(self, current_step, total_steps):
        if self.schedule is None:
            return None
        v = self.schedule.value(current_step, total_steps)
        if v is None:
            return None
        if isinstance(v, (int, float)):
            self.target = v
            return v  # loggable
        elif v != self.target:
            print(f"Setting {self.name} to {v}")
            self.target = v  # not loggable
        return None

    @property
    def is_itr(self):
        return isinstance(self.schedule, ItrSchedule)

    def __str__(self):
        return f"{self.object.__class__.__name__} <{hex(id(self.object))}> {self.name} {self.schedule}"
