from collections import defaultdict

import torch


class LogInValMixin:
    """Little helper when computing val metrics on training data"""

    def __int__(self):
        self.__log_in_val_dict = defaultdict(list)

    def __process_all_log_in_val(self) -> None:
        for key, l in self.__log_in_val_dict.items():
            if l:
                val = torch.cat([v[0] for v in l], dim=0).mean()
                kwargs = {}
                for _, kw in l:
                    kwargs.update(kw)
                self.log(key, val, **kwargs)
        self._log_in_val_dict = defaultdict(list)

    def log_in_val(self, key, val, **kwargs):
        self.__log_in_val_dict[key].append((val, kwargs))

    def on_validation_epoch_start(self) -> None:
        self.__process_all_log_in_val()
