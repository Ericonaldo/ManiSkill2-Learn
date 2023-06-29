import math
import torch


class ForwardRate:
    def get_rate(self, dims, ts):
        """
        Gets the rate evaluated at times ts (B,)
        Dims is ignored
        """
        raise NotImplementedError


class StateIndependentForwardRate(ForwardRate):
    """
    Generic class representing a state independent forward rate function
    """

    def __init__(self, max_dim):
        self.max_dim = max_dim
        self.max_num_deletions = self.max_dim - 1
        self.std_mult = 0.7
        self.offset = 0.1

        # scaling of the rate function is such that the mean number of deletions
        # is std_mult standard deviations above c so that a good proportion of
        # trajectories will reach the maximum number of deletions during the
        # forward process

        # add a small offset so we never have 0 rate which may have problems with
        # optimization

    def get_rate_integral(self, ts):
        """
        Gets the integral of the rate between time 0 and ts (B,)
        """
        raise NotImplementedError

    def get_dims_at_t(self, start_dims, ts):
        dims_deleted_at_t = torch.poisson(self.get_rate_integral(ts))
        dims_xt = (start_dims - dims_deleted_at_t).clamp(min=1).int()
        return dims_xt

    def get_dims_at_t2_starting_t1(self, dims_t1, t1, t2):
        integral = self.get_rate_integral(t2) - self.get_rate_integral(t1)
        dims_deleted = torch.poisson(integral)
        dims_t2 = (dims_t1 - dims_deleted).clamp(min=1).int()
        return dims_t2


class StepForwardRate(StateIndependentForwardRate):
    def __init__(self, max_dim, rate_cut_t):
        super().__init__(max_dim)
        self.rate_cut_t = rate_cut_t
        assert self.rate_cut_t > 0
        assert self.rate_cut_t < 1

    def get_scalar(self):
        T = self.rate_cut_t  # the step change point
        c = self.max_num_deletions
        return (
            2 * (1 - T) * c
            + self.std_mult ** 2 * (1 - T)
            + math.sqrt(
                (-2 * (1 - T) * c - self.std_mult ** 2 * (1 - T)) ** 2
                - 4 * (1 - T) ** 2 * c ** 2
            )
        ) / (2 * (1 - T) ** 2)

    def get_rate(self, dims, ts):
        T = self.rate_cut_t
        return self.get_scalar() * (ts > T).long() + self.offset

    def get_rate_integral(self, ts):
        T = self.rate_cut_t
        return (ts - T) * self.get_scalar() * (ts > T).long() + self.offset * ts


class ConstForwardRate(StateIndependentForwardRate):
    def __init__(self, max_dim, scalar=None):
        super().__init__(max_dim)
        self.scalar = scalar

    def get_scalar(self):
        try:
            if self.scalar is None:
                c = self.max_num_deletions
                return (
                    2 * c
                    + self.std_mult ** 2
                    + math.sqrt((self.std_mult ** 2 + 2 * c) ** 2 - 4 * c ** 2)
                ) / 2
            else:
                return self.scalar
        except AttributeError:
            print(
                "ConstForwardRate: scalar not set. Presumably because old checkpoint was loaded. Reverting to old method. TODO delete this exception later."
            )
            c = self.max_num_deletions
            return (
                2 * c
                + self.std_mult ** 2
                + math.sqrt((self.std_mult ** 2 + 2 * c) ** 2 - 4 * c ** 2)
            ) / 2

    def get_rate(self, dims, ts):
        return self.get_scalar() * torch.ones_like(ts)

    def get_rate_integral(self, ts):
        return self.get_scalar() * ts


def get_forward_rate(rate_function_name, max_problem_dim, rate_cut_t):
    if rate_function_name == "step":
        return StepForwardRate(max_problem_dim, rate_cut_t)
    elif rate_function_name == "const":
        return ConstForwardRate(max_problem_dim, None)
    else:
        raise ValueError(rate_function_name)
