"""
Code taken from https://stackoverflow.com/questions/14533420/can-you-suggest-a-good-minhash-implementation.
"""

from random import randint


class Minhash:
    def __init__(self, dimensions: int):
        self.dimensions = dimensions

    def from_string_array(self, s: list, prime=4294967311):
        """
        Given a list `s`, pass each member of the set through all permutation
        functions, and set the `ith` position of `vec` to the `ith` permutation
        function's output if that output is smaller than `vec[i]`.
        """

        # specify the length of each minhash vector
        max_val = (2**32) - 1

        # create N tuples that will serve as permutation functions
        # these permutation values are used to hash all input sets
        perms = [
            (randint(0, max_val), randint(0, max_val))
            for _ in range(self.dimensions)
        ]

        # initialize a minhash of length N with positive infinity values
        vec = [float("inf") for i in range(self.dimensions)]

        for val in s:
            # ensure s is composed of integers
            if not isinstance(val, int):
                val = hash(val)

            # loop over each "permutation function"
            for perm_idx, perm_vals in enumerate(perms):
                a, b = perm_vals

                # pass `val` through the `ith` permutation function
                output = (a * val + b) % prime

                # conditionally update the `ith` value of vec
                if vec[perm_idx] > output:
                    vec[perm_idx] = output

        # the returned vector represents the minimum hash of the set s
        return vec
