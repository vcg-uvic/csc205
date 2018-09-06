def fib_sequence(n):
    """Return a list containing the first n terms of the Fibonacci sequence
    (https://en.wikipedia.org/wiki/Fibonacci_number). Assume 0 is the first fibonacci number.

    Examples
    --------
    >>> fib_sequence(5)
   [0, 1, 1, 2, 3]
    >>> fib_sequence(10)
   [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]"""
    result = []
    # TODO put your code here
    return result


def diagonal_entries(matrix):
    """Given a square matrix specified as a list of lists, return a list containing its diagonal entries.
    Examples
    --------
    >>> diagonal_entries([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]])
   [1, 5, 9]"""
    result = []
    # TODO put your code here
    return result


def spiral_entries(matrix):
    """Given a square matrix specified as a list of lists, return a list containing its elements in spiral order.

    Examples
    --------
    >>> spiral_entries([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])
   [1, 2, 3, 6, 9, 8, 7, 4, 5]

    >>> spiral_entries([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9,10,11,12]])
   [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]
   """
    result = []
    # TODO put your code here
    return result


def main():
    # TODO edit code below to test your functions
    print("Testing fib_sequence...")
    y = fib_sequence(5)
    print("fib_sequence(5) = {}".format(y))

    print("Testing diagonal_entries...")
    y = diagonal_entries([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]])
    print("diagonal_entries([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) = {}".format(y))


if __name__ == "__main__":
    main()


