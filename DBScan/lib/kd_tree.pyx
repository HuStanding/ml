
cdef helper(int n):
    if n == 1 or n ==2:
        return 1
    return helper(n-1) + helper(n-2)


def fibonacci(int n):
    return helper(n)