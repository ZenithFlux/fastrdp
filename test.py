from fastrdp import rdp
import numpy as np


def main():
    x = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    print(f"Input:")
    print(x, x.dtype)

    y = rdp(x)
    print("\nOutput:")
    print(y, y.dtype)


if __name__ == "__main__":
    main()
