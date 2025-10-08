from fastrdp import rdp
import numpy as np


def main():
    x = np.array(
        [
            [435, 1577],
            [437, 1577],
            [439, 1577],
            [441, 1577],
            [443, 1577],
            [445, 1577],
            [447, 1577],
            [449, 1578],
            [451, 1578],
            [453, 1578],
            [456, 1578],
            [458, 1578],
            [460, 1578],
            [462, 1578],
            [464, 1578],
            [466, 1578],
            [468, 1578],
            [470, 1578],
            [472, 1578],
            [474, 1578],
            [476, 1578],
        ]
    )
    print(f"Input:")
    print(x, x.dtype)

    y = rdp(x, 21.68)
    print("\nOutput:")
    print(y, y.dtype)

    # Expected output:
    # [[ 435 1577]
    #  [ 476 1578]] int64


if __name__ == "__main__":
    main()
