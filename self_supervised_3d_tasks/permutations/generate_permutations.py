from pathlib import Path
import numpy as np


if __name__ == "__main__":
    perms = []

    for i in range(100):
        pp = np.random.permutation(27)
        # print(pp)

        for ref in perms:
            if (pp==ref).all():
                print("REGENERATE!!!!")

        perms.append(pp)

    perms = np.stack(perms)

    permutation_path = str(
        Path(__file__).parent / "permutations3d_100_27.npy"
    )

    print(permutation_path)

    with open(permutation_path, 'wb') as f:
        np.save(f, perms)
