from typing import Iterable


def print_mat(mat: Iterable[Iterable], rjust: bool = True):
    if not mat:
        print("Empty mat")
        return

    max_lengths = [max(len(str(row[i])) for row in mat) for i in range(len(mat[0]))]

    for row in mat:
        for i, cell in enumerate(row):
            if i == len(row) - 1:
                print(str(cell), end="")
                break
            if rjust:
                print(str(cell).rjust(max_lengths[i]), end="  ")
            else:
                print(str(cell).ljust(max_lengths[i]), end="  ")
        print()
