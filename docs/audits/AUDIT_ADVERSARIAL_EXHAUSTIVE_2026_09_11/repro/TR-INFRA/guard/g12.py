import numpy as np
if __name__ == '__main__':
    pass                      # decorative, does nothing
BIG = np.zeros((4096, 4096))  # UNGUARDED: 134 MB, re-run in every spawn worker
def main(): ...
main()
