# based on 8 digit display points
ALPHABET_DATA_FILE = {'A': [[0, 0], [0, 2000], [1000, 2000], [1000, 1000], [1000, 0], [0, 0], [0, 1000], [1000, 1000], [1000, 2000]],
                    'B': [[0, 0], [0, 2000], [1000, 2000], [1000, 1000], [0, 1000], [2000, 1000], [2000, 0]],
                    'C': [[0, 0], [1000, 0], [2000, 0], [2000, 1000], [2000, 2000], [1000, 2000], [0, 2000], [1000, 2000], [2000, 2000], [2000, 1000], [2000, 0], [1000, 0], [0, 0]],
                    'D': [[2000, 0], [2000, 1000], [2000, 2000], [1000, 2000], [0, 1000], [1000, 0], [2000, 0]], 
                    'E': [[0, 0], [2000, 0], [2000, 2000], [0, 2000], [2000, 2000], [2000, 1000], [0, 1000], [2000, 1000], [2000, 0], [0, 0]], 
                    'F': [[2000, 0], [2000, 2000], [0, 2000], [2000, 2000], [2000, 1000], [1000, 1000], [2000, 1000], [2000, 0]],
                    'G': [[0, 0], [1000, 0], [2000, 0], [2000, 1000], [2000, 2000], [1000, 2000], [0, 2000], [1000, 2000], [2000, 2000], [2000, 1000], [2000, 0], [1000, 0], [0, 0], [0, 500], [0, 1000], [500, 1000], [1000, 1000], [500, 1000], [0, 1000], [0, 0]],
                    'H': [[0, 0], [0, 1000], [0, 2000], [0, 1000], [0, 1000], [1000, 1000], [2000, 1000], [2000, 1000], [2000, 2000], [2000, 0], [2000, 1000], [1000, 1000], [0, 1000], [0, 1000], [0,0]],
                    'I': [[0, 0], [1000, 0], [2000, 0], [1000, 0], [1000, 500], [1000, 1000], [1000, 1500], [1000, 2000], [2000, 2000], [0, 2000], [1000, 2000], [1000, 1500], [1000, 1000], [1000, 500], [1000, 0], [0, 0]],
                    'J': [[2000, 500], [2000, 0], [1000, 0], [0, 0], [0, 500], [0, 1000], [0, 2000], [0, 2000], [1000, 2000], [0, 2000], [0, 1000], [0, 500], [0, 0], [1000, 0], [2000, 500]],
                    'K': [[2000, 0], [2000, 1000], [2000, 2000], [2000, 1000], [1000, 1500], [0, 2000], [1000, 1500], [2000, 1000], [1000, 500], [0, 0], [1000, 500], [2000, 1000], [2000, 0]],
                    'L': [[0, 0], [1500, 0], [1500, 2000], [1500, 0], [0, 0]],
                    'M': [[0, 0], [0, 1000], [0, 2000], [500, 1500], [1000, 1000], [1500, 1500], [2000, 2000], [2000, 1000], [2000, 0], [2000, 1000], [2000, 2000], [1500, 1500], [1000, 1000], [500, 1500], [0, 2000], [0,0]],
                    'N': [[2000, 0], [2000, 1000], [2000, 2000], [1500, 1500], [1000, 1000], [500, 500], [0, 0], [0, 2000], [0, 0], [500, 500], [1000, 1000], [1500, 1500], [2000, 2000], [2000, 1000], [2000, 0]],
                    'O': [[0, 0], [0, 2000], [0, 2000], [2000, 2000], [2000, 2000], [2000, 0], [2000, 0], [0, 0]],
                    'P': [[2000, 0], [2000, 1000], [2000, 2000], [500, 2000], [0, 1500], [500, 1000], [2000, 1000], [2000, 0]],
                    'Q': [[0, 0], [0, 2000], [0, 2000], [2000, 2000], [2000, 2000], [2000, 0], [2000, 0], [0, 0], [500, 500], [500, 500], [0, 0]] 
                    'R': [[2000, 0], [2000, 1000], [2000, 2000], [500, 2000], [0, 1500], [500, 1000], [0, 0], [500, 1000], [2000, 1000], [2000, 0]], 
                    'S': [[2000, 0], [0, 0], [0, 1000], [2000, 1000], [2000, 2000], [0, 2000], [2000, 2000], [2000, 1000], [0, 1000], [0, 0], [2000, 0]],
                    'T': [[1000, 0], [1000, 1000], [1000, 2000], [2000, 2000], [1000, 2000], [0, 2000], [1000, 2000], [1000, 1000], [1000, 0]]
                    }