# pylint: disable=missing-docstring
# $DELETE_BEGIN
"""Sudoku solver"""
# This implementation is for a 9x9 grid
LEN_GRID = 9
def sudoku_validator(grid):
    """Sudoku validator"""
    valid_set = set(list(range(1, 10)))
    print(valid_set)
    for row in  grid:
        if set(row) != valid_set:
            return False

    for i in range(0, 9):
        l_n = []
        for j in range(0, 9):
            l_n.append(grid[j][i])
        if set(l_n) != valid_set:
            return False

    indexes = list(range(0, 9, 3))
    l_n = []
    for r_r in indexes:
        for c_c in indexes:
            l_n = [item[0 + c_c:3 + c_c] for item in grid[0 + r_r:3 + r_r]]
            concat_l = [j for i in l_n for j in i]
            if set(concat_l) != valid_set:
                return False
    return True

def find_empty(grid):
    """Find next empty position"""
    for i in range(LEN_GRID):
        for j in range(LEN_GRID):
            if grid[i][j] == 0:
                return (i, j)  # row, col
    return None

def valid(grid, num, pos):
    """Check if row, col and box are valid"""
    # Check row
    for i in range(LEN_GRID):
        if grid[pos[0]][i] == num and pos[1] != i:
            return False

    # Check column
    for i in range(LEN_GRID):
        if grid[i][pos[1]] == num and pos[0] != i:
            return False

    # Check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if grid[i][j] == num and (i, j) != pos:
                return False
    return True

def solve(grid):
    """Recursive call to solve until done"""
    find = find_empty(grid)
    if not find:
        return True

    row, col = find
    for i in range(1, 10):
        if valid(grid, i, (row, col)):
            grid[row][col] = i
            if solve(grid):
                return True
            grid[row][col] = 0
    return False
# $DELETE_END
def sudoku_solver(grid):
    """Sudoku solver"""
    # $CHALLENGIFY_BEGIN
    if not isinstance(grid, list):
        return 'invalid grid'

    for row in grid:
        #print(row)
        if len(row) != LEN_GRID:
            return 'invalid grid'

    if len(grid) != LEN_GRID:
        return 'invalid grid'

    solve(grid)
    return grid
    # $CHALLENGIFY_END
