# pylint: disable=missing-docstring
# $DELETE_BEGIN
VALID_SET = list(range(1, 10))

def valid_rows(grid):
    for row in grid:
        if sorted(row) != VALID_SET:
            return False
    return True

def valid_columns(grid):
    for j in range(0, 9):
        col = []
        for i in range(0, 9):
            col.append(grid[i][j])
        if sorted(col) != VALID_SET:
            return False
    return True

def valid_boxes(grid):
    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            square = []
            for i in range(0, 3):
                i += box_row
                for j in range(0, 3):
                    j += box_col
                    square.append(grid[i][j])
            if sorted(square) != VALID_SET:
                return False
    return True
# $DELETE_END

def sudoku_validator(grid):
    # $CHALLENGIFY_BEGIN
    return valid_rows(grid) and valid_columns(grid) and valid_boxes(grid)
    # $CHALLENGIFY_END
