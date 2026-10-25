
LOOKAHEAD = 0.07


def heuristic_action(raw, index):
    lowest = None
    for k in range(3):
        if raw[index[f"b{k}_act"]] < 0.5:
            continue
        y = raw[index[f"b{k}_y"]]
        if lowest is None or y > lowest[1]:
            lowest = (raw[index[f"b{k}_x"]], y, raw[index[f"b{k}_vx"]], raw[index[f"b{k}_vy"]])
    if lowest is None:
        return 0, 0

    x, y, vx, vy = lowest
    predicted_x, predicted_y = x + vx * LOOKAHEAD, y + vy * LOOKAHEAD
    action = []
    for side in ("l", "r"):
        pivot_x, pivot_y = raw[index[f"{side}flip_px"]], raw[index[f"{side}flip_py"]]
        close = abs(predicted_x - pivot_x) < 2.9 and predicted_y > pivot_y - 1.2 and y < pivot_y + 0.4
        own_side = predicted_x * pivot_x > -0.6
        falling = vy > -3.0
        action.append(int(close and own_side and falling))
    return action[0], action[1]
