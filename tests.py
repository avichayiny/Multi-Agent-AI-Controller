import ex1_check
import search
import utils
import ex1


def main():
    state = {
    "Size":   (3, 3),
    "Walls":  {(0, 1), (2, 1)},
    "Taps":   {(1, 1): 6},
    "Plants": {(2, 0): 1, (0, 2): 0},
    "Robots": {10: (0, 2, 0, 2), 11: (1, 2, 0, 2)},
    }
    p = ex1.create_watering_problem(state) 

    successors = p.successor(p.initial)
    end = p.goal_test(p.initial)
    print(end)

    # 4. הדפסה יפה של התוצאות
    print(f"Found {len(successors)} legal moves:")
    for action, next_state in successors:
        print(f"Action: {action}")


if __name__ == '__main__':
    main()
