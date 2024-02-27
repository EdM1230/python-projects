def binary_search(list, target):
    # all the code below are constant runtime O(n)
    first = 0
    last = list[-1]

    while first <= last:
        midpoint = (first + last) // 2 # floor division

        if list[midpoint] == target:
            return midpoint
        elif list[midpoint] < target:
            first = midpoint + 1
        else:
            last = midpoint - 1

    return None

def verify(index):
    if index is not None:
        print(f"Target found at index: {index}")
    print("Target not found in list.")

numbers = [i for i in range(1,11)]

result = binary_search(numbers, 6)
verify(result)