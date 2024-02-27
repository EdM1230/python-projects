def linear_search(list, target):
    """
    Returns the position of the target if found,
    else returns None
    """

    for idx in range(len(list)):
        if list[idx] == target:
            return idx
    return None

def verify(index):
    if index is not None:
        print(f"Target found at index: {index}")
    print("Target not found in list.")

numbers = [i for i in range(1, 11)]

result = linear_search(numbers, 10)
verify(result)