# Python code​​​​​​‌‌​‌‌​‌​‌​​​‌‌​​​‌​​‌​​‌‌ below
# You will need this class for your solution. Do not edit it.
import pdb


class Stack:
    def __init__(self):
        self.items = []

    def is_empty(self):
        # return len(self.items) == 0
        return not self.items

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[-1]

    def size(self):
        return len(self.items)

    def __str__(self):
        return str(self.items)

# Complete the function definition for reverse_string.
# Use print("messages...") to debug your solution.

def reverse_string(my_string: str) -> str:
    stack = Stack()
    for char in my_string:
        stack.push(char)
    # Use a list for efficient string concatenation
    reversed_chars = []
    while not stack.is_empty():
        reversed_chars.append(stack.pop())
    return ''.join(reversed_chars)

def test_reverse_string():
    # Test cases
    assert reverse_string("hello") == "olleh"
    assert reverse_string("") == ""
    assert reverse_string("a") == "a"
    assert reverse_string("gninraeL nIdekniL htiw tol a nraeL") == "Learn a lot with LinkedIn Learning"

    print("all tests passed successfully!")

test_reverse_string()
