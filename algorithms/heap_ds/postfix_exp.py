"""
function read string which will be arithmetic expression composed
of only integers and operators: +, -, *, and /.  The input expression
will be in postfix notation (reverse polish notation  ), an
example (1+2) * 3 would be 1 2 + 3 * in postfix notation.
Your program should determine the answer for the given postfix expression.
For exampla: if sting is 2 12 + 7 / then your program should output 2
Examples:
Input: 1 1 + 1 + 1 +
Output: 4
Input: 2 12 + 7 /
Output: 2
Input: 4 5 + 2 1 + *
Output: 27
"""

from typing import AnyStr, List


def postfix_notation(str_expr: str) -> int:
    """
    Evaluates an arithmetic expression in postfix notation
    Args:
        str_expr: A string representing a postfix expression.

    Returns:
        int: The integer result of the expression
    """
    stack = []
    for t in (tokens := str_expr.split(" ")):
        if t.isdigit() or (t.startswith("-") and t[1:].isdigit()):
            stack.append(int(t))
        else:
            second = stack.pop()
            first = stack.pop()

            if t == "+":
                result = first + second
            elif t == "-":
                result = first - second
            elif t == "*":
                result = first * second
            elif t == "/":
                result = int(first / second)

            stack.append(result)
    return stack.pop()


print(postfix_notation("2 12 + 7 /"))


#############
def postfix_notation2(str_expr: str) -> int:
    """
    Evaluates an arithmetic expression in postfix notation
    Args:
        str_expr: A string representing a postfix expression.

    Returns:
        int: The integer result of the expression
    """
    stack = []
    operators = ('+', '-', '/', '*')
    for t in (tokens := str_expr.split(" ")):
        if t.isdigit() or (t.startswith("-") and t[1:].isdigit()):
            stack.append(int(t))
        else:
            second, first = stack.pop(), stack.pop()
            if t in operators:
                result = int(eval(f"{first}{t}{second}"))
                stack.append(result)
    return stack.pop()


print(postfix_notation2("2 12 + 7 /"))