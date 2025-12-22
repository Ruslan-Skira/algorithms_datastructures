"""function take the string parameter being passed
and reuturn a cpmpressed version of the string using the run-length encoding
algorithm. For example: "wwwggopp" would return 3w2g1o2p.This algorithm works
by taking the occurrence of each character and outputting that number along
with a single character of the repeating sequence. Fo example: "aabbcc" would
return 2a2b2c. The string will not contain any numbers, punctuation,or
symbols. The string will only consist of upper and lowercase letters (a - z).
Examples
Input: "aabbcc"
Output: "2a2b2c"
Input: "wwwggopp"
Output: "3w2g1o2p"
Input: "abc"
Output: "1a1b1c"
"""
def string_traversal(s: str) -> str:
    """
    Compresses a string using the run-length encoding algorithm.
    Args:
        s (str): Input string it to be compressed.
    Returns:
        str: Compressed string.
    """
    if not s:
        return ""
    result = ''
    lef = 0
    right = 0
    while lef < len(s):
        count = 1
        current = s[lef]
        right = lef + 1
        while right < len(s) and current == s[right]:
            count += 1
            right += 1
        result += str(count) + current
        lef = right
    return result

print(string_traversal("aabcc"))
#  time complexity O(n)
# Space complexity O(n)
