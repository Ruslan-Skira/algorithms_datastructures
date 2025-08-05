# Big O Notation: Why O(n) + O(n) = O(n)

## The Question

Why is **O(n) + O(n) = O(n)** and not O(2n)?

## Big O notation ignores constants

In Big O analysis, we **drop constant factors** because we're interested in how the algorithm scales as input size grows to infinity.

### Mathematical explanation:

- O(n) + O(n) = O(2n)
- But **O(2n) = O(n)** because we drop the constant 2

### Why we drop constants:

```python
# These are all O(n):
def example1(nums):
    for x in nums:          # O(n)
        print(x)

def example2(nums):
    for x in nums:          # O(n)
        print(x)
    for x in nums:          # O(n)
        print(x * 2)
    # Total: O(n) + O(n) = O(2n) = O(n)

def example3(nums):
    for x in nums:          # O(n)
        print(x)
    for x in nums:          # O(n)
        print(x * 2)
    for x in nums:          # O(n)
        print(x * 3)
    # Total: O(3n) = O(n)
```

### Real-world example from two-pointers algorithm:

- While loop: **O(n)** operations
- `result.reverse()`: **O(n)** operations
- Total: **O(n) + O(n) = O(2n) = O(n)**

### When constants DO matter:

Constants are dropped because for large n:

- O(n) vs O(2n): both grow linearly
- O(n) vs O(n²): completely different growth rates

**O(2n)** is still linear growth, so it's classified as **O(n)**.

## Key Insight

Big O describes the **shape of growth**, not the exact number of operations.

### Growth comparison for large values of n:

| n      | O(n)   | O(2n)  | O(n²)       |
| ------ | ------ | ------ | ----------- |
| 100    | 100    | 200    | 10,000      |
| 1,000  | 1,000  | 2,000  | 1,000,000   |
| 10,000 | 10,000 | 20,000 | 100,000,000 |

As you can see, O(n) and O(2n) have the same growth pattern (linear), while O(n²) grows much faster (quadratic).

## Common Big O Rules

1. **Drop constants**: O(2n) → O(n)
2. **Drop lower-order terms**: O(n² + n) → O(n²)
3. **Different inputs use different variables**: O(a + b), not O(n)
