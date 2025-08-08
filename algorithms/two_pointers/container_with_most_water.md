# Container With Most Water - X-axis and Y-axis Explanation

## Problem Description

Given an integer input array heights representing the heights of vertical lines, write a function that returns the maximum area of water that can be contained by two of the lines (and the x-axis). The function should take in an array of integers and return an integer.

## Where is the X-axis and Y-axis?

### Visual Explanation:

```
Y-axis (height)
↑
|     |
|  |  |     |
|  |  |  |  |
|  |  |  |  |  |
+--+--+--+--+--+--→ X-axis (array indices)
0  1  2  3  4  5
```

## Understanding the Axes:

### X-axis

The **X-axis represents the array indices (positions)** in the heights array.

### Y-axis

The **Y-axis represents the height values** from the heights array.

### Example with `heights = [1, 8, 6, 2, 5, 4, 8, 3, 7]`:

```
Y (height)
9 |
8 |   |           |
7 |   |           |     |
6 |   | |         |     |
5 |   | |   |     |     |
4 |   | |   | |   |     |
3 |   | |   | |   | |   |
2 |   | |   | |   | |   |
1 | | | |   | |   | |   |
0 +─+─+─+─+─+─+─+─+─+─→ X-axis
  0 1 2 3 4 5 6 7 8   (indices)
```

## How the Area Calculation Works:

- **Width** = difference between X-coordinates (indices): `right - left`
- **Height** = minimum of the two line heights: `min(heights[left], heights[right])`
- **Area** = `width × height`

### Example Calculation:

If we choose lines at index 1 and index 8:

- **Width** = `8 - 1 = 7` (X-axis distance)
- **Height** = `min(heights[1], heights[8]) = min(8, 7) = 7` (Y-axis)
- **Area** = `7 × 7 = 49`

## Key Insight

The X-axis is essentially the "ground" or "bottom" of the container, and the distance along it determines how wide your water container is!

## Water Container Visualization:

```
    8 |   |           |
    7 |   |~~~~~~~~~~~|     |
    6 |   |~~~~~~~~~~~|     |
    5 |   |~~~~~~~~~~~|     |
    4 |   |~~~~~~~~~~~|     |
    3 |   |~~~~~~~~~~~|     |
    2 |   |~~~~~~~~~~~|     |
    1 |   |~~~~~~~~~~~|     |
    0 +─+─+─+─+─+─+─+─+─+─→
      0 1 2 3 4 5 6 7 8
```

In this example, the water (~~~) is contained between indices 1 and 8, with width=7 and height=7.
