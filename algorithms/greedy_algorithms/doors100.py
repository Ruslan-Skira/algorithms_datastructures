"""
100 doors in a row that are all initially closed
You make 100 passes by the doors
On the first pass, you visit everydoor in swquence and toggle its
state(if the door is closed, you open it; if it is open, you closet it)
The second time, you only visit every second door (door 2, 4, 6,...) and toggle it.
The third time, you visit every third door (door 3, 6, 9, ...)
Continue this pattern until you only visit the 100th door
which door are open and which are closed after the last pass.
"""

doors_status = [False]*101
for i in range(1, 101):
    for j in range(i, 101, i):
        doors_status[j] = not doors_status[j]

print([i for i in range(1, 101 ) if doors_status[i]])