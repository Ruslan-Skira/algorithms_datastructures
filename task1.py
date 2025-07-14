'''algorithm to reverse list.'''

l = [4,2,1,3,5]

def revert_l(arr):
    i_start =0
    i_end = len(arr)-1

    for i in range(len(arr)//2):

        arr[i_start], arr[i_end] = arr[i_end], arr[i_start]
        i_start +=1
        i_end -=1

    return arr

print(revert_l(l))
assert l == [5, 3, 1, 2, 4]
