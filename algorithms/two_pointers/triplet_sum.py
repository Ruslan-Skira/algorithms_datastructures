import asyncio

"""[nums[i], nums[j], nums[k]] such that i, j, and k are distinct indices, and the sum of nums[i], nums[j], and nums[k] equals zero. Ensure that the resulting list does not contain any duplicate triplets."""


async def three_sum(numbers):
    answer = []
    numbers.sort()

    for i in range(len(numbers) - 2):

        if i > 0 and numbers[i] == numbers[i - 1]:
            continue
        l = i + 1
        r = len(numbers) - 1
        while l < r:
            triplet = [numbers[i], numbers[l], numbers[r]]
            triplet_sum = sum(triplet)
            if triplet_sum == 0:
                answer.append(triplet)

                while l < r and numbers[l] == numbers[l + 1]:
                    l += 1
                while l < r and numbers[r] == numbers[r - 1]:
                    r -= 1
                l += 1
                r -= 1
            elif triplet_sum < 0:
                l += 1
            else:
                r -= 1

    return answer


print(asyncio.run(three_sum([-1, 0,0, 1,2, 2, -1, -4])))
