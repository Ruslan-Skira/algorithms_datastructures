import asyncio


async def make_change(target: int) -> str:
    coins = [100, 50, 20, 10, 5, 2, 1]
    answer = []
    for coin in coins:
        while target >= coin:
            target -= coin
            answer.append(str(coin))
    answer_str = " ".join(answer)
    return f"{len(answer)} {answer_str}"


# print(asyncio.run(make_change(24)))
print(asyncio.run(make_change(163)))
print(asyncio.run(make_change(24)))
