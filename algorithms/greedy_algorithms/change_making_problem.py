import asyncio

denominations = [200, 100, 50, 20, 10, 5, 2, 1]


async def divide_coins(coins, target):
    answer = []
    coin_count = 0

    for coin in coins:
        while target >= coin:
            target -= coin

            answer.append(str(coin))
            coin_count += 1
    answer.append(f"# {coin_count=}")
    return " ".join(answer)


if __name__ == "__main__":
    target = 163
    result = asyncio.run(divide_coins(denominations, target))
    print(result)
