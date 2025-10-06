"""
Fizz Buzz is a game for two or more players
Take it in turns to count aloud from 1 to 100, but each time you are going
to say a multiple of 3, replace it with the word 'fiz'.
For multiples of 5, say 'buzz', and for numbers that are multiples of both 3
and 3 and 5, say 'fizz,buuzz'
"""
for i in range(1, 101):
    if not i%3 and not i%5:
        print('fizz, buzz')
    elif not i%3:
        print('fizz')
    elif not i%5:
        print('buzz')
    else:
        print(i)