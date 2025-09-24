primes = []
for num in range(2, 100):
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            break
    else:
        primes.append(num)
print(primes)
