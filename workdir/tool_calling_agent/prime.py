def get_primes(n=100):
    primes = []
    for num in range(2, n):
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                break
        else:
            primes.append(num)
    return primes

if __name__ == "__main__":
    print(get_primes(100))
