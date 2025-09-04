def fibonacci(n):
    seq = [0, 1]
    for i in range(2, n):
        seq.append(seq[-1] + seq[-2])
    return seq[:n]

if __name__ == "__main__":
    n = int(input("请输入要计算的斐波那契数列项数: "))
    result = fibonacci(n)
    print("斐波那契数列前{}项: ".format(n), result)
