def fibonacci(n):
    sequence = []
    a, b = 0, 1
    for _ in range(n):
        sequence.append(a)
        a, b = b, a + b
    return sequence

if __name__ == "__main__":
    n = 10  # 可修改为你想要的项数
    print(f"前{n}项斐波那契数列为：{fibonacci(n)}")
