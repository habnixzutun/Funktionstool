from functools import lru_cache
from sympy import integrate, symbols, diff


# f = "((1 - x**342)/(x**(5 + x) + 1*x))"
f = "2*x**81 + 8*x**(2*x) + 6*x + 8"
x = symbols("x")
print(f)

f1 = f
tmp = ""
while f1 != "0":
    tmp = diff(f1)
    if "x" not in str(tmp):
        del tmp
        break
    f1 = tmp
    print(str(f1))

# print(f"({}|{})")
