import os
import threading
from math import *
from cmath import *
import warnings

modules = [("click", "from click import clear", "pip3 install click"),
           ("numpy", "import numpy as np", "pip3 install numpy"),
           ("matplotlib", "import matplotlib.pyplot as plt", "pip3 install matplotlib"),
           ("prettytable", "from prettytable import PrettyTable", "pip3 install prettytable"),
           ("sympy", "from sympy import diff, integrate, symbols", "pip3 install sympy")]

for name, imp, ins in modules:
    try:
        exec(imp)
    except ModuleNotFoundError:
        print(f"[+] Installing {name} ({ins})")
        os.system(ins)
        try:
            exec(imp)
        except ModuleNotFoundError:
            print(f"[!] There was a problem installing and/or importing the {name} module. "
                  f"Please check your internet connection")
            print(f"[+] Press enter to close the program")
            input()
            os._exit(-1)
        print()

warnings.filterwarnings("ignore")
import numpy as np
from click import clear
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sympy import diff, integrate, symbols
del os

clear()

abc = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
       'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


cmath_help = """    acos(x, /)
        Return the arc cosine (measured in radians) of x.

        The result is between 0 and pi.

    acosh(x, /)
        Return the inverse hyperbolic cosine of x.

    asin(x, /)
        Return the arc sine (measured in radians) of x.

        The result is between -pi/2 and pi/2.

    asinh(x, /)
        Return the inverse hyperbolic sine of x.

    atan(x, /)
        Return the arc tangent (measured in radians) of x.

        The result is between -pi/2 and pi/2.

    atan2(y, x, /)
        Return the arc tangent (measured in radians) of y/x.

        Unlike atan(y/x), the signs of both x and y are considered.

    atanh(x, /)
        Return the inverse hyperbolic tangent of x.

    ceil(x, /)
        Return the ceiling of x as an Integral.

        This is the smallest integer >= x.

    comb(n, k, /)
        Number of ways to choose k items from n items without repetition and without order.

        Evaluates to n! / (k! * (n - k)!) when k <= n and evaluates
        to zero when k > n.

        Also called the binomial coefficient because it is equivalent
        to the coefficient of k-th term in polynomial expansion of the
        expression (1 + x)**n.

        Raises TypeError if either of the arguments are not integers.
        Raises ValueError if either of the arguments are negative.

    copysign(x, y, /)
        Return a float with the magnitude (absolute value) of x but the sign of y.

        On platforms that support signed zeros, copysign(1.0, -0.0)
        returns -1.0.

    cos(x, /)
        Return the cosine of x (measured in radians).

    cosh(x, /)
        Return the hyperbolic cosine of x.

    degrees(x, /)
        Convert angle x from radians to degrees.
        
    diff(x, /)
        Return the derivative of x

    dist(p, q, /)
        Return the Euclidean distance between two points p and q.

        The points should be specified as sequences (or iterables) of
        coordinates.  Both inputs must have the same dimension.

        Roughly equivalent to:
            sqrt(sum((px - qx) ** 2.0 for px, qx in zip(p, q)))

    erf(x, /)
        Error function at x.

    erfc(x, /)
        Complementary error function at x.

    exp(x, /)
        Return e raised to the power of x.

    expm1(x, /)
        Return exp(x)-1.

        This function avoids the loss of precision involved in the direct evaluation of exp(x)-1 for small x.

    fabs(x, /)
        Return the absolute value of the float x.

    factorial(x, /)
        Find x!.

        Raise a ValueError if x is negative or non-integral.

    floor(x, /)
        Return the floor of x as an Integral.

        This is the largest integer <= x.

    fmod(x, y, /)
        Return fmod(x, y), according to platform C.

        x % y may differ.

    frexp(x, /)
        Return the mantissa and exponent of x, as pair (m, e).

        m is a float and e is an int, such that x = m * 2.**e.
        If x is 0, m and e are both 0.  Else 0.5 <= abs(m) < 1.0.

    fsum(seq, /)
        Return an accurate floating point sum of values in the iterable seq.

        Assumes IEEE-754 floating point arithmetic.

    gamma(x, /)
        Gamma function at x.

    gcd(*integers)
        Greatest Common Divisor.

    hypot(...)
        hypot(*coordinates) -> value

        Multidimensional Euclidean distance from the origin to a point.

        Roughly equivalent to:
            sqrt(sum(x**2 for x in coordinates))

        For a two dimensional point (x, y), gives the hypotenuse
        using the Pythagorean theorem:  sqrt(x*x + y*y).

        For example, the hypotenuse of a 3/4/5 right triangle is:

            >>> hypot(3.0, 4.0)
            5.0

    isclose(a, b, *, rel_tol=1e-09, abs_tol=0.0)
        Determine whether two floating point numbers are close in value.

          rel_tol
            maximum difference for being considered "close", relative to the
            magnitude of the input values
          abs_tol
            maximum difference for being considered "close", regardless of the
            magnitude of the input values

        Return True if a is close in value to b, and False otherwise.

        For the values to be considered close, the difference between them
        must be smaller than at least one of the tolerances.

        -inf, inf and NaN behave similarly to the IEEE 754 Standard.  That
        is, NaN is not close to anything, even itself.  inf and -inf are
        only close to themselves.

    isfinite(x, /)
        Return True if x is neither an infinity nor a NaN, and False otherwise.

    isinf(x, /)
        Return True if x is a positive or negative infinity, and False otherwise.

    isnan(x, /)
        Return True if x is a NaN (not a number), and False otherwise.

    isqrt(n, /)
        Return the integer part of the square root of the input.

    lcm(*integers)
        Least Common Multiple.

    ldexp(x, i, /)
        Return x * (2**i).

        This is essentially the inverse of frexp().

    lgamma(x, /)
        Natural logarithm of absolute value of Gamma function at x.

    log(...)
        log(x, [base=math.e])
        Return the logarithm of x to the given base.

        If the base not specified, returns the natural logarithm (base e) of x.

    log10(x, /)
        Return the base 10 logarithm of x.

    log1p(x, /)
        Return the natural logarithm of 1+x (base e).

        The result is computed in a way which is accurate for x near zero.

    log2(x, /)
        Return the base 2 logarithm of x.

    modf(x, /)
        Return the fractional and integer parts of x.

        Both results carry the sign of x and are floats.

    nextafter(x, y, /)
        Return the next floating-point value after x towards y.

    perm(n, k=None, /)
        Number of ways to choose k items from n items without repetition and with order.

        Evaluates to n! / (n - k)! when k <= n and evaluates
        to zero when k > n.

        If k is not specified or is None, then k defaults to n
        and the function returns n!.

        Raises TypeError if either of the arguments are not integers.
        Raises ValueError if either of the arguments are negative.

    pow(x, y, /)
        Return x**y (x to the power of y).

    prod(iterable, /, *, start=1)
        Calculate the product of all the elements in the input iterable.

        The default start value for the product is 1.

        When the iterable is empty, return the start value.  This function is
        intended specifically for use with numeric values and may reject
        non-numeric types.

    radians(x, /)
        Convert angle x from degrees to radians.

    remainder(x, y, /)
        Difference between x and the closest integer multiple of y.

        Return x - n*y where n*y is the closest integer multiple of y.
        In the case where x is exactly halfway between two multiples of
        y, the nearest even value of n is used. The result is always exact.
    
    root(x, y, /)
        Return the y-th root of x.

    sin(x, /)
        Return the sine of x (measured in radians).

    sinh(x, /)
        Return the hyperbolic sine of x.

    sqrt(x, /)
        Return the square root of x.

    tan(x, /)
        Return the tangent of x (measured in radians).

    tanh(x, /)
        Return the hyperbolic tangent of x.

    trunc(x, /)
        Truncates the Real x to the nearest Integral toward 0.

        Uses the __trunc__ magic method.

    ulp(x, /)
        Return the value of the least significant bit of the float x.

    e = 2.718281828459045
    pi = 3.141592653589793
    tau = 6.283185307179586 (2*pi)
    
    a + b  = add b to a.
    a - b  = subtract b from a.
    a * b  = multiply a with b.
    a / b  = divide a with b.
    a % b  = remainder of a / b.
    a ** b = raise a to the power of b.
    """


class Thread(threading.Thread):
    def __init__(self, xp=tuple([x for x in range(-5, 5)]), yp=tuple([x for x in range(-5, 5)]), function="1*x + 0"):
        threading.Thread.__init__(self)
        self.xpoints = xp
        self.ypoints = yp
        self.function = function

    def run(self):
        self.show(self.xpoints, self.ypoints)

    def show(self, xp=tuple([x for x in range(-5, 5)]), yp=tuple([x for x in range(-5, 5)])):
        try:
            plt.plot(xp, yp)
            plt.grid(color="black", linewidth=0.3)
            plt.show()
        except ValueError:
            Thread(self.xpoints, self.ypoints, self.function).start()
            return -1

    def point(self, x, y):
        plt.scatter(x, y)


def f(x):
    try:
        x = complex_to_num(eval(str(x)))
        return x
    except TypeError:
        return ""


def root(wurzelexponent, radikant):
    return radikant**(1/wurzelexponent)


def quadratic_formula(a, b, c):
    try:
        temp = ((b ** 2) - (4 * a * c)) ** 0.5
        x1 = ((-b) + temp) / (2 * a)
        x2 = ((-b) - temp) / (2 * a)
        return x1, x2
    except ZeroDivisionError:
        return inf, -inf


def fractal_formula(y, a, d, e):
    try:
        y = f(y)
        x1 = sqrt((-4*a*e+4*a*y)/(4*e*e-8*e*y+4*y*y)) + ((2*d*e-2*d*y)/(2*e-2*y))
        x2 = -sqrt((-4*a*e+4*a*y)/(4*e*e-8*e*y+4*y*y)) + ((2*d*e-2*d*y)/(2*e-2*y))
        return x1, x2
    except ZeroDivisionError:
        return inf, -inf


def prepare_f(function):
    function2 = ""
    for i in function:
        if i == "x":
            function2 += "(x)"
        else:
            function2 += i
    return function2


def get_input(var, skip=False, extra=""):
    while True:
        temp = input(var).strip().lower()
        temp = temp.replace(",", ".")
        temp = temp.replace("^", "**")
        if (temp == "") and skip:
            return ""
        elif (temp == "") and (not skip):
            continue
        elif (extra == "calc") and ("x" in temp):
            print("[!] Bitte einen gültigen Term eingeben")
        elif (extra == "square") and ("x" in temp) and (not ("xp" in temp)) and (not ("xt" in temp)):
            print("[!] Extra X-Werte in quadratischen Funktionen sind noch nicht verfügbar.")
            continue
        elif (extra == "square") and ("a" in var.lower()) and (temp == "0"):
            print("[!] Bitte a ≠ 0 eingeben.")
            continue
        elif temp == "help":
            print(cmath_help + "\n")
        elif temp == "e":
            return e
        elif temp == "pi":
            return pi
        elif temp == "tau":
            return tau
        elif ("os." in temp) or ("subprocess." in temp) or ("print" in temp) or ("input" in temp):
            print("[+] Nice try")
            continue
        elif ("os" == temp) or ("subprocess" == temp) or ("print" == temp) or ("input" == temp):
            print("[+] Nice try")
            continue
        else:
            try:
                if f(temp) is None:
                    print("[!] Bitte eine Zahl oder einen gültigen Term eingeben")
                if extra == "float":
                    if isinstance(f(temp), float) or isinstance(f(temp), int):
                        return temp
                    else:
                        print("[!] Bitte eine reelle Zahl eingeben oder Term eingeben")
                        return get_input(var, skip, extra)
                try:
                    temp = complex_to_num(complex(temp))
                    return temp
                except ValueError:
                    pass
                f(temp.lower().replace("x", "1"))
                return f"({temp})"
            except NameError:
                print("[!] Bitte eine Zahl oder einen gültigen Term eingeben")
            except ValueError:
                print("[!] Bitte eine Zahl oder einen gültigen Term eingeben")
            except SyntaxError:
                print("[!] Bitte eine Zahl oder einen gültigen Term eingeben")
            except ZeroDivisionError:
                print("[!] Bitte eine Zahl oder einen gültigen Term eingeben")


def complex_to_num(var):
    if str(var) == "0j" or str(var) == "-0j":
        var = 0
    elif "+0j" in str(var) or "-0j" in str(var) or "+nanj" in str(var) or "-nanj" in str(var):
        var = str(var)
        var = var.replace("+0j", "")
        var = var.replace("-0j", "")
        var = var.replace("+nanj", "")
        var = var.replace("-nanj", "")
        var = var.replace("(", "")
        var = var.replace(")", "")
        var = float(var)
        var = round(var, 5)
    return var


def f2(var):
    if type(var) == str:
        return var[1:-1]
    return var


def get_x_y(functions, extra="", params=(0, 0)):
    t1 = Thread(0, 0)
    t1.start()

    print()
    if extra == "square":
        table = PrettyTable(["", "Scheitelform", "Allgemeine Form", "Ableitung", "Stammfunktion", "Scheitelpunkt", "Schnittpunkte x-Achse", "Schnittpunkt y-Achse"], align="l")
        for char, temp in zip(abc, params):
            form1, form2, ableitung, stammfunktion, sp, sx, sy = temp
            table.add_row([f"{char}(x)", form1, form2, ableitung, stammfunktion, sp, sx, sy])
    else:
        x = symbols("x")
        table = PrettyTable(["", "Funktionsvorschrift", "Ableitung", "Stammfunktion"])
        for char, function in zip(abc, functions):
            table.add_row([f"{char}(x)", function, str(diff(function)), str(integrate(function, x))])
    print(table)
    print()

    all_ypoints = []
    von = f(get_input("[>] Kleinster x Wert?\t", extra="float"))
    bis = f(get_input("[>] Größter x Wert?\t", extra="float"))
    dx = f(get_input("[>] dx?\t\t\t", extra="float"))
    if bis < von:
        von, bis = bis, von
    if dx < 0.000001:
        dx = 0.000001
    if not ((extra == "fractal") and (extra == "fractal2")):
        params = functions
    for function, param in zip(functions, params):
        function = function.replace("x", "(x)")
        xps = [round(x, 6) for x in np.arange(von, bis + dx/10, dx)]
        ypoints = []
        zde = False
        for x in xps:
            try:
                if (extra == "fractal") and (x == float(param[1])):
                    x += dx/10
                if (extra == "fractal2") and (x == float(param[1])):
                    ypoints.append(inf * (param[0] / abs(param[0])))
                    continue
                print(function.replace("x", str(x)), x)
                y = f(function.replace("x", str(x)))
                ypoints.append(round(y, 6))
            except TypeError:
                ypoints.append(None)
                continue
            except ValueError:
                ypoints.append(None)
                continue
            except OverflowError:
                ypoints.append(inf)
                continue
            except ZeroDivisionError:
                zde = True
                ypoints.append(inf)
                continue
        try:
            xps[xps.index(-0)] = 0.0
            ypoints[ypoints.index(-0)] = 0.0
        except ValueError:
            pass
        try:
            if (extra == "fractal") or (extra == "fractal2"):
                ypoints[xps.index(param[1])] = inf * (param[0]/abs(param[0]))
        except ValueError:
            pass

        t1.show(xps, ["" if ypoint == "n.d. in R" else ypoint for ypoint in ypoints])

        ypoints2 = []
        try:
            if extra == "fractal":
                for i in ypoints:
                    ypoints2.append(i)
                ypoints2[ypoints2.index(inf * (param[0]/abs(param[0])))] = "nicht definiert"
            else:
                ypoints2 = ypoints
        except ValueError:
            pass

        ypoints3 = []
        if zde:
            for i in ypoints2:
                ypoints3.append(i)
                if i == inf:
                    ypoints3[-1] = "nicht definiert"
        else:
            ypoints3 = ypoints2
        all_ypoints.append(ypoints3)
        del xps, ypoints, ypoints2, ypoints3

    x = ["x"]
    x.extend([abc[x] + "(x)" for x in range(len(functions))])
    table = PrettyTable(x)

    max = 0
    for i in all_ypoints:
        if len(i) > max:
            max = len(i)
    for i in all_ypoints:
        for j in range(max - len(i)):
            i.append("")

    xps = [round(x, 6) for x in np.arange(von, bis + dx/10, dx)]
    for temp in range(len(xps)):
        temp2 = [xps[temp]]
        temp2.extend([i[temp] for i in all_ypoints])
        table.add_row(temp2)
    return table, t1


def extra_x(functions, t1, extra="", params=(0, 0)):
    while True:
        response = get_input("\n[>] Noch ein spezieller X-Wert? ", skip=True)
        if response != "":
            try:
                x = ["x"]
                x.extend([abc[x] + "(x)" for x in range(len(functions))])
                table = PrettyTable(x)
                ys = []
                x = response
                if f(x) == "":
                    print("[!] Bitte eine Zahl oder einen Term eingeben")
                    continue
                if not (extra == "fractal" or extra == "fractal2"):
                    params = functions
                for function, param, in zip(functions, params):
                    try:
                        if ((extra == "fractal") or (extra == "fractal2")) and (x == f(param[1])):
                            ys.append("n.d.")
                            continue
                        else:
                            y = f(function.replace("x", str(f(x))))
                        if type(x) == float:
                            x = round(x, 5)
                        if type(y) == float:
                            y = round(y, 5)
                        ys.append(y)
                    except ValueError:
                        ys.append("n.d.")
                        continue
                    except ZeroDivisionError:
                        ys.append("n.d.")
                        continue
                temp2 = [f2(x)]
                temp2.extend(ys)
                table.add_row(temp2)
                print(table)
                for y in ys:
                    t1.point(f(x), y)
            except OverflowError:
                print("\n[!] OverflowError: (34, 'Result too large')\n"
                      "[!] Die Ergebnismenge ist zu groß. Wähle einen kleineren Bereich")
            except ZeroDivisionError:
                print("[!] ZeroDivisionError: float division by zero")
        else:
            while True:
                confirm = input("[>] Weiter? (y/n): ").lower()
                if confirm == "y" or confirm == "n":
                    break
            if confirm == "y":
                break


def extra_y(functions=(), t1=None, extra=None, params=(0, 0, 0, 0)):
    while True:
        response = get_input("\n[>] Noch ein spezieller Y-Wert? ", skip=True)
        if response != "":
            try:
                y = ["y"]
                if extra == "root":
                    functions = params
                y.extend([abc[x] + "^-1(x)" for x in range(len(functions))])
                table = PrettyTable(y)
                xs = []
                y = response
                if f(y) == "":
                    print("[!] Bitte eine Zahl oder einen Term eingeben")
                    continue
                if not (extra == "fractal" or extra == "root"):
                    params = functions
                for function, param in zip(functions, params):
                    try:
                        if extra == "root":
                            root, a, d, e = function[0], function[1], function[2], function[3]
                            if (type(f(y)) != complex) and (f(y) - e <= 0) and (root % 2 == 0):
                                xs.append("")
                                continue
                            else:
                                function = f"(-(({e} - y) / {a}))**{root} + {d}"
                        if (extra == "fractal") and (y == param[2]):
                            xs.append("n.d.")
                            continue
                        x = f(function.replace("y", str(y)))
                        if type(x) == float:
                            x = round(x, 5)
                        if type(y) == float:
                            y = round(y, 5)
                        xs.append(x)
                        continue
                    except ValueError:
                        xs.append("n.d.")
                        continue
                    except ZeroDivisionError:
                        xs.append("n.d.")
                        continue
                temp2 = [y]
                temp2.extend(xs)
                table.add_row(temp2)
                print(table)
                for x in xs:
                    if (x != "") and (x != "n.d."):
                        t1.point(f(x), f(y))
            except ValueError:
                break
            except OverflowError:
                print("\n[!] OverflowError: (34, 'Result too large')\n"
                      "[!] Die Ergebnismenge ist zu groß. Wähle einen kleineren Bereich\n\n")
            except ZeroDivisionError:
                print("[!] ZeroDivisionError: float division by zero")
        else:
            while True:
                confirm = input("[>] Weiter? (y/n): ").lower()
                if confirm == "y" or confirm == "n":
                    break
            if confirm == "y":
                break


def steigung(functions=(), extra=None, params=()):
    functions = [str(f1) for f1 in [diff(function) for function in functions]]
    while True:
        response = get_input("\n[>] Steigung an speziellem X-Wert? ", skip=True)
        if response != "":
            try:
                x = ["x"]
                x.extend([abc[x] + "´(x)" for x in range(len(functions))])
                table = PrettyTable(x)
                s = []
                x = response
                if f(x) == "":
                    print("[!] Bitte eine Zahl oder einen Term eingeben")
                    continue
                if not (extra == "root" or extra == "log" or extra == "fractal" or extra == "fractal2"):
                    params = functions
                for function, vars in zip(functions, params):
                    try:
                        if "I" in function:
                            s.append("n.d.")
                        elif (extra == "root" or extra == "log") and (x <= vars[2]):
                            s.append("n.d. in R")
                        elif (extra == "fractal" or extra == "fractal2") and (x == vars[1]):
                            s.append("n.d.")
                        else:
                            s_temp = f(function.replace("x", str(x)))
                            if isinstance(s, float) or isinstance(s, int):
                                s_temp = round(s_temp, 5)
                            s.append(s_temp)
                    except ValueError:
                        s.append("n.d.")
                        continue
                    except ZeroDivisionError:
                        s.append("n.d.")
                        continue
                temp2 = [f2(x)]
                temp2.extend(s)
                table.add_row(temp2)
                print(table)
            except OverflowError:
                print("\n[!] OverflowError: (34, 'Result too large')\n"
                      "[!] Die Ergebnismenge ist zu groß. Wähle einen kleineren Bereich")
            except ZeroDivisionError:
                print("[!] ZeroDivisionError: float division by zero")
        else:
            while True:
                confirm = input("[>] Weiter? (y/n): ").lower()
                if confirm == "y" or confirm == "n":
                    break
            if confirm == "y":
                break


def flaeche(functions):
    while True:
        x = symbols("x")
        response = input("\n[>] Noch eine Fläche zwischen Graph und X-Achse? (y/n): ").strip().lower()
        if response == "y":
            try:
                von = f(get_input("[>] Kleinster x Wert?\t", extra="float"))
                bis = f(get_input("[>] Größter x Wert?\t", extra="float"))
                if bis < von:
                    von, bis = bis, von
                temp = [""]
                temp.extend([abc[x].upper() + f"({bis}) - " + abc[x].upper() + f"({von})" for x in range(len(functions))])
                table = PrettyTable(temp)
                areas = []
                for function in functions:
                    try:
                        area = str(f(integrate(function, (x, von, bis))))
                        areas.append(area)
                        continue
                    except ValueError:
                        areas.append("n.d.")
                        continue
                    except NameError:
                        areas.append("n.d.")
                        continue
                temp2 = [f"{von}<x<{bis}"]
                temp2.extend(areas)
                table.add_row(temp2)
                print(table)
            except ValueError:
                break
            except OverflowError:
                print("\n[!] OverflowError: (34, 'Result too large')\n"
                      "[!] Die Ergebnismenge ist zu groß. Wähle einen kleineren Bereich\n\n")
            except ZeroDivisionError:
                print("[!] ZeroDivisionError: float division by zero")
        elif response == "n":
            break
        else:
            pass


def linear_funktion():
    try:
        functions = []
        i_functions = []
        while len(functions) < len(abc):
            m = get_input("[>] M:\t")
            c = get_input("[>] C:\t")
            if f"({m})*(x)+({c})" not in functions:
                functions.append(f"({m})*(x)+({c})")
                i_functions.append(f"(({c})+(y))/({m})")
            else:
                print("[!] Du hast diese Funktion bereits eingegeben")
            while True:
                confirm = input("[>] Noch eine Funktion? (y/n): ").lower()
                if confirm == "y" or confirm == "n":
                    break
            if confirm == "n":
                break
        table, t1 = get_x_y(functions)
        print(table)
        extra_x(functions, t1)
        extra_y(i_functions, t1)
        steigung(functions)
        flaeche(functions)
    except OverflowError:
        print("\n[!] OverflowError: (34, 'Result too large')\n"
              "[!] Die Ergebnismenge ist zu groß. Wähle einen kleineren Bereich\n\n")
        linear_funktion()
    except ZeroDivisionError:
        print("[!] ZeroDivisionError: float division by zero")


def quadratic_funktion():
    x = symbols("x")
    try:
        functions = []
        vars = []
        results = []
        while len(functions) < len(abc):
            a = get_input("[>] A:\t", extra="square")
            d = get_input("[>] D:\t", extra="square")
            e = get_input("[>] E:\t", extra="square")
            b = -2*f(a)*f(d)
            c = f(a)*(f(d)**2)+f(e)
            if f"({a})*((x)-({d}))**2+({e})" not in functions:
                functions.append(f"({a})*((x)-({d}))**2+({e})")
                vars.append((a, b, c, d, e))
            else:
                print("[!] Du hast diese Funktion bereits eingegeben")
            result = ["", "", "", "", "", "", ""]
            result[0] = f"({a}) * (x - ({d}))^2 + ({e})"
            result[1] = f"({a}) * x^2 + ({b}) * x + ({c})"
            result[2] = str(diff(result[0]))
            result[3] = str(integrate(result[0], x))
            result[4] = f"S({d}|{e})"
            x1, x2 = quadratic_formula(a, b, c)
            if x1 == x2:
                nullstellen = f"{x1}"
            else:
                nullstellen = f"{x1}, {x2}"
            result[5] = nullstellen
            result[6] = f(eval(f"{a}*(0-{d})**2+{e}"))
            results.append(result)
            while True:
                confirm = input("[>] Noch eine Funktion? (y/n): ").lower()
                if confirm == "y" or confirm == "n":
                    break
            if confirm == "n":
                break
        table, t1 = get_x_y(functions, extra="square", params=results)
        print(table)
        extra_x(functions, t1)
        while True:
            response = get_input("\n[>] Noch ein spezieller Y-Wert? ", skip=True)
            if response != "":
                x = ["x"]
                x.extend([abc[x] + "^-1(x)" for x in range(len(functions))])
                table = PrettyTable(x)
                xs = []
                y = response
                if f(y) == "":
                    print("[!] Bitte eine Zahl oder einen Term eingeben")
                    continue
                for a, b, c, d, e in vars:
                    try:
                        if (f(y) == e) and (a == 0):
                            xs.append((inf, -inf))
                        elif f(y) == e:
                            xs.append((d, d))
                        else:
                            c_temp = c - f(y)
                            x1, x2 = quadratic_formula(a, b, c_temp)
                            if type(x1) == float:
                                x1, x2 = round(x1, 5), round(x2, 5)
                            xs.append((x1, x2))
                    except ValueError:
                        xs.append(("n.d.", "n.d."))
                        continue
                temp2 = [y]
                for x in xs:
                    temp = f"x1 = {x[0]}"
                    t1.point(f(x[0]), f(y))
                    if x[0] != x[1]:
                        temp = f"x1 = {x[0]}; x2 = {x[1]}"
                        t1.point(f(x[1]), f(y))
                    temp2.append(temp)
                table.add_row(temp2)
                print(table)
            else:
                while True:
                    confirm = input("[>] Weiter? (y/n): ").lower()
                    if confirm == "y" or confirm == "n":
                        break
                if confirm == "y":
                    break
        steigung(functions)
        flaeche(functions)
    except OverflowError:
        print("\n[!] OverflowError: (34, 'Result too large')\n"
              "[!] Die Ergebnismenge ist zu groß. Wähle einen kleineren Bereich\n\n")
        quadratic_funktion()
    except ZeroDivisionError:
        print("\n[!] ZeroDivisionError: float division by zero\n"
              "[!] Bitte a ≠ 0 eingeben\n\n")
        quadratic_funktion()


def quadratic_funktion2():
    x = symbols("x")
    try:
        functions = []
        vars = []
        results = []
        while len(functions) < len(abc):
            a = get_input("[>] A:\t", extra="square")
            b = get_input("[>] D:\t", extra="square")
            c = get_input("[>] E:\t", extra="square")
            d = f(b) / (2 * f(a))
            e = f(c) - ((f(b) ** 2) / (4 * f(a)))
            if f"({a})*((x)-({d}))**2+({e})" not in functions:
                functions.append(f"({a})*((x)-({d}))**2+({e})")
                vars.append((a, b, c))
            else:
                print("[!] Du hast diese Funktion bereits eingegeben")
            result = ["", "", "", "", "", "", ""]
            result[0] = f"({a}) * (x - ({d}))^2 + ({e})"
            result[1] = f"({a}) * x^2 + ({b}) * x + ({c})"
            result[2] = str(diff(result[0]))
            result[3] = str(integrate(result[0], x))
            result[4] = f"S({d}|{e})"
            x1, x2 = quadratic_formula(a, b, c)
            if x1 == x2:
                nullstellen = f"{x1}"
            else:
                nullstellen = f"{x1}, {x2}"
            result[5] = nullstellen
            result[6] = f(eval(f"{a}*(0-{d})**2+{e}"))
            results.append(result)
            while True:
                confirm = input("[>] Noch eine Funktion? (y/n): ").lower()
                if confirm == "y" or confirm == "n":
                    break
            if confirm == "n":
                break
        table, t1 = get_x_y(functions, extra="square", params=results)
        print(table)
        extra_x(functions, t1)
        while True:
            response = get_input("\n[>] Noch ein spezieller Y-Wert? ", skip=True)
            if response != "":
                x = ["x"]
                x.extend([abc[x] + "^-1(x)" for x in range(len(functions))])
                table = PrettyTable(x)
                xs = []
                y = response
                if f(y) == "":
                    print("[!] Bitte eine Zahl oder einen Term eingeben")
                    continue
                for a, d, e in vars:
                    try:
                        if (f(y) == e) and (a == 0):
                            xs.append((inf, -inf))
                        elif f(y) == e:
                            xs.append((d, d))
                        else:
                            c_temp = c - f(y)
                            x1, x2 = quadratic_formula(a, b, c_temp)
                            if type(x1) == float:
                                x1, x2 = round(x1, 5), round(x2, 5)
                            xs.append((x1, x2))
                    except ValueError:
                        xs.append(("n.d.", "n.d."))
                        continue
                temp2 = [y]
                for x in xs:
                    temp = f"x1 = {x[0]}"
                    t1.point(f(x[0]), f(y))
                    if x[0] != x[1]:
                        temp = f"x1 = {x[0]}; x2 = {x[1]}"
                        t1.point(f(x[1]), f(y))
                    temp2.append(temp)
                table.add_row(temp2)
                print(table)
            else:
                while True:
                    confirm = input("[>] Weiter? (y/n): ").lower()
                    if confirm == "y" or confirm == "n":
                        break
                if confirm == "y":
                    break
        steigung(functions)
        flaeche(functions)
    except OverflowError:
        print("\n[!] OverflowError: (34, 'Result too large')\n"
              "[!] Die Ergebnismenge ist zu groß. Wähle einen kleineren Bereich\n\n")
        quadratic_funktion2()
    except ZeroDivisionError:
        print("\n[!] ZeroDivisionError: float division by zero\n"
              "[!] Bitte a ≠ 0 eingeben\n\n")
        quadratic_funktion2()


def root_funktion():
    try:
        functions = []
        vars = []
        while len(functions) < len(abc):
            root = get_input("[>] Wurzel:\t")
            a = get_input("[>] A:\t\t")
            d = get_input("[>] D:\t\t")
            e = get_input("[>] E:\t\t")
            if f"{a} * (x - {d}) ** (1 / {root}) + {e}" not in functions:
                functions.append(f"{a} * (x - {d}) ** (1 / {root}) + {e}")
                vars.append((root, a, d, e))
            else:
                print("[!] Du hast diese Funktion bereits eingegeben")
            while True:
                confirm = input("[>] Noch eine Funktion? (y/n): ").lower()
                if confirm == "y" or confirm == "n":
                    break
            if confirm == "n":
                break
        table, t1 = get_x_y(functions)
        print(table)
        extra_x(functions, t1)
        extra_y(t1=t1, extra="root", params=vars)
        steigung(functions, extra="root", params=vars)
        flaeche(functions)
    except OverflowError:
        print("\n[!] OverflowError: (34, 'Result too large')\n"
              "[!] Die Ergebnismenge ist zu groß. Wähle einen kleineren Bereich\n\n")
        root_funktion()
    except ZeroDivisionError:
        print("\n[!] ZeroDivisionError: float division by zero\n"
              "[!] Bitte Wurzel ≠ 0 eingeben\n\n")
        root_funktion()


def exponential_funktion():
    try:
        functions = []
        i_functions = []
        while len(functions) < len(abc):
            basis = get_input("[>] Basis:\t")
            a = get_input("[>] A:\t\t")
            d = get_input("[>] D:\t\t")
            e = get_input("[>] E:\t\t")
            if f"({a}) * ({basis})**((x) - ({d})) + ({e})" not in functions:
                functions.append(f"({a}) * ({basis})**((x) - ({d})) + ({e})")
                i_functions.append(f"log(-({e} - y)/{a}, {basis}) + {d}")
            else:
                print("[!] Du hast diese Funktion bereits eingegeben")
            while True:
                confirm = input("[>] Noch eine Funktion? (y/n): ").lower()
                if confirm == "y" or confirm == "n":
                    break
            if confirm == "n":
                break
        table, t1 = get_x_y(functions)
        print(table)
        extra_x(functions, t1)
        extra_y(i_functions, t1)
        steigung(functions)
        flaeche(functions)
    except OverflowError:
        print("\n[!] OverflowError: (34, 'Result too large')\n"
              "[!] Die Ergebnismenge ist zu groß. Wähle einen kleineren Bereich\n\n")
        exponential_funktion()
    except ZeroDivisionError:
        print("\n[!] ZeroDivisionError: float division by zero\n"
              "[!] Bitte a ≠ 0 eingeben\n\n")
        exponential_funktion()


def logarithmic_funktion():
    try:
        functions = []
        i_functions = []
        while len(functions) < len(abc):
            basis = get_input("[>] Basis:\t")
            a = get_input("[>] A:\t\t")
            d = get_input("[>] D:\t\t")
            e = get_input("[>] E:\t\t")
            if f"{a} * log(x - {d}, {basis}) + {e}" not in functions:
                functions.append(f"{a} * log(x - {d}, {basis}) + {e}")
                i_functions.append(f"{basis}**(-({e} - y)/{a}) + {d}")
            else:
                print("[!] Du hast diese Funktion bereits eingegeben")
            while True:
                confirm = input("[>] Noch eine Funktion? (y/n): ").lower()
                if confirm == "y" or confirm == "n":
                    break
            if confirm == "n":
                break
        table, t1 = get_x_y(functions)
        print(table)
        extra_x(functions, t1)
        extra_y(i_functions, t1)
        steigung(functions)
        flaeche(functions)
    except OverflowError:
        print("\n[!] OverflowError: (34, 'Result too large')\n"
              "[!] Die Ergebnismenge ist zu groß. Wähle einen kleineren Bereich\n\n")
        logarithmic_funktion()
    except ZeroDivisionError:
        print("\n[!] ZeroDivisionError: float division by zero\n"
              "[!] Bitte a ≠ 0 eingeben\n\n")
        logarithmic_funktion()


def fractional_funktion():
    try:
        functions = []
        i_functions = []
        vars = []
        while len(functions) < len(abc):
            a = get_input("[>] A:\t\t")
            d = get_input("[>] D:\t\t")
            e = get_input("[>] E:\t\t")
            if f"({a}) * (1/((x) - ({d})))) + ({e})" not in functions:
                functions.append(f"(({a}) * (1/((x) - ({d})))) + ({e})")
                i_functions.append(f"(-({a}) + ({d}) * ({e}) - ({d}) * (y)) / (({e}) - (y))")
                vars.append((a, d, e))
            else:
                print("[!] Du hast diese Funktion bereits eingegeben")
            while True:
                confirm = input("[>] Noch eine Funktion? (y/n): ").lower()
                if confirm == "y" or confirm == "n":
                    break
            if confirm == "n":
                break
        table, t1 = get_x_y(functions, extra="fractal", params=vars)
        print(table)
        extra_x(functions, t1, extra="fractal", params=vars)
        extra_y(i_functions, t1, extra="fractal", params=vars)
        steigung(functions, extra="fractal", params=vars)
        flaeche(functions)
    except OverflowError:
        print("\n[!] OverflowError: (34, 'Result too large')\n"
              "[!] Die Ergebnismenge ist zu groß. Wähle einen kleineren Bereich\n\n")
        fractional_funktion()
    except ZeroDivisionError:
        print("\n[!] ZeroDivisionError: float division by zero\n"
              "[!] Bitte a ≠ 0 eingeben\n\n")
        fractional_funktion()


def fractional_funktion2():
    try:
        functions = []
        vars = []
        while len(functions) < len(abc):
            a = get_input("[>] A:\t\t")
            d = get_input("[>] D:\t\t")
            e = get_input("[>] E:\t\t")
            if f"({a}) * (1/(((x) - ({d}))**2)) + ({e})" not in functions:
                functions.append(f"({a}) * (1/(((x) - ({d}))**2)) + ({e})")
                vars.append((a, d, e))
            else:
                print("[!] Du hast diese Funktion bereits eingegeben")
            while True:
                confirm = input("[>] Noch eine Funktion? (y/n): ").lower()
                if confirm == "y" or confirm == "n":
                    break
            if confirm == "n":
                break
        table, t1 = get_x_y(functions, extra="fractal2", params=vars)
        print(table)
        extra_x(functions, t1, extra="fractal2", params=vars)
        while True:
            response = get_input("\n[>] Noch ein spezieller Y-Wert? ", skip=True)
            if response != "":
                x = ["x"]
                x.extend([abc[x] + "^-1(x)" for x in range(len(functions))])
                table = PrettyTable(x)
                xs = []
                y = response
                if f(y) == "":
                    print("[!] Bitte eine Zahl oder einen Term eingeben")
                    continue
                for a, d, e in vars:
                    try:
                        x1, x2 = fractal_formula(y, a, d, e)
                        x1, x2 = f(x1), f(x2)
                        xs.append((x1, x2))
                    except ValueError:
                        xs.append(("n.d.", "n.d."))
                        continue
                temp2 = ["y"]
                for x in xs:
                    temp = f"x1 = {x[0]}"
                    t1.point(f(x[0]), f(y))
                    if x[0] != x[1]:
                        temp = f"x1 = {x[0]}; x2 = {x[1]}"
                        t1.point(f(x[1]), f(y))
                    temp2.append(temp)
                table.add_row(temp2)
                print(table)
            else:
                while True:
                    confirm = input("[>] Weiter? (y/n): ").lower()
                    if confirm == "y" or confirm == "n":
                        break
                if confirm == "y":
                    break
        steigung(functions, extra="fractal2", params=vars)
        flaeche(functions)
    except OverflowError:
        print("\n[!] OverflowError: (34, 'Result too large')\n"
              "[!] Die Ergebnismenge ist zu groß. Wähle einen kleineren Bereich\n\n")
        fractional_funktion2()
    except ZeroDivisionError:
        print("\n[!] ZeroDivisionError: float division by zero\n"
              "[!] Bitte a ≠ 0 eingeben\n\n")
        fractional_funktion2()


def trigonometric_function(ftype):
    try:
        functions = []
        i_functions = []
        while len(functions) < len(abc):
            a = get_input("[>] A:\t\t")
            d = get_input("[>] D:\t\t")
            e = get_input("[>] E:\t\t")
            if f"({a})*{ftype}((x) - ({d})) + ({e})" not in functions:
                functions.append(f"({a})*{ftype}((x) - ({d})) + ({e})")
                i_functions.append(f"a{ftype}(-((({e}) - (y))/({a}))) + ({d})")
            else:
                print("[!] Du hast diese Funktion bereits eingegeben")
            while True:
                confirm = input("[>] Noch eine Funktion? (y/n): ").lower()
                if confirm == "y" or confirm == "n":
                    break
            if confirm == "n":
                break
        table, t1 = get_x_y(functions)
        print(table)
        extra_x(functions, t1)
        extra_y(i_functions, t1)
        steigung(functions)
        flaeche(functions)
    except OverflowError:
        print("\n[!] OverflowError: (34, 'Result too large')\n"
              "[!] Die Ergebnismenge ist zu groß. Wähle einen kleineren Bereich\n\n")
        trigonometric_function(ftype)
    except ZeroDivisionError:
        print("\n[!] ZeroDivisionError: float division by zero\n"
              "[!] Bitte a ≠ 0 eingeben\n\n")
        trigonometric_function(ftype)


def custom_function():
    print("[+] Bitte alle Multiplikationen als solche kennzeichnen")
    functions = []
    while True:
        while len(functions) < len(abc): 
            funktion = input("[>] Funktion:\t").lower()
            funktion = prepare_f(funktion)
            if funktion.lower() == "help":
                help(cmath_help)
            elif "x" in funktion.lower():
                try:
                    if ("os." in funktion) or ("subprocess." in funktion) or ("print(" in funktion) or ("input(" in funktion):
                        print("[+] Nice try")
                        continue
                    elif ("os" == funktion) or ("subprocess" == funktion) or ("print" == funktion) or ("input" == funktion):
                        print("[+] Nice try")
                        continue
                    valid = False
                    for x in range(-10000, 10000, 100):
                        try:
                            f(funktion.lower().replace("x", f"({x})"))
                            valid = True
                            break
                        except ValueError:
                            pass
                    if not valid:
                        print("[!] Bitte gültige Funktion eingeben")
                        continue
                    break
                except NameError or SyntaxError:
                    print("[!] Bitte gültige Funktion eingeben")
        if funktion not in functions:
            functions.append(funktion)
        else:
            print("[!] Du hast diese Funktion bereits eingegeben")
        while True:
            confirm = input("[>] Noch eine Funktion? (y/n): ").lower()
            if confirm == "y" or confirm == "n":
                break
        if confirm == "n":
            break
    table, t1 = get_x_y(functions)
    print(table)
    extra_x(functions, t1)
    steigung(functions)
    flaeche(functions)


def schnittpunkt_lineare_gleichung():
    try:
        m1 = get_input("[>] M1:\t")
        c1 = get_input("[>] C1:\t")
        m2 = get_input("[>] M2:\t")
        c2 = get_input("[>] C2:\t")
        if (m1 == m2) and (c1 == c2):
            print("\n[+] Die Geraden sind identisch")
        elif (m1 == m2) and (c1 != c2):
            print("\n[+] Die Geraden sind parallel")
        else:
            schnittpunkt_x = (c2 - c1) / (m1 - m2)
            schnittpunkt_y = m1 * schnittpunkt_x + c1
            print("\n[+] Schnittpunkt:\t({}|{})".format(schnittpunkt_x, schnittpunkt_y))
    except OverflowError:
        print("\n[!] OverflowError: (34, 'Result too large')")
    except ZeroDivisionError:
        print("\n[!] Die Geraden sind parallel")


def schnittpunkt_quadratische_gleichung1():
    a1 = get_input("[>] A1:\t", extra="float")
    d1 = get_input("[>] D1:\t", extra="float")
    e1 = get_input("[>] E1:\t", extra="float")
    a2 = get_input("[>] A2:\t", extra="float")
    d2 = get_input("[>] D2:\t", extra="float")
    e2 = get_input("[>] E2:\t", extra="float")
    b1 = -2 * a1 * d1
    c1 = a1 * (d1 ** 2) + e1
    b2 = -2 * a2 * d2
    c2 = a2 * (d2 ** 2) + e2
    try:
        if (a1 == a2) and (b1 == b2) and (c1 == c2):
            print("\n[!] Die Parabeln sind identisch")
        else:
            a = a2 - a1
            b = b2 - b1
            c = c2 - c1
            x1, x2 = quadratic_formula(a, b, c)
            x1 = round(float(x1), 5)
            x2 = round(float(x2), 5)
            y = a1 * x1 ** 2 + b1 * x1 + c1
            if (x1 == inf) and (x2 == -inf):
                print("\n[!] Die Parabeln schneiden sich nicht")
                return
            if x1 == x2:
                print(f"\n[+] Schnittpunkt:\t({x1}|{y})")
            else:
                print(f"\n[+] Schnittpunkte:\t({x1}|{y}), ({x2}|{y})")
    except TypeError or ZeroDivisionError:
        print("\n[!] Die Parabeln schneiden sich nicht")


def schnittpunkt_quadratische_gleichung2():
    a1 = get_input("[>] A1:\t", extra="float")
    b1 = get_input("[>] B1:\t", extra="float")
    c1 = get_input("[>] C1:\t", extra="float")
    a2 = get_input("[>] A2:\t", extra="float")
    b2 = get_input("[>] B2:\t", extra="float")
    c2 = get_input("[>] C2:\t", extra="float")
    try:
        if (a1 == a2) and (b1 == b2) and (c1 == c2):
            print("\n[!] Die Parabeln sind identisch")
        else:
            a = a2 - a1
            b = b2 - b1
            c = c2 - c1
            x1, x2 = quadratic_formula(a, b, c)
            x1 = round(float(x1), 5)
            x2 = round(float(x2), 5)
            y = a1 * x1 ** 2 + b1 * x1 + c1
            if x1 == x2:
                print(f"\n[+] Schnittpunkt:\t({x1}|{y})")
            else:
                print(f"\n[+] Schnittpunkte:\t({x1}|{y}), ({x2}|{y})")
    except TypeError or ZeroDivisionError:
        print("\n[!] Die Parabeln schneiden sich nicht")


def streckfaktor_bestimmen():
    try:
        xs = get_input("[>] X von Scheitelpunkt:\t", extra="float")
        ys = get_input("[>] Y von Scheitelpunkt:\t", extra="float")
        xp = get_input("[>] X von Punkt 2:\t\t", extra="float")
        yp = get_input("[>] Y von Punkt 2:\t\t", extra="float")
        a = (yp - ys) / ((xp - xs) ** 2)
        d = -xs
        e = ys
        if (d >= 0) and (e >= 0):
            result = "\n[+] Scheitelform:\t\t{}(x + {})^2 + {}".format(a, d, e)
        elif e >= 0 >= d:
            result = "\n[+] Scheitelform:\t\t{}(x {})^2 + {}".format(a, d, e)
        elif e <= 0 <= d:
            result = "\n[+] Scheitelform:\t\t{}(x + {})^2 {}".format(a, d, e)
        else:
            result = "\n[+] Scheitelform:\t\t{}(x {})^2 {}".format(a, d, e)
        print(result)
        b = -2 * a * (-d)
        c = a * ((-d) ** 2) + e
        if b > 0 and c > 0:
            print("[+] Allgemeine Form:\t{}x^2 + {}x + {}".format(a, b, c))
        elif c < 0 < b:
            print("[+] Allgemeine Form:\t{}x^2 + {}x {}".format(a, b, c))
        elif c > 0 > b:
            print("[+] Allgemeine Form:\t{}x^2 {}x + {}".format(a, b, c))
        else:
            print("[+] Allgemeine Form:\t{}x^2 {}x {}".format(a, b, c))
        print("[+] Scheitelpunkt:\t\t({}|{})".format((-d) * 1, e))
        if (e > 0) and (a > 0) or (e < 0) and (a < 0):
            x1, x2 = quadratic_formula(a, b, c)
            if 0 >= e:
                if x1 == x2:
                    nullstellen = f"[+] Nullstelle:\t({x1}|0)"
                else:
                    nullstellen = f"[+] Nullstellen:\t({x1}|0), ({x2}|0)"
                print(nullstellen)
    except OverflowError:
        print("[!] OverflowError: (34, 'Result too large')")
    except ZeroDivisionError:
        print("[!] ZeroDivisionError: float division by zero")


def calculator():
    while True:
        term = get_input("[>] Term:\t", extra="calc")
        print(f(term))
        while True:
            confirm = input("[>] Zurück zur Übersicht? (y/n): ").lower()
            if confirm == "y" or confirm == "n":
                break
        if confirm == "n":
            break


def main():
    while True:
        print("Funktionen:\n"
              "[1]  f(x) = m * x + c\n"
              "[2]  f(x) = a * (x - d)^2 + e\n"
              "[3]  f(x) = a * x^2 + b * x + c\n"
              "[4]  f(x) = a * √(x - d) + e\n"
              "[5]  f(x) = a * basis^(x - d) + e\n"
              "[6]  f(x) = a * log basis(x - d) + e\n"
              "[7]  f(x) = a * 1/(x - d) + e\n"
              "[8]  f(x) = a * 1/((x - d)^2) + e\n"
              "[9]  f(x) = a * sin(x - d) + e\n"
              "[10] f(x) = a * cos(x - d) + e\n"
              "[11] f(x) = a * tan(x - d) + e\n"
              "[12] Sonstiges\n"
              "\n"
              "Schnittpunkte:\n"
              "[13] Schnittpunkt Geraden\n"
              "[14] Schnittpunkte Parabeln (aus Scheitelform)\n"
              "[15] Schnittpunkte Parabeln (aus allgemeiner Form)\n"
              "\n"
              "Andere:\n"
              "[16] Parabelgleichung Parabel aus 2 Punkten\n"
              "[17] Normaler Rechner\n"
              "\n[exit] Programm verlassen\n")
        funktionstyp = input("[>] Option: ").strip().lower()
        if funktionstyp == "exit":
            while True:
                confirm = input("[>] Verlassen bestätigen (y/n): ").strip().lower()
                if confirm == "y":
                    import os
                    os._exit(0)
                elif confirm == "n":
                    clear()
                    break
        print("")
        if funktionstyp == "1":
            linear_funktion()
        elif funktionstyp == "2":
            quadratic_funktion()
        elif funktionstyp == "3":
            quadratic_funktion2()
        elif funktionstyp == "4":
            root_funktion()
        elif funktionstyp == "5":
            exponential_funktion()
        elif funktionstyp == "6":
            logarithmic_funktion()
        elif funktionstyp == "7":
            fractional_funktion()
        elif funktionstyp == "8":
            fractional_funktion2()
        elif funktionstyp == "9":
            trigonometric_function("sin")
        elif funktionstyp == "10":
            trigonometric_function("cos")
        elif funktionstyp == "11":
            trigonometric_function("tan")
        elif funktionstyp == "12":
            custom_function()
        elif funktionstyp == "13":
            schnittpunkt_lineare_gleichung()
            input()
        elif funktionstyp == "14":
            schnittpunkt_quadratische_gleichung1()
            input()
        elif funktionstyp == "15":
            schnittpunkt_quadratische_gleichung2()
            input()
        elif funktionstyp == "16":
            streckfaktor_bestimmen()
            input()
        if funktionstyp == "17":
            calculator()
        clear()


if __name__ == "__main__":
    main()
