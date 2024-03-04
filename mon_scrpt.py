#!/usr/bin/env python

import sys

def addition(a, b):
    return a + b

def soustraction(a, b):
    return a - b

def multiplication(a, b):
    return a * b

def division(a, b):
    if b == 0:
        raise ValueError("Division par zéro !")
    return a / b

def puissance(a, b):
    return a ** b

def calculer_operation(operation, a, b):
    if operation == "addition":
        return addition(a, b)
    elif operation == "soustraction":
        return soustraction(a, b)
    elif operation == "multiplication":
        return multiplication(a, b)
    elif operation == "division":
        return division(a, b)
    elif operation == "puissance":
        return puissance(a, b)
    else:
        raise ValueError("Opération non valide")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: mon_script.py operation a b")
        sys.exit(1)

    operation = sys.argv[1]
    a = float(sys.argv[2])
    b = float(sys.argv[3])

    try:
        resultat = calculer_operation(operation, a, b)
        print(f"Le résultat de {operation}({a}, {b}) est {resultat}")
    except ValueError as e:
        print(f"Erreur: {e}")
        sys.exit(1)
