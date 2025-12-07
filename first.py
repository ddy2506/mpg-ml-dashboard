income = float(input("Enter today income: "))
expense = float(input("Enter today expense: "))

profit = income - expense

if profit > 0:
    print("You made profit:", profit)
else:
    print("You had a loss:", profit)
