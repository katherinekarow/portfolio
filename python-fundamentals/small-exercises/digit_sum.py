input_number = input("Input a number: ")
n = int(input_number)
if n<=0:
    print("Enter a positive integer")
    input_number = input("Input a number: ")
    n = int(input_number)
else:
    def sum_of_digits(n):
        s = 0

        while n:
            s += n % 10
            n //= 10
        if s > 9:
            return sum_of_digits(s)
            
        return s

#n = int(input("Enter an integer: "))
print(sum_of_digits(n))