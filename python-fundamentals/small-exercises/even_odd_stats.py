user_integer_str = input("Input an integer (0 terminates): ")
user_integer = int(user_integer_str)
even_count = 0
even_sum = 0
odd_count = 0
odd_sum = 0
positive_count = 0

while user_integer != 0:
    if user_integer < 0:
        print("A negative integer was entered.")
        user_integer_str = input("Input an integer (0 terminates): ")
        user_integer = int(user_integer_str)
    else: 
        if user_integer %2 == 0:
            even_count = even_count + 1
            even_sum = even_sum + user_integer
            positive_count = positive_count + 1
            user_integer_str = input("Input an integer (0 terminates): ")
            user_integer = int(user_integer_str)
        else: 
            odd_count = odd_count + 1
            odd_sum = odd_sum + user_integer
            positive_count = positive_count + 1
            user_integer_str = input("Input an integer (0 terminates): ")
            user_integer = int(user_integer_str)

    
print("Sum of odds: ", odd_sum)
print("Sum of evens: ", even_sum)
print("Odd count: ", odd_count)
print("Even count: ", even_count)
print("Total positive int count: ", positive_count)




