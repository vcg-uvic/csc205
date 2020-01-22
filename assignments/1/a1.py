# This is a comment

# Simple output
print("Hello World!")

# Variable assignment and usage
x = 5
print(x)

# Conditional execution
if x > 3:
    print("x is bigger than 3")
elif x == 3:
    print("x is equal to 3")
else:
    print("x is smaller than 3")

# Lists
y = [4, 8, 3, 7, 0]
print(y[2])
print(y[2:5])
print(len(y))

# Looping
for i in range(5):
    print(y[i])

# you can also loop through list items like this:
for item in y:
    print(item)

# Some other snippets that might be useful:
#
# input_file = open("input.txt", "r")
# input_file.readlines()
#
# output_file = open("output.txt", "w")
#
# some_list.append(x)
#
# "the value of x is {}. y is {}".format(x, y)
#
# "these are some words seperated by whitespace".split()
#
# int("42") / int("2")