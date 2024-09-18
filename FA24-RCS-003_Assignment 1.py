#Task:01
nums = [3, 5, 7, 8, 12]

# Initialize an empty list for the cubes
cubes = []

# Iterate over each number in nums, compute its cube, and append it to cubes
for num in nums:
    cubes.append(num ** 3)

# Print the cubes list
print(cubes)
animal_dict = {}

# Add key-value pairs to the dictionary
animal_dict['parrot'] = 2
animal_dict['goat'] = 4
animal_dict['spider'] = 8
animal_dict['crab'] = 10

# Print the dictionary to verify its contents
print(animal_dict)
animal_dict = {
    'parrot': 2,
    'goat': 4,
    'spider': 8,
    'crab': 10
}

# Initialize a variable to hold the total number of legs
total_legs = 0

# Loop over the dictionary using the items method
for animal, legs in animal_dict.items():
    print(f'{animal}: {legs} legs')
    total_legs += legs

# Print the total number of legs
print(f'Total number of legs: {total_legs}')
# Create the tuple
A = (3, 9, 4, [5, 6])

# Access the list inside the tuple and change the value from 5 to 8
A[3][0] = 8

# Print the modified tuple to verify the change
print(A)
# Create the tuple
A = (3, 9, 4, [8, 6])

# Delete the tuple
del A

# Attempting to print A now will raise a NameError
try:
    print(A)
except NameError as e:
    print(e)  # This will print the error message indicating that A is no longer defined

# Create the tuple
B = ('a', 'p', 'p', 'l', 'e')

# Count the number of occurrences of 'p'
count_p = B.count('p')

# Print the number of occurrences of 'p'
print(count_p)
# Create the tuple
B = ('a', 'p', 'p', 'l', 'e')

# Find the index of 'l'
index_l = B.index('l')

# Print the index of 'l'
print(index_l)

#Task:02
import numpy as np

# Define the matrix A and convert it into a NumPy array
A = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

# Define the vector z as a NumPy array
z = np.array([1, 0, 1])

# Print the matrix A and vector z
print("Matrix A:")
print(A)

print("\nVector z:")
print(z)
import numpy as np

# Define the matrix A as a NumPy array
A = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

# Use slicing to extract the subarray consisting of the first 2 rows and columns 1 and 2
b = A[:2, 1:3]

# Print the subarray b
print("Subarray b:")
print(b)
import numpy as np

# Define the matrix A
A = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

# Create an empty matrix C with the same shape as A
C = np.empty_like(A)

# Print the matrix C
print("Matrix C:")
print(C)
import numpy as np

# Define the matrix A
A = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

# Define the vector z
z = np.array([1, 0, 1])

# Initialize an empty matrix C with the same shape as A
C = np.empty_like(A)

# Add the vector z to each column of A using an explicit loop
for i in range(A.shape[1]):  # Iterate over columns
    C[:, i] = A[:, i] + z

# Print the matrix C
print("Matrix C:")
print(C)

# Create the arrays X, Y, and v
X = np.array([[1, 2], [3, 4]])
Y = np.array([[5, 6], [7, 8]])
v = np.array([9, 10])

# Print the arrays X, Y, and v
print("\nArray X:")
print(X)

print("\nArray Y:")
print(Y)

print("\nArray v:")
print(v)
import numpy as np

# Define the matrices X and Y
X = np.array([[1, 2], [3, 4]])
Y = np.array([[5, 6], [7, 8]])

# Add the matrices
result = X + Y

# Print the result
print("Sum of matrices X and Y:")
print(result)
import numpy as np

# Define the matrices X and Y
X = np.array([[1, 2], [3, 4]])
Y = np.array([[5, 6], [7, 8]])

# Perform matrix multiplication
result = X @ Y  # Alternatively, you can use np.matmul(X, Y)

# Print the result
print("Product of matrices X and Y:")
print(result)
import numpy as np

# Define the matrix Y
Y = np.array([[5, 6], [7, 8]])

# Compute the element-wise square root of matrix Y
sqrt_Y = np.sqrt(Y)

# Print the result
print("Element-wise square root of matrix Y:")
print(sqrt_Y)
import numpy as np

# Define the matrix X and vector v
X = np.array([[1, 2], [3, 4]])
v = np.array([9, 10])

# Compute the dot product of the matrix X and vector v
dot_product = X @ v  # Alternatively, you can use np.dot(X, v)

# Print the result
print("Dot product of matrix X and vector v:")
print(dot_product)
import numpy as np

# Define the matrix X
X = np.array([[1, 2], [3, 4]])

# Compute the sum of each column
column_sums = np.sum(X, axis=0)

# Print the result
print("Sum of each column in matrix X:")
print(column_sums)

#Task:03
def Compute(distance, time):
    """
    Calculate the velocity given distance and time.

    Parameters:
    distance (float): The distance traveled.
    time (float): The time taken to travel the distance.

    Returns:
    float: The velocity, which is distance divided by time.
    """
    # Check if time is zero to avoid division by zero
    if time == 0:
        raise ValueError("Time cannot be zero.")
    
    # Calculate velocity
    velocity = distance / time 
    
    return velocity

# Example usage:
distance = 100  # distance in meters
time = 20       # time in seconds

velocity = Compute(distance, time)
print(f"Velocity: {velocity} meters per second")
# Step 1: Create the list of even numbers up to 12
even_num = [x for x in range(2, 13, 2)]

# Print the list to verify
print("List of even numbers up to 12:", even_num)

# Step 2: Define the function to calculate the product of all entries
def mult(numbers):
    """
    Calculate the product of all entries in the given list.

    Parameters:
    numbers (list): A list of numbers.

    Returns:
    int or float: The product of all numbers in the list.
    """
    product = 1  # Initialize the product to 1
    
    # Loop through each number in the list and multiply
    for number in numbers:
        product *= number
    
    return product

# Use the function to calculate the product of even_num
product_of_evens = mult(even_num)

# Print the result
print("Product of all even numbers up to 12:", product_of_evens)

#Task:04
import pandas as pd

# Create the DataFrame with the specified data
data = {
    'C1': [1, 2, 3, 5, 5],
    'C2': [6, 7, 5, 4, 8],
    'C3': [7, 9, 8, 6, 5],
    'C4': [7, 5, 2, 8, 8]
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Print the first two rows of the DataFrame
print("First two rows of the DataFrame:")
print(df.head(2))
import pandas as pd

# Create the DataFrame with the specified data
data = {
    'C1': [1, 2, 3, 5, 5],
    'C2': [6, 7, 5, 4, 8],
    'C3': [7, 9, 8, 6, 5],
    'C4': [7, 5, 2, 8, 8]
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Print the second column (C2)
print("Second column (C2):")
print(df['C2'])
import pandas as pd

# Create the DataFrame with the specified data
data = {
    'C1': [1, 2, 3, 5, 5],
    'C2': [6, 7, 5, 4, 8],
    'C3': [7, 9, 8, 6, 5],
    'C4': [7, 5, 2, 8, 8]
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Rename the third column from 'C3' to 'B3'
df.rename(columns={'C3': 'B3'}, inplace=True)

# Print the DataFrame to verify the change
print("DataFrame with column 'C3' renamed to 'B3':")
print(df)
import pandas as pd

# Create the DataFrame with the specified data
data = {
    'C1': [1, 2, 3, 5, 5],
    'C2': [6, 7, 5, 4, 8],
    'B3': [7, 9, 8, 6, 5],  # Note the updated column name
    'C4': [7, 5, 2, 8, 8]
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)
print (df)
# Compute the sum of each row
df['Sum'] = df.sum(axis=1)

# Print the DataFrame to verify the addition of the 'Sum' column
print("DataFrame with new 'Sum' column:")
print(df)
import pandas as pd

# Create the DataFrame with the specified data
data = {
    'C1': [1, 2, 3, 5, 5],
    'C2': [6, 7, 5, 4, 8],
    'B3': [7, 9, 8, 6, 5],  # Note the updated column name
    'C4': [7, 5, 2, 8, 8]
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Compute the sum of each row and add it as a new column 'Sum'
df['Sum'] = df.sum(axis=1)

# Print the DataFrame to verify the addition of the 'Sum' column
print("DataFrame with new 'Sum' column:")
print(df)

import csv
#Read csv file(hello_sample.csv)
df = pd.read_csv(r'hello_sample.csv')
#Print complete dataframe
print(df.to_string)

#Print bottom 2 records of dataframe
print(df.tail(2))
#Print information about dataframe
print(df.info())
#Print shape of dataframe
print(df.shape)
#Sort the data of dataframe using column 'Weight'
print(df.sort_values(by='Weight'))
#Use isnull() and dropna() methods of pandas dataframe
print(df.isnull())
print(df.dropna())
