# Values provided
values = [0.9303, 0.9294, 0.9206, 0.917, 0.9251]

# Calculate mean
n = len(values)
mean = sum(values) / n

# Calculate sample standard deviation
squared_deviations = [(x - mean) ** 2 for x in values]

# Calculate population standard deviation
population_variance = sum(squared_deviations) / n
population_std = population_variance ** 0.5

# Print results
print(f"Values: {values}")
print(f"Mean: {mean:.5f}")
print(f"Population Standard Deviation: {population_std:.5f}")