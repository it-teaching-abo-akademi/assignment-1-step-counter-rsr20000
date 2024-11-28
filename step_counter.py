import numpy as np  # Importing NumPy for numerical operations
import matplotlib.pyplot as plt  # Importing Matplotlib for plotting graphs
import pandas as pd  # Importing pandas for handling data in tabular format

out = pd.read_csv('out.csv')  # Reading the CSV file 'out.csv' and storing it in the variable 'out'

# Simple function to visualize 4 arrays that are given to it
def visualize_data(timestamps, x_arr, y_arr, z_arr, s_arr):
    # Plotting accelerometer readings
    plt.figure(1)  # Create the first figure
    plt.plot(timestamps, x_arr, color="blue", linewidth=1.0)  # Plot the x-axis accelerometer data
    plt.plot(timestamps, y_arr, color="red", linewidth=1.0)  # Plot the y-axis accelerometer data
    plt.plot(timestamps, z_arr, color="green", linewidth=1.0)  # Plot the z-axis accelerometer data
    plt.show()  # Display the plot

    # Magnitude array calculation
    m_arr = [magnitude(x, y, z) for x, y, z in zip(x_arr, y_arr, z_arr)]  # Compute magnitude of accelerometer vectors
    plt.figure(2)  # Create the second figure

    # Plotting magnitude and steps
    plt.plot(timestamps, s_arr, color="black", linewidth=1.0)  # Plot step visualization
    plt.plot(timestamps, m_arr, color="red", linewidth=1.0)  # Plot the calculated magnitudes
    plt.show()  # Display the plot

# Function to read the data from the log file
def read_data(filename):
    # Reading data from CSV file
    data = np.loadtxt(filename, delimiter=',', skiprows=1)  # Load data delimited by ','
    timestamps = data[:, 0]  # Extract timestamps (the first column)
    x_array = data[:, 1]  # Extract x-axis data (second column)
    y_array = data[:, 2]  # Extract y-axis data (third column)
    z_array = data[:, 3]  # Extract z-axis data (fourth column)
    return timestamps, x_array, y_array, z_array  # Return the extracted arrays

# Function to count steps with fixed threshold
def count_steps(timestamps, x_arr, y_arr, z_arr):
    step_times = []  # Initialize an empty list to store timestamps of detected steps

    # Calculate magnitude of accelerometer data
    magnitudes = [magnitude(x, y, z) for x, y, z in zip(x_arr, y_arr, z_arr)]  # Compute magnitudes

    # Define a fixed threshold for step detection
    threshold = 10  # Value above which steps are considered detected

    # Loop through magnitudes to detect steps
    for i in range(1, len(magnitudes) - 1):
        # Check if the current magnitude exceeds the threshold and is a local peak
        if magnitudes[i] > threshold and magnitudes[i] > magnitudes[i - 1] and magnitudes[i] > magnitudes[i + 1]:
            step_times.append(timestamps[i])  # Append the timestamp of the detected step

    return step_times  # Return the list of step timestamps

# Function to count steps with dynamic threshold
def enhanced_count_steps(timestamps, x_arr, y_arr, z_arr):
    step_times = []  # Initialize an empty list to store timestamps of detected steps

    # Calculate magnitude of accelerometer data
    magnitudes = [magnitude(x, y, z) for x, y, z in zip(x_arr, y_arr, z_arr)]  # Compute magnitudes

    # Define dynamic threshold
    dynamic_threshold = (z_arr.max() + z_arr.min()) / 2  # Calculate a dynamic threshold based on z-axis data
    for i in range(1, len(magnitudes) - 1):
        # Check if the current magnitude exceeds the dynamic threshold and is a local peak
        if magnitudes[i] > dynamic_threshold and magnitudes[i] > magnitudes[i - 1] and magnitudes[i] > magnitudes[i + 1]:
            step_times.append(timestamps[i])  # Append the timestamp of the detected step

    # Plot intervals for visualization
    plot_with_25s_intervals(timestamps, z_arr, dynamic_threshold)  # Plot data with 25-second intervals

    return step_times  # Return the list of step timestamps

# Calculate the magnitude of the given vector
def magnitude(x, y, z):
    return np.linalg.norm([x, y, z])  # Compute the Euclidean norm of the vector (x, y, z)

# Function to convert array of times where steps happened into an array to give into graph visualization
def generate_step_array(timestamps, step_times):
    s_arr = []  # Initialize an empty list for step visualization data
    ctr = 0  # Initialize a counter for step times
    for time in timestamps:
        # Check if the current time is close to the next step time
        if ctr < len(step_times) and step_times[ctr] <= time:
            ctr += 1  # Move to the next step time
            s_arr.append(50000)  # Append a large value for visualization
        else:
            s_arr.append(0)  # Append zero if no step detected at this time
    return s_arr  # Return the step visualization data

# Check that the sizes of arrays match
def check_data(t, x, y, z):
    if len(t) != len(x) or len(y) != len(z) or len(x) != len(y):  # Validate data consistency
        print("Arrays of incorrect length")  # Print error message if lengths mismatch
        return False  # Return False if data is inconsistent
    print(f"The amount of data read from accelerometer is {len(t)} entries")  # Print data size if consistent
    return True  # Return True if data is consistent

def plot_with_25s_intervals(timestamps, z_arr, threshold, interval=25):
    # Convert to numpy arrays for easier processing
    timestamps = np.array(timestamps)
    z_arr = np.array(z_arr)

    # Find the start and end times
    start_time = timestamps[0]
    end_time = timestamps[-1]

    # Initialize arrays to store results
    max_values = []
    min_values = []
    interval_centers = []

    # Loop over each interval
    for t in range(int(start_time), int(end_time), interval):
        indices = (timestamps >= t) & (timestamps < t + interval)  # Filter indices for the current interval

        if any(indices):  # Ensure there are data points in the interval
            max_values.append(np.max(z_arr[indices]))  # Store the max value
            min_values.append(np.min(z_arr[indices]))  # Store the min value
            interval_centers.append(t + interval / 2)  # Store the interval center

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, z_arr, label="Z-axis data", color='black', linewidth=1)

    # Plot threshold, max, and min for each interval
    for i, center in enumerate(interval_centers):
        plt.hlines(y=(max_values[i]+min_values[i])/2, xmin=center - interval/2, xmax=center + interval/2,
                   colors='red', linestyles='--', linewidth=2, label="Threshold" if i == 0 else "") # Plot Threshold of each interval
        plt.hlines(y=max_values[i], xmin=center - interval/2, xmax=center + interval/2,
                   colors='green', linestyles='--', linewidth=2, label="Max" if i == 0 else "") # Plot Max of each interval
        plt.hlines(y=min_values[i], xmin=center - interval/2, xmax=center + interval/2,
                   colors='orange', linestyles='--', linewidth=2, label="Min" if i == 0 else "") # Plot Min of each interval

    # Add labels, title, and legend
    plt.title("Z-axis Data with Threshold, Max, and Min Values (25s Intervals)")
    plt.xlabel("Timestamps")
    plt.ylabel("Z-axis acceleration")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()  # Show the plot

def main():
    # Read data from a measurement file, change the input file name if needed
    timestamps, x_array, y_array, z_array = read_data("out.csv")

    # Check that the data does not produce errors
    if not check_data(timestamps, x_array, y_array, z_array):
        return

    # Count the steps based on array of measurements from accelerometer
    print("\nCounting steps (fix thresholds)")
    st_fix = count_steps(timestamps, x_array, y_array, z_array)

    # Print the result
    print(f"This data contains {len(st_fix)} steps according to current algorithm")

    # Convert array of step times into graph-compatible format
    s_array = generate_step_array(timestamps, st_fix)

    # Visualize data and steps
    visualize_data(timestamps, x_array, y_array, z_array, s_array)

    # Count the steps with dynamic thresholds based on array of measurements from accelerometer
    print("\n---------------------------------------------------------")
    print("\nCounting steps (dynamic thresholds)")
    st_dynamic = enhanced_count_steps(timestamps, x_array, y_array, z_array)

    # Print the result
    print(f"This data contains {len(st_dynamic)} steps according to current algorithm")

    # Convert array of step times into graph-compatible format
    s_array = generate_step_array(timestamps, st_dynamic)

    # Visualize data and steps
    visualize_data(timestamps, x_array, y_array, z_array, s_array)

main()