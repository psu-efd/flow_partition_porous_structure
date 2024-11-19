import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'

# First set of values (upstream 72 inches)
# values1 = [0.159, 0.15828, 0.14064, 0.17036, 0.1974]    # Experiment 13
# values1 = [0.2058, 0.197, 0.17086, 0.22468, 0.23558]    # Experiment 14
# values1 = [0.260867, 0.200533, 0.174933, 0.211233, 0.202233]    # Experiment 15
values1 = [0.346067, 0.2741, 0.2496, 0.2393, 0.216467]    # Experiment 16
y_positions1 = [0.15, 0.45, 0.75, 1.05, 1.35]  # Adjusted y-positions for 0.3m spacing

# Second set of values (upstream 36 inches)
# values2 = [0.1392, 0.14738, 0.1504, 0.1738, 0.20766]    # Experiment 13
# values2 = [0.18392, 0.18488, 0.18582, 0.22472, 0.25718]    # Experiment 14
# values2 = [0.212133, 0.1949, 0.193467, 0.217833, 0.246067]    # Experiment 15
values2 = [0.2819, 0.266133, 0.253, 0.2675, 0.270133]    # Experiment 16
y_positions2 = y_positions1  # Same y-positions for consistency

# Third set of values (only three values, with the first two removed, obstruction 00 inches)
# values3 = [0.1997, 0.30588, 0.30756]    # Experiment 13
# values3 = [0.2494, 0.38214, 0.39144]    # Experiment 14
# values3 = [0.200967, 0.418833, 0.405733]    # Experiment 15
values3 = [0.244933, 0.538833, 0.4994]    # Experiment 16
y_positions3 = y_positions1[2:]  # Keeping the positions for the remaining three arrows

# Fourth set of values (downstream 36 inches)
# values4 = [0.03402, 0.03252, 0.01858, 0.36968, 0.38428]     # Experiment 13
# values4 = [0.04048, 0.04312, 0.02374, 0.46876, 0.48606]     # Experiment 14
# values4 = [0.0391, 0.042566667, 0.021133, 0.5168, 0.522]     # Experiment 15
values4 = [0.062, 0.063933, 0.028133, 0.671933, 0.672167]     # Experiment 16
y_positions4 = y_positions1  # Same y-positions for consistency

# Fifth set of values (downstream 72 inches)
# values5 = [0.01384, 0.01244, 0.07918, 0.33488, 0.38092]     # Experiment 13
# values5 = [0.0285, 0.02416, 0.09756, 0.42988, 0.48924]     # Experiment 14
# values5 = [0.0326, 0.033467, 0.111133, 0.4688, 0.529433]     # Experiment 15
values5 = [0.0514, 0.04067, 0.128533, 0.6009, 0.6785]     # Experiment 16
y_positions5 = y_positions1  # Same y-positions for consistency

# Distance between each set of arrows at the base in meters
base_distance = 0.9

# Starting position for the first set of arrows
x_start = 0.5

# Create the plot
plt.figure(figsize=(15, 6))

# Plot the first set of arrows, dots, and curve
for x, y, delta_x in zip([x_start]*5, y_positions1, values1):
    plt.arrow(x, y, delta_x, 0, head_width=0.04, head_length=0.02, fc='blue', ec='blue', length_includes_head=True)
    plt.plot(x + delta_x + 0.01, y, 'ko', markersize=4)  # Smaller dot (size 4) to the right of the arrow tip

dot_x_positions1 = [x_start + value + 0.01 for value in values1]
plt.plot(dot_x_positions1, y_positions1, 'k-', linewidth=1)  # Curve connecting the dots for the first set
plt.plot([x_start, x_start], [y_positions1[0], y_positions1[-1]], 'k--', linewidth=1)  # Dotted vertical line for first set

# Plot the second set of arrows, dots, and curve (shifted by 0.9 meters)
for x, y, delta_x in zip([x_start + base_distance]*5, y_positions2, values2):
    plt.arrow(x, y, delta_x, 0, head_width=0.04, head_length=0.02, fc='blue', ec='blue', length_includes_head=True)
    plt.plot(x + delta_x + 0.01, y, 'ko', markersize=4)  # Smaller dot (size 4) to the right of the arrow tip

dot_x_positions2 = [x_start + base_distance + value + 0.01 for value in values2]
plt.plot(dot_x_positions2, y_positions2, 'k-', linewidth=1)  # Curve connecting the dots for the second set
plt.plot([x_start + base_distance, x_start + base_distance], [y_positions2[0], y_positions2[-1]], 'k--', linewidth=1)  # Dotted vertical line for second set

# Plot the third set of arrows, dots, and curve (shifted by an additional 0.9 meters, with only three values)
for x, y, delta_x in zip([x_start + 2 * base_distance]*3, y_positions3, values3):
    plt.arrow(x, y, delta_x, 0, head_width=0.04, head_length=0.02, fc='blue', ec='blue', length_includes_head=True)
    plt.plot(x + delta_x + 0.01, y, 'ko', markersize=4)  # Smaller dot (size 4) to the right of the arrow tip

dot_x_positions3 = [x_start + 2 * base_distance + value + 0.01 for value in values3]
plt.plot(dot_x_positions3, y_positions3, 'k-', linewidth=1)  # Curve connecting the dots for the third set
plt.plot([x_start + 2 * base_distance, x_start + 2 * base_distance], [y_positions3[0], y_positions3[-1]], 'k--', linewidth=1)  # Dotted vertical line for third set

# Plot the fourth set of arrows, dots, and curve (shifted by an additional 0.9 meters)
for x, y, delta_x in zip([x_start + 3 * base_distance]*5, y_positions4, values4):
    plt.arrow(x, y, delta_x, 0, head_width=0.04, head_length=0.02, fc='blue', ec='blue', length_includes_head=True)
    plt.plot(x + delta_x + 0.01, y, 'ko', markersize=4)  # Smaller dot (size 4) to the right of the arrow tip

dot_x_positions4 = [x_start + 3 * base_distance + value + 0.01 for value in values4]
plt.plot(dot_x_positions4, y_positions4, 'k-', linewidth=1)  # Curve connecting the dots for the fourth set
plt.plot([x_start + 3 * base_distance, x_start + 3 * base_distance], [y_positions4[0], y_positions4[-1]], 'k--', linewidth=1)  # Dotted vertical line for fourth set

# Plot the fifth set of arrows, dots, and curve (shifted by an additional 0.9 meters)
for x, y, delta_x in zip([x_start + 4 * base_distance]*5, y_positions5, values5):
    plt.arrow(x, y, delta_x, 0, head_width=0.04, head_length=0.02, fc='blue', ec='blue', length_includes_head=True)
    plt.plot(x + delta_x + 0.01, y, 'ko', markersize=4)  # Smaller dot (size 4) to the right of the arrow tip

dot_x_positions5 = [x_start + 4 * base_distance + value + 0.01 for value in values5]
plt.plot(dot_x_positions5, y_positions5, 'k-', linewidth=1)  # Curve connecting the dots for the fifth set
plt.plot([x_start + 4 * base_distance, x_start + 4 * base_distance], [y_positions5[0], y_positions5[-1]], 'k--', linewidth=1)  # Dotted vertical line for fifth set

# Setting the limits of the plot
plt.ylim(0, max(y_positions1) + 0.15)
plt.xlim(0, 5.0)  # Extend the x-axis to 5.0

# Removing the x-axis and y-axis labels and the grid
plt.xticks([])
plt.yticks([])
plt.grid(False)

# Labeling the x-axis only
plt.xlabel('Measurement Cross Sections', fontsize=18)
plt.ylabel('Flume Width', fontsize=18)

# Adding a title
# plt.title('Velocity Profile for Experiment 13', fontsize=22)
# plt.title('Velocity Profile for Experiment 14', fontsize=22)
# plt.title('Velocity Profile for Experiment 15', fontsize=22)
plt.title('Velocity Profile for Experiment 16', fontsize=22)

# Display the plot
plt.show()
