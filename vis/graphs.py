import json
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size': 6
})

def filter_data(task, threshold=50002):
    filtered_x = [x for x in task["x"] if x <= threshold]
    filtered_y = [task["y"][i] for i, x in enumerate(task["x"]) if x <= threshold]
    return {"name": task["name"], "x": filtered_x, "y": filtered_y, "type": task["type"], "task": task["task"]}

def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def plot_data(ax, task_data, linestyle='-o'):
    for task_name, task_info in task_data.items():
        ax.plot(task_info["x"], task_info["y"], linestyle, label=task_name, color=task_info["color"], linewidth = '0.5',markersize=2)

# Load data from the files
file_paths = [
    'val_imagine2_chamfer_distance _ val_imagine2_chamfer_distance.json',
    'val_imagine1_chamfer_distance_val_imagine1_chamfer_distance.json',
    'val_imagine2_psnr _ val_imagine2_psnr.json',
    'val_imagine1_psnr _ val_imagine1_psnr.json'
]

data_1 = [filter_data(task) for task in load_data(file_paths[0])]
data_2 = [filter_data(task) for task in load_data(file_paths[1])]
data_3 = [filter_data(task) for task in load_data(file_paths[2])]
data_4 = [filter_data(task) for task in load_data(file_paths[3])]

# Create a dictionary to store data for each task
task_data_1 = {}
task_data_2 = {}
task_data_3 = {}
task_data_4 = {}

# Define a list of colors to use for each task
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
task_names = ["PP-BEV-AVG", "PP-BEV-FC", "RV-BEV-FC", "PP-RN-TR", "PP-BEV-TR", "RV-BEV-AVG", "RV-RN-TR", "RV-BEV-TR"]

# Process filtered data
for i, (filtered_data, task_data) in enumerate(zip([data_1, data_2, data_3, data_4], [task_data_1, task_data_2, task_data_3, task_data_4])):
    for j, item in enumerate(filtered_data):
        #task_name = item["task"]
        task_name = task_names[j]
        color = colors[j % len(colors)]
        if task_name not in task_data:
            task_data[task_name] = {"x": [], "y": [], "name": item["name"], "color": color}

        task_data[task_name]["x"].extend(item["x"])
        task_data[task_name]["y"].extend(item["y"])

# Plot the data for each task in subplots
subplot_width = 1.71875  # 6.875 inches divided by 4 subplots

fig, axs = plt.subplots(1, 4, figsize=(6.875, subplot_width), gridspec_kw={'width_ratios': [subplot_width] * 4})
#fig.set_size_inches(w=6.875, h=1.5)

plot_data(axs[0], task_data_1)
plot_data(axs[1], task_data_2)
plot_data(axs[2], task_data_3)
plot_data(axs[3], task_data_4)

# Set titles and labels
axs[0].set_title(f"$\mathcal{{D}}_{{val}}^{{RL}}$")
axs[1].set_title(f"$\mathcal{{D}}_{{val}}^{{DS}}$")
axs[2].set_title(f"$\mathcal{{D}}_{{val}}^{{RL}}$")
axs[3].set_title(f"$\mathcal{{D}}_{{val}}^{{DS}}$")
axs[0].set_ylabel(f"Chamfer Distance (Lidar) $\\downarrow$")
axs[2].set_ylabel(f"PSNR (Camera) $\\uparrow$")

for ax in axs:
    ax.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

# Only show legend in the top-left subplot
axs[0].legend(fontsize=4,fancybox=True)

# Save the plot
fig.tight_layout()
plt.savefig('sensor_fusion.pgf')

