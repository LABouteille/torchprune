from collections import OrderedDict

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import Layout, interact

# Format ...
# parameter,sparsity,loss,top1_acc
# conv1,0.0,0.7166622844696044,0.7874999642372131
# conv1,0.05,0.7166622844696044,0.7874999642372131
# ...


def df_to_sensitivities(df):
    param_names = list(set(df["parameter"]))

    sensitivities = {}
    for param_name in param_names:
        sensitivities[param_name] = OrderedDict()
        param_stats = df[(df.parameter == param_name)]

        for row in range(len(param_stats.index)):
            s = param_stats.iloc[[row]].sparsity
            top1_acc = param_stats.iloc[[row]].top1_acc
            sensitivities[param_name][float(s)] = float(top1_acc)
    return sensitivities


def view_layer_sparsities(df):
    param_names = sorted(list(set(df["parameter"])))

    param_dropdown = widgets.Dropdown(description="Parameter:", options=param_names)

    def plot_layer_sparsities(param_name):
        display(df[df["parameter"] == param_name])

    interact(plot_layer_sparsities, param_name=param_dropdown)


def view_layers_sensitivies_comparison(df):
    out = widgets.Output(layout=widgets.Layout(height="100%"))

    # assign a different color to each parameter (otherwise, colors change on us as we make different selections)
    param_names = sorted(df["parameter"].unique().tolist())
    color_idx = np.linspace(0, 1, len(param_names))
    colors = {}
    for i, pname in zip(color_idx, param_names):
        colors[pname] = plt.get_cmap("tab20")(i)
    plt.rcParams.update({"font.size": 18})

    items = ["All"] + param_names

    def plot_layers_sensitivies_comparison(weights="", acc=0):
        sensitivities = None
        if weights[0] == "All":
            sensitivities = df_to_sensitivities(df)
        else:
            mask = False
            mask = [(df.parameter == pname) for pname in weights]
            mask = np.logical_or.reduce(mask)
            sensitivities = df_to_sensitivities(df[mask])

        # Plot the sensitivities
        plt.figure(figsize=(25, 10))
        for param_name, sensitivity in sensitivities.items():
            sense, sparsities = [], []
            for sparsity, value in sensitivity.items():
                sparsities.append(sparsity)
                sense.append(value)
            plt.plot(
                sparsities,
                sense,
                label=param_name,
                marker="o",
                markersize=10,
                color=colors[param_name],
            )

        plt.ylabel("top1_acc")
        plt.xlabel("sparsity")
        plt.title("Pruning Sensitivity")
        # plt.legend(loc='lower center', ncol=2, mode="expand", borderaxespad=0.);
        plt.grid()
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            fancybox=True,
            shadow=True,
            ncol=3,
        )
        plt.show()

    w = widgets.SelectMultiple(
        options=items,
        value=[items[0]],
        layout=Layout(width="100%"),
        description="Weights:",
        rows=len(items),
    )
    acc_widget = widgets.RadioButtons(
        options={"top1_acc": 0}, value=0, description="Accuracy:"
    )
    out = widgets.interactive_output(
        plot_layers_sensitivies_comparison, {"acc": acc_widget, "weights": w}
    )

    ui = widgets.HBox([w, acc_widget])
    display(ui, out)
