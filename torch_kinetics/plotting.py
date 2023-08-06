from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

CMAP = plt.get_cmap("tab10")


def plot_simulated_vs_true(
    timeline: np.ndarray,
    simulated_trajectory: np.ndarray,
    true_trajectory: np.ndarray,
    state_mapping: Mapping[str, int],
):
    cmap_ind = 0
    for metabolite, ind in state_mapping.items():
        # for now, we assume that enzymes concentration doesn't change
        if not metabolite.startswith("enz"):
            plt.plot(
                timeline,
                true_trajectory[:, ind],
                ls="--",
                color=CMAP(cmap_ind),
                alpha=0.5,
                label=f"{metabolite}_true",
            )

            plt.plot(
                timeline,
                simulated_trajectory[:, ind],
                color=CMAP(cmap_ind),
                label=metabolite,
            )

            cmap_ind += 1

    plt.legend()
    plt.show()


def plot_batched_simulated_vs_true(
    batched_timeline: np.ndarray,
    batched_simulated_trajectory: np.ndarray,
    true_trajectory: np.ndarray,
    state_mapping: Mapping[str, int],
):
    cmap_ind = 0
    for metabolite, ind in state_mapping.items():
        # for now, we assume that enzymes concentration doesn't change
        if not metabolite.startswith("enz"):
            plt.plot(
                # convert from batched timeline to normal
                batched_timeline.reshape(-1),
                true_trajectory[:, ind],
                ls="--",
                color=CMAP(cmap_ind),
                alpha=0.5,
                label=f"{metabolite}_true",
            )

            for batch_ind, tt in enumerate(batched_timeline):
                plt.plot(
                    tt,
                    batched_simulated_trajectory[:, batch_ind, ind],
                    color=CMAP(cmap_ind),
                    label=metabolite if batch_ind == 0 else None,
                )

            cmap_ind += 1

    plt.legend()
    plt.show()
