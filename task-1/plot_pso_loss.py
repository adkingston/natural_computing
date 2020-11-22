"""
Takes the loss recorded in epoch_loss_train.csv and epoch_loss_test.csv and
creates a plot of the average losses across the iterations at each epoch
"""

import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd


def make_plot(optimizer, inputType, color1, color2):
    train_losses = pd.read_csv(
        f"data/figures/{optimizer}_{inputType}_train.csv").mean(axis=0)
    test_losses = pd.read_csv(
        f"data/figures/{optimizer}_{inputType}_test.csv").mean(axis=0)

    domain = [x for x in range(len(train_losses))]

    plt.figure()

    plt.plot(
        domain,
        train_losses,
        color=color1,
        label='training loss PSO')
    plt.plot(domain, test_losses, color=color2, label='testing loss PSO')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.savefig(f"data/figures/{optimizer}_{inputType}.png")


def make_dual_plot(inputType, color1, color2, color3, color4):
    pso_train_losses = pd.read_csv(
        f"data/figures/pso_{inputType}_train.csv").mean(axis=0)
    pso_test_losses = pd.read_csv(
        f"data/figures/pso_{inputType}_test.csv").mean(axis=0)
    sgd_train_losses = pd.read_csv(
        f"data/figures/sgd_{inputType}_train.csv").mean(axis=0)
    sgd_test_losses = pd.read_csv(
        f"data/figures/sgd_{inputType}_test.csv").mean(axis=0)

    plt.figure()

    domain = range(len(pso_train_losses))
    plt.plot(
        domain,
        pso_train_losses,
        color=color1,
        label='training loss PSO')
    plt.plot(
        domain,
        pso_test_losses,
        color=color2,
        label='testing loss PSO')
    plt.plot(
        domain,
        sgd_train_losses,
        color=color3,
        label='training loss SGD')
    plt.plot(
        domain,
        sgd_test_losses,
        color=color4,
        label='testing loss SGD')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.savefig(f"data/figures/{inputType}.png")


if __name__ == "__main__":
    make_plot("pso", "nonlinear", "royalblue", "navy")
    make_plot("pso", "linear", "darkviolet", "indigo")
    make_plot("sgd", "nonlinear", "darkgreen", "springgreen")
    make_plot("sgd", "linear", "red", "darkred")

    make_dual_plot(
        "nonlinear",
        "royalblue",
        "navy",
        "darkgreen",
        "springgreen")
    make_dual_plot("linear", "darkviolet", "indigo", "red", "darkred")
