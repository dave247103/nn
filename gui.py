import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from main import NeuralNet, generate_dataset, train_test_split


class App:
    def __init__(self, root):
        self.root = root
        root.title("Shallow NN")

        self.m0_var = tk.StringVar(value="2")
        self.m1_var = tk.StringVar(value="2")
        self.spm_var = tk.StringVar(value="100")
        self.hidden_var = tk.StringVar(value="8,8")
        self.lr_var = tk.StringVar(value="0.5")
        self.epochs_var = tk.StringVar(value="500")
        self.batch_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready")

        self.X = None
        self.y = None
        self.net = None

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(main)
        controls.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))

        data_box = ttk.LabelFrame(controls, text="Data")
        data_box.pack(fill=tk.X, pady=(0, 8))
        self._add_row(data_box, "Modes class 0", self.m0_var)
        self._add_row(data_box, "Modes class 1", self.m1_var)
        self._add_row(data_box, "Samples/mode", self.spm_var)
        ttk.Button(data_box, text="Generate", command=self.on_generate).pack(fill=tk.X, pady=4)

        net_box = ttk.LabelFrame(controls, text="Network")
        net_box.pack(fill=tk.X, pady=(0, 8))
        self._add_row(net_box, "Hidden sizes", self.hidden_var)
        self._add_row(net_box, "Learning rate", self.lr_var)
        self._add_row(net_box, "Epochs", self.epochs_var)
        self._add_row(net_box, "Batch size", self.batch_var)
        ttk.Button(net_box, text="Train", command=self.on_train).pack(fill=tk.X, pady=(4, 2))
        ttk.Button(net_box, text="Clear", command=self.on_clear).pack(fill=tk.X)

        ttk.Label(controls, textvariable=self.status_var, wraplength=220).pack(fill=tk.X, pady=(8, 0))

        fig = Figure(figsize=(6, 6), dpi=100)
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(fig, master=main)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self._plot()

    def _add_row(self, parent, label, var):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=14).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=var, width=10).pack(side=tk.RIGHT, fill=tk.X, expand=True)

    def _parse_int(self, var, name, min_val=1):
        try:
            v = int(var.get())
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if v < min_val:
            raise ValueError(f"{name} must be >= {min_val}")
        return v

    def _parse_float(self, var, name, min_val=None):
        try:
            v = float(var.get())
        except ValueError as exc:
            raise ValueError(f"{name} must be a number") from exc
        if min_val is not None and v <= min_val:
            raise ValueError(f"{name} must be > {min_val}")
        return v

    def _parse_hidden(self):
        s = self.hidden_var.get().strip()
        if not s:
            raise ValueError("Hidden sizes required")
        parts = [p.strip() for p in s.split(",") if p.strip()]
        hidden = []
        for p in parts:
            try:
                v = int(p)
            except ValueError as exc:
                raise ValueError("Hidden sizes must be integers") from exc
            if v <= 0:
                raise ValueError("Hidden sizes must be > 0")
            hidden.append(v)
        if len(hidden) == 0:
            raise ValueError("At least one hidden layer")
        if len(hidden) > 3:
            raise ValueError("At most 3 hidden layers (<=5 layers total)")
        return hidden

    def on_generate(self):
        try:
            m0 = self._parse_int(self.m0_var, "Modes class 0")
            m1 = self._parse_int(self.m1_var, "Modes class 1")
            spm = self._parse_int(self.spm_var, "Samples/mode")
        except ValueError as e:
            self.status_var.set(str(e))
            return
        self.X, self.y = generate_dataset(m0, m1, spm)
        self.net = None
        self.status_var.set(f"Generated {len(self.y)} samples")
        self._plot()

    def on_train(self):
        if self.X is None or len(self.y) == 0:
            self.status_var.set("Generate data first")
            return
        try:
            hidden = self._parse_hidden()
            eta = self._parse_float(self.lr_var, "Learning rate", min_val=0.0)
            epochs = self._parse_int(self.epochs_var, "Epochs")
            bs_text = self.batch_var.get().strip()
            batch = int(bs_text) if bs_text else None
            if batch is not None and batch <= 0:
                raise ValueError("Batch size must be > 0")
        except ValueError as e:
            self.status_var.set(str(e))
            return

        layer_sizes = [2] + hidden + [2]
        if len(layer_sizes) > 5:
            self.status_var.set("Total layers must be <= 5")
            return

        Xtr, Xte, ytr, yte = train_test_split(self.X, self.y, test_ratio=0.2, seed=0)
        net = NeuralNet(layer_sizes, seed=0)
        net.train(Xtr, ytr, eta=eta, epochs=epochs, batch_size=batch)
        self.net = net

        tr_acc = (net.predict(Xtr) == ytr).mean() if len(ytr) else 0.0
        te_acc = (net.predict(Xte) == yte).mean() if len(yte) else 0.0
        self.status_var.set(f"Train acc: {tr_acc:.3f} | Test acc: {te_acc:.3f}")
        self._plot()

    def on_clear(self):
        self.X = None
        self.y = None
        self.net = None
        self.status_var.set("Cleared")
        self._plot()

    def _plot(self):
        self.ax.clear()
        if self.X is not None and len(self.y) > 0:
            if self.net is not None:
                self._plot_regions()
            self.ax.scatter(self.X[self.y == 0, 0], self.X[self.y == 0, 1], s=18, c="#d62728", label="class 0")
            self.ax.scatter(self.X[self.y == 1, 0], self.X[self.y == 1, 1], s=18, c="#1f77b4", label="class 1")
            self.ax.legend(loc="upper right", frameon=False)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_title("Data and Decision Regions")
        self.canvas.draw()

    def _plot_regions(self):
        x_min, x_max = self.X[:, 0].min() - 0.6, self.X[:, 0].max() + 0.6
        y_min, y_max = self.X[:, 1].min() - 0.6, self.X[:, 1].max() + 0.6
        xs = np.linspace(x_min, x_max, 200)
        ys = np.linspace(y_min, y_max, 200)
        xx, yy = np.meshgrid(xs, ys)
        grid = np.c_[xx.ravel(), yy.ravel()]
        proba = self.net.predict_proba(grid)
        preds = np.argmax(proba, axis=1).reshape(xx.shape)
        self.ax.contourf(xx, yy, preds, levels=[-0.5, 0.5, 1.5], colors=["#ffe0e0", "#e0ecff"], alpha=0.8)
        diff = (proba[:, 1] - proba[:, 0]).reshape(xx.shape)
        self.ax.contour(xx, yy, diff, levels=[0.0], colors="k", linewidths=1.0)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
