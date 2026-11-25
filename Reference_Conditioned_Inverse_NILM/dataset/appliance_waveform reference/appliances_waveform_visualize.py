"""Browse the consolidated appliance event CSVs. Click Run to open the viewer.

Choose an appliance, optionally filter dataset/house, and select an event.
Use the event-list scrollbar, Previous/Next buttons, or arrow keys to browse.
Power is shown in original watts; indices refer to the aligned source table.
"""

from pathlib import Path
import tkinter as tk
from tkinter import ttk

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


DATA_ROOT = Path(__file__).resolve().parent
APPLIANCES = ["kettle", "fridge", "dishwasher", "washingmachine", "microwave"]


class EventViewer:
    def __init__(self, root):
        self.root = root
        self.data = pd.DataFrame()
        self.events = pd.DataFrame()
        self.visible_events = pd.DataFrame()
        self.event_rows = {}
        root.title("Appliance waveform reference - event browser")
        root.geometry("1400x850")

        controls = ttk.Frame(root, padding=10)
        controls.pack(fill="x")
        self.appliance = tk.StringVar(value=APPLIANCES[0])
        self.dataset = tk.StringVar(value="All")
        self.house = tk.StringVar(value="All")
        for label, variable, values in [
            ("Appliance", self.appliance, APPLIANCES),
            ("Dataset", self.dataset, ["All"]),
            ("House", self.house, ["All"]),
        ]:
            ttk.Label(controls, text=label).pack(side="left", padx=(8, 4))
            box = ttk.Combobox(controls, textvariable=variable, values=values,
                               state="readonly", width=18)
            box.pack(side="left")
            if label == "Appliance":
                box.bind("<<ComboboxSelected>>", self.load_appliance)
            else:
                box.bind("<<ComboboxSelected>>", self.filter_events)
            if label == "Dataset":
                self.dataset_box = box
            elif label == "House":
                self.house_box = box

        self.status = tk.StringVar(value="Choose an appliance")
        ttk.Label(root, textvariable=self.status, padding=(12, 4)).pack(anchor="w")
        body = ttk.Panedwindow(root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=10, pady=5)
        sidebar = ttk.Frame(body, width=340)
        body.add(sidebar, weight=1)
        list_frame = ttk.Frame(sidebar)
        list_frame.pack(fill="both", expand=True)
        self.event_list = tk.Listbox(list_frame, exportselection=False, width=42)
        scrollbar = ttk.Scrollbar(list_frame, command=self.event_list.yview)
        self.event_list.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.event_list.pack(fill="both", expand=True)
        self.event_list.bind("<<ListboxSelect>>", self.show_event)

        buttons = ttk.Frame(sidebar)
        buttons.pack(fill="x", pady=6)
        ttk.Button(buttons, text="Previous", command=lambda: self.move(-1)).pack(side="left")
        ttk.Button(buttons, text="Next", command=lambda: self.move(1)).pack(side="right")
        self.info = tk.StringVar()
        ttk.Label(sidebar, textvariable=self.info, justify="left",
                  padding=6, wraplength=340).pack(fill="x")

        plot_frame = ttk.Frame(body)
        body.add(plot_frame, weight=4)
        self.figure = Figure(figsize=(10, 7), constrained_layout=True)
        self.axes = self.figure.subplots(2, 1, sharex=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, plot_frame)
        root.bind("<Left>", lambda _: self.move(-1))
        root.bind("<Right>", lambda _: self.move(1))
        root.after(100, self.load_appliance)

    def load_appliance(self, _=None):
        path = DATA_ROOT / self.appliance.get() / "events.csv"
        self.status.set(f"Loading {path.name} for {self.appliance.get()}...")
        self.root.update_idletasks()
        self.event_list.delete(0, tk.END)
        self.info.set("")
        for ax in self.axes:
            ax.clear()
        self.canvas.draw_idle()
        if not path.exists():
            self.data = pd.DataFrame()
            self.status.set(f"No extracted file yet: {path}")
            return
        try:
            self.data = pd.read_csv(path)
            required = {"event_id", "dataset", "house", "channel", "source_idx",
                        "unix_time", "appliance_power_w", "aggregate_power_w",
                        "on_start_idx", "on_end_idx", "ref_start_idx", "ref_end_idx",
                        "off_pre_samples", "off_post_samples", "duration_sec",
                        "peak_power_w", "mean_on_power_w", "timestamp"}
            missing = required - set(self.data.columns)
            if missing:
                raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
            # Store row locations once, so switching events needs no full CSV scan.
            self.event_rows = self.data.groupby("event_id", sort=False).indices
            self.events = self.data.drop_duplicates("event_id").copy()
            self.dataset.set("All")
            self.house.set("All")
            self.dataset_box["values"] = ["All"] + sorted(self.events["dataset"].unique())
            self.house_box["values"] = ["All"] + [str(h) for h in sorted(self.events["house"].unique())]
            self.filter_events()
        except (OSError, ValueError, pd.errors.EmptyDataError) as error:
            self.data = pd.DataFrame()
            self.status.set(f"Cannot load {path}: {error}")

    def filter_events(self, _=None):
        if self.data.empty:
            return
        selected = self.events
        if self.dataset.get() != "All":
            selected = selected[selected["dataset"] == self.dataset.get()]
        if self.house.get() != "All":
            selected = selected[selected["house"].astype(str) == self.house.get()]
        self.visible_events = selected.reset_index(drop=True)
        self.event_list.delete(0, tk.END)
        for number, row in self.visible_events.iterrows():
            self.event_list.insert(tk.END, f"{number + 1}. {row['event_id']}")
        self.status.set(f"{self.appliance.get()} | {len(selected):,} matching events / "
                        f"{len(self.events):,} total | {len(self.data):,} sample rows")
        if len(selected):
            self.event_list.selection_set(0)
            self.show_event()
        else:
            self.info.set("No events match these filters.")
            for ax in self.axes:
                ax.clear()
            self.canvas.draw_idle()

    def move(self, delta):
        selected = self.event_list.curselection()
        if not selected:
            return
        index = min(max(selected[0] + delta, 0), self.event_list.size() - 1)
        self.event_list.selection_clear(0, tk.END)
        self.event_list.selection_set(index)
        self.event_list.see(index)
        self.show_event()

    def show_event(self, _=None):
        selected = self.event_list.curselection()
        if not selected:
            return
        index = selected[0]
        meta = self.visible_events.iloc[index]
        event = self.data.iloc[self.event_rows[meta["event_id"]]]
        seconds = event["unix_time"].to_numpy() - event["unix_time"].iloc[0]
        # Inclusive source endpoints: extend the shaded region by half a sample.
        step = float(np.median(np.diff(seconds))) if len(seconds) > 1 else 8.0
        on = event["source_idx"].between(meta["on_start_idx"], meta["on_end_idx"]).to_numpy()
        on_time = seconds[on]
        for ax, column, label, color in zip(
            self.axes, ["aggregate_power_w", "appliance_power_w"],
            ["Aggregate", self.appliance.get().capitalize()], ["#777777", "#1678b4"]
        ):
            ax.clear()
            ax.plot(seconds, event[column], color=color, linewidth=1.5, label=label)
            if len(on_time):
                ax.axvspan(on_time[0] - step / 2, on_time[-1] + step / 2,
                           color="#74bf83", alpha=0.22, label="ON event")
            ax.set_ylabel("Power (W)")
            ax.grid(alpha=0.2)
            ax.legend(loc="upper right")
        self.axes[-1].set_xlabel("Seconds from reference start (OFF context included)")
        self.figure.suptitle(f"Event {index + 1:,}/{len(self.visible_events):,}\n{meta['event_id']}")
        self.info.set(
            f"Dataset: {meta['dataset']} | House: {meta['house']} | {meta['channel']}\n"
            f"Duration: {meta['duration_sec']:.0f} s\n"
            f"Peak: {meta['peak_power_w']:.1f} W\n"
            f"Mean during event: {meta['mean_on_power_w']:.1f} W\n"
            f"OFF context: {meta['off_pre_samples']} before / {meta['off_post_samples']} after\n"
            f"ON indices: {meta['on_start_idx']} - {meta['on_end_idx']}\n"
            f"Reference indices: {meta['ref_start_idx']} - {meta['ref_end_idx']}\n"
            f"Samples: {len(event):,}\n"
            f"Start: {event['timestamp'].iloc[0]}\nEnd: {event['timestamp'].iloc[-1]}"
        )
        self.canvas.draw_idle()


def main():
    root = tk.Tk()
    viewer = EventViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
