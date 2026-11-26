"""Browse the consolidated appliance event CSVs. Click Run to open the viewer.

Choose an appliance, optionally filter dataset/house, and select an event.
Use the event-list scrollbar, Previous/Next buttons, or arrow keys to browse.
Power is shown in original watts; indices refer to the aligned source table.
"""

from pathlib import Path
from io import BytesIO
import json
import queue
import threading
import tkinter as tk
from tkinter import ttk

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


DATA_ROOT = Path(__file__).resolve().parent
APPLIANCES = ["kettle", "fridge", "dishwasher", "washingmachine", "microwave"]


def load_event_index(appliance, cancel, progress):
    """Index CSV byte ranges in a worker; never load the full waveform table."""
    source = DATA_ROOT / appliance / "events.csv"
    manifest = DATA_ROOT / "manifest.csv"
    signature = [source.stat().st_size, source.stat().st_mtime_ns,
                 manifest.stat().st_size, manifest.stat().st_mtime_ns]
    cache = DATA_ROOT / ".viewer_cache"
    index_file = cache / f"{appliance}.csv"
    signature_file = cache / f"{appliance}.json"
    if index_file.exists() and signature_file.exists():
        if json.loads(signature_file.read_text()) == signature:
            return pd.read_csv(index_file)

    events = pd.read_csv(manifest)
    paths = events["file"].str.replace("\\", "/", regex=False)
    events = events[paths.str.startswith(appliance + "/")].sort_values("sample_start_row").reset_index(drop=True)
    starts, ends = [], []
    expected_row = 0
    with source.open("rb") as stream:
        stream.readline()  # Header is excluded from manifest row numbers.
        for number, row in enumerate(events.itertuples(), start=1):
            if cancel.is_set():
                return None
            if row.sample_start_row != expected_row:
                raise ValueError("Manifest row ranges are not contiguous; regenerate the manifest.")
            starts.append(stream.tell())
            for _ in range(int(row.samples)):
                if not stream.readline():
                    raise ValueError("CSV is shorter than its manifest; extraction may be incomplete.")
            ends.append(stream.tell())
            expected_row += int(row.samples)
            if number % 2000 == 0:
                progress(f"Indexing {appliance}: {number:,}/{len(events):,} events; window remains usable")
        if stream.read(1):
            raise ValueError("CSV has extra rows missing from manifest; extraction may be incomplete.")
    if cancel.is_set():
        return None
    if signature != [source.stat().st_size, source.stat().st_mtime_ns,
                     manifest.stat().st_size, manifest.stat().st_mtime_ns]:
        raise ValueError("CSV or manifest changed during indexing; reload after extraction finishes.")
    events["_byte_start"], events["_byte_end"] = starts, ends
    cache.mkdir(exist_ok=True)
    events.to_csv(index_file, index=False)
    signature_file.write_text(json.dumps(signature))
    return events


def read_event(source, meta):
    with source.open("rb") as stream:
        header = stream.readline()
        stream.seek(int(meta["_byte_start"]))
        payload = stream.read(int(meta["_byte_end"]) - int(meta["_byte_start"]))
    event = pd.read_csv(BytesIO(header + payload))
    if len(event) != int(meta["samples"]) or not event["event_id"].eq(meta["event_id"]).all():
        raise ValueError("Event does not match manifest; CSV may have changed. Reload the appliance.")
    return event


class EventViewer:
    def __init__(self, root):
        self.root = root
        self.data = pd.DataFrame()
        self.events = pd.DataFrame()
        self.visible_events = pd.DataFrame()
        self.messages = queue.Queue()
        self.generation = 0
        self.event_request = 0
        self.cancel = threading.Event()
        self.loaded_event_id = None
        self.page_start = 0
        self.page_size = 500
        root.title("Appliance waveform reference - event browser")
        root.geometry("1400x850")

        controls = ttk.Frame(root, padding=10)
        controls.pack(fill="x")
        self.appliance = tk.StringVar(value=APPLIANCES[0])
        self.dataset = tk.StringVar(value="All")
        self.house = tk.StringVar(value="All")
        self.plot_style = "line"
        self.show_aggregate_background = True
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

        self.style_button = ttk.Button(controls, text="Plot: Line + points",
                                       command=self.toggle_plot_style)
        self.style_button.pack(side="left", padx=12)
        self.aggregate_button = ttk.Button(controls, text="Aggregate background: ON",
                                           command=self.toggle_aggregate_background)
        self.aggregate_button.pack(side="left", padx=4)

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
        pages = ttk.Frame(sidebar)
        pages.pack(fill="x", pady=3)
        ttk.Button(pages, text="Previous page", command=lambda: self.change_page(-1)).pack(side="left")
        ttk.Button(pages, text="Next page", command=lambda: self.change_page(1)).pack(side="right")
        self.page_info = tk.StringVar()
        ttk.Label(sidebar, textvariable=self.page_info).pack()
        self.info = tk.StringVar()
        ttk.Label(sidebar, textvariable=self.info, justify="left",
                  padding=6, wraplength=340).pack(fill="x")

        plot_frame = ttk.Frame(body)
        body.add(plot_frame, weight=4)
        # The appliance waveform already shows aggregate power as an optional
        # grey background. A separate aggregate-only plot repeated that view.
        self.figure = Figure(figsize=(10, 6), constrained_layout=True)
        self.axes = self.figure.subplots(2, 1, sharex=True,
                                         gridspec_kw={"height_ratios": [4, 1]})
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, plot_frame)
        root.bind("<Left>", lambda _: self.move(-1))
        root.bind("<Right>", lambda _: self.move(1))
        root.after(100, self.load_appliance)
        root.after(100, self.poll_messages)

    def load_appliance(self, _=None):
        self.cancel.set()
        self.cancel = threading.Event()
        self.generation += 1
        self.event_request += 1
        self.loaded_event_id = None
        self.data = pd.DataFrame()
        self.events = pd.DataFrame()
        self.visible_events = pd.DataFrame()
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
        generation, appliance, cancel = self.generation, self.appliance.get(), self.cancel
        def worker():
            try:
                events = load_event_index(appliance, cancel,
                    lambda message: self.messages.put((generation, "progress", message)))
                if events is not None:
                    self.messages.put((generation, "index", events))
            except Exception as error:
                self.messages.put((generation, "error", str(error)))
        threading.Thread(target=worker, daemon=True).start()

    def poll_messages(self):
        """Only the Tk thread updates widgets; stale worker results are ignored."""
        while True:
            try:
                generation, kind, payload = self.messages.get_nowait()
            except queue.Empty:
                break
            if generation != self.generation:
                continue
            if kind in ("progress", "error"):
                self.status.set(payload)
            elif kind == "index":
                self.events = payload
                self.dataset.set("All")
                self.house.set("All")
                self.dataset_box["values"] = ["All"] + sorted(self.events["dataset"].unique())
                self.house_box["values"] = ["All"] + [str(h) for h in sorted(self.events["house"].unique())]
                self.filter_events()
            elif kind == "event":
                request, event = payload
                if request == self.event_request:
                    self.data = event
                    self.loaded_event_id = event["event_id"].iloc[0]
                    self.show_event()
        self.root.after(100, self.poll_messages)

    def filter_events(self, _=None):
        if self.events.empty:
            return
        self.event_request += 1
        selected = self.events
        if self.dataset.get() != "All":
            selected = selected[selected["dataset"] == self.dataset.get()]
        if self.house.get() != "All":
            selected = selected[selected["house"].astype(str) == self.house.get()]
        self.visible_events = selected.reset_index(drop=True)
        self.page_start = 0
        self.populate_page()
        self.status.set(f"{self.appliance.get()} | {len(selected):,} matching events / "
                        f"{len(self.events):,} total | {int(self.events.samples.sum()):,} sample rows")
        if len(selected):
            self.event_list.selection_set(0)
            self.show_event()
        else:
            self.info.set("No events match these filters.")
            for ax in self.axes:
                ax.clear()
            self.canvas.draw_idle()

    def populate_page(self):
        self.event_list.delete(0, tk.END)
        end = min(self.page_start + self.page_size, len(self.visible_events))
        labels = [f"{number}. {event_id}" for number, event_id in
                  enumerate(self.visible_events["event_id"].iloc[self.page_start:end],
                            start=self.page_start + 1)]
        if labels:
            self.event_list.insert(tk.END, *labels)
        self.page_info.set(f"Events {self.page_start + 1 if labels else 0:,}-{end:,} / {len(self.visible_events):,}")

    def change_page(self, delta):
        if self.visible_events.empty:
            return
        last_page = (len(self.visible_events) - 1) // self.page_size * self.page_size
        self.page_start = min(max(0, self.page_start + delta * self.page_size), last_page)
        self.populate_page()
        self.event_list.selection_set(0)
        self.show_event()

    def move(self, delta):
        selected = self.event_list.curselection()
        if not selected:
            return
        index = min(max(self.page_start + selected[0] + delta, 0), len(self.visible_events) - 1)
        page = index // self.page_size * self.page_size
        if page != self.page_start:
            self.page_start = page
            self.populate_page()
        local_index = index - self.page_start
        self.event_list.selection_clear(0, tk.END)
        self.event_list.selection_set(local_index)
        self.event_list.see(local_index)
        self.show_event()

    def toggle_plot_style(self):
        """Change only the power display; keep samples and state labels intact."""
        self.plot_style = "step" if self.plot_style == "line" else "line"
        label = "Step" if self.plot_style == "step" else "Line"
        self.style_button.configure(text=f"Plot: {label} + points")
        self.show_event()

    def toggle_aggregate_background(self):
        self.show_aggregate_background = not self.show_aggregate_background
        state = "ON" if self.show_aggregate_background else "OFF"
        self.aggregate_button.configure(text=f"Aggregate background: {state}")
        self.show_event()

    def show_event(self, _=None):
        selected = self.event_list.curselection()
        if not selected:
            return
        index = self.page_start + selected[0]
        meta = self.visible_events.iloc[index]
        if self.loaded_event_id != meta["event_id"]:
            self.event_request += 1
            request, generation = self.event_request, self.generation
            source = DATA_ROOT / self.appliance.get() / "events.csv"
            self.info.set(f"Loading event {index + 1:,}...")
            def worker():
                try:
                    self.messages.put((generation, "event", (request, read_event(source, meta))))
                except Exception as error:
                    self.messages.put((generation, "error", str(error)))
            threading.Thread(target=worker, daemon=True).start()
            return
        event = self.data
        seconds = event["unix_time"].to_numpy() - event["unix_time"].iloc[0]
        # Shade the saved event labels on actual time boundaries, not a median dt.
        on = event["event_on" if "event_on" in event else "is_on"].to_numpy(dtype=bool)
        on_time = seconds[on]
        if "on_end_unix" in meta:
            on_end_time = meta["on_end_unix"] - event["unix_time"].iloc[0]
        elif "sample_end_unix" in event:
            on_end_time = event.loc[on, "sample_end_unix"].iloc[-1] - event["unix_time"].iloc[0]
        else:
            on_end_time = on_time[-1]
        power_ax = self.axes[0]
        power_ax.clear()
        if self.show_aggregate_background:
            power_ax.plot(seconds, event["aggregate_power_w"], color="#999999",
                          alpha=0.55, linewidth=1.0, zorder=1,
                          drawstyle="steps-post" if self.plot_style == "step" else "default",
                          label="Aggregate (background)")
        power_ax.plot(seconds, event["appliance_power_w"], marker="o", markersize=3,
                      drawstyle="steps-post" if self.plot_style == "step" else "default",
                      color="#1678b4", linewidth=1.2, zorder=2,
                      label=self.appliance.get().capitalize() + " (sample values)")
        if len(on_time):
            power_ax.axvspan(on_time[0], on_end_time,
                             color="#74bf83", alpha=0.22, label="ON event")
        power_ax.set_ylabel("Power (W)")
        power_ax.grid(alpha=0.2)
        power_ax.legend(loc="upper right")
        labels_ax = self.axes[-1]
        labels_ax.clear()
        labels_ax.step(seconds, on.astype(int), where="post", label="Event ON", color="#348a4c")
        if "above_threshold" in event:
            labels_ax.step(seconds, event["above_threshold"], where="post",
                           label="Above threshold", color="#d66c13", linestyle="--")
        labels_ax.set_ylim(-0.15, 1.25)
        labels_ax.set_yticks([0, 1])
        labels_ax.set_ylabel("State")
        labels_ax.legend(loc="upper right", fontsize=8)
        labels_ax.grid(alpha=0.2)
        labels_ax.set_xlabel("Seconds from reference start (actual timestamps; no time shifting)")
        self.figure.suptitle(f"Event {index + 1:,}/{len(self.visible_events):,}\n{meta['event_id']}")
        self.info.set(
            f"Dataset: {meta['dataset']} | House: {meta['house']} | {meta['channel']}\n"
            f"Duration: {meta['duration_sec']:.0f} s\n"
            f"Incomplete boundary: {meta.get('duration_censored', 'unknown (old CSV)')}\n"
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
