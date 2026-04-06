"""
gui.py
------
Visualization and user interaction.
Tkinter-based GUI with:
  - File loading and data type selection
  - All 6 filter options (LPF/HPF/BPF × IIR/FIR) with user-adjustable parameters
  - Time-domain plot (raw vs filtered) with zoom/pan
  - FFT plots with dominant frequency highlighting
  - Statistics and feature extraction panel
  - Tooltips for usability
  - Reset / reload without restarting
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from data_loader import (
    VALID_SIGNAL_TYPES,
    load_csv_file,
    validate_and_clean_data,
    estimate_sampling_frequency,
)
from preprocessing import apply_preprocessing
from filters import (
    apply_filter,
    get_default_lowpass_cutoff,
    get_default_highpass_cutoff,
    get_default_bandpass_cutoffs,
)
from analysis import (
    compute_statistics,
    compute_fft,
    find_dominant_frequencies,
    extract_features,
)


# ── Filter option map ────────────────────────────────────────────────────────

FILTER_OPTIONS = {
    "Low-Pass Filter (IIR)":   "LPF_IIR",
    "High-Pass Filter (IIR)":  "HPF_IIR",
    "Band-Pass Filter (IIR)":  "BPF_IIR",
    "Low-Pass Filter (FIR)":   "LPF_FIR",
    "High-Pass Filter (FIR)":  "HPF_FIR",
    "Band-Pass Filter (FIR)":  "BPF_FIR",
}


# ── Tooltip helper ───────────────────────────────────────────────────────────

class ToolTip:
    """Simple hover tooltip for any tkinter widget."""

    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event):
        if self.tip_window:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw, text=self.text, justify=tk.LEFT,
            background="#ffffe0", relief=tk.SOLID, borderwidth=1,
            font=("Arial", 9),
        )
        label.pack(ipadx=4, ipady=2)

    def _hide(self, _event):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


# ── Plotting helpers ─────────────────────────────────────────────────────────

def _clear_frame(frame: tk.Frame):
    for w in frame.winfo_children():
        w.destroy()


def _embed_figure(frame: tk.Frame, fig: Figure):
    """Embed a matplotlib figure in a tkinter frame with navigation toolbar."""
    _clear_frame(frame)
    toolbar_frame = tk.Frame(frame)
    toolbar_frame.pack(side=tk.TOP, fill=tk.X)
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    NavigationToolbar2Tk(canvas, toolbar_frame).update()


def create_time_figure(time, raw_signal, filtered_signal,
                       signal_type, filter_name):
    """Matplotlib figure: raw vs filtered signal in time domain."""
    fig = Figure(figsize=(7, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(time, raw_signal, label="Raw Signal", alpha=0.6)
    ax.plot(time, filtered_signal, label=f"Filtered ({filter_name})", linewidth=1.2)
    ax.set_title(f"{signal_type} — Raw vs {filter_name}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def create_fft_figure(frequencies, magnitudes, signal_type, title,
                      dominant_freqs=None):
    """
    Matplotlib figure for FFT magnitude spectrum.
    If *dominant_freqs* is provided, marks them on the plot.
    """
    fig = Figure(figsize=(7, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(frequencies, magnitudes, linewidth=0.8)
    ax.set_title(f"{signal_type} — {title}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.grid(True, alpha=0.3)

    # Highlight dominant frequencies
    if dominant_freqs:
        for i, d in enumerate(dominant_freqs):
            ax.axvline(d["frequency"], color="red", linestyle="--",
                       linewidth=0.8, alpha=0.7)
            ax.annotate(
                f'{d["frequency"]:.1f} Hz',
                xy=(d["frequency"], d["magnitude"]),
                xytext=(5, 10), textcoords="offset points",
                fontsize=8, color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
            )

    fig.tight_layout()
    return fig


# ── Main GUI ─────────────────────────────────────────────────────────────────

def launch_gui() -> None:
    root = tk.Tk()
    root.title("Sensor Signal Analysis System — SYSC 2010")
    root.geometry("1050x900")
    root.minsize(900, 750)

    # ── Variables ────────────────────────────────────────────────────────
    selected_file = tk.StringVar()
    selected_signal = tk.StringVar(value=VALID_SIGNAL_TYPES[0])
    selected_filter = tk.StringVar(value="Low-Pass Filter (IIR)")
    cutoff_var = tk.StringVar(value="")
    low_cutoff_var = tk.StringVar(value="")
    high_cutoff_var = tk.StringVar(value="")

    # ── Title ────────────────────────────────────────────────────────────
    tk.Label(root, text="Sensor Signal Analysis System",
             font=("Arial", 16, "bold")).pack(pady=(10, 5))
    tk.Label(root, text="Load a CSV, choose signal type & filter, then run.",
             font=("Arial", 10)).pack()

    # ── File selection row ───────────────────────────────────────────────
    file_frame = tk.Frame(root)
    file_frame.pack(pady=8, fill=tk.X, padx=15)

    tk.Label(file_frame, text="CSV File:").pack(side=tk.LEFT, padx=5)
    file_entry = tk.Entry(file_frame, textvariable=selected_file, width=50)
    file_entry.pack(side=tk.LEFT, padx=5)
    browse_btn = tk.Button(
        file_frame, text="Browse",
        command=lambda: _browse_file(selected_file),
    )
    browse_btn.pack(side=tk.LEFT, padx=5)
    ToolTip(browse_btn, "Open a CSV file containing time-series sensor data.")

    # ── Signal type row ──────────────────────────────────────────────────
    sig_frame = tk.Frame(root)
    sig_frame.pack(pady=4, fill=tk.X, padx=15)

    tk.Label(sig_frame, text="Signal Type:").pack(side=tk.LEFT, padx=5)
    sig_combo = ttk.Combobox(sig_frame, textvariable=selected_signal,
                             values=VALID_SIGNAL_TYPES, state="readonly", width=18)
    sig_combo.pack(side=tk.LEFT, padx=5)
    ToolTip(sig_combo, "Select the sensor type so the system can apply\n"
                       "appropriate filter defaults and feature extraction.")

    # ── Filter type row ──────────────────────────────────────────────────
    filt_frame = tk.Frame(root)
    filt_frame.pack(pady=4, fill=tk.X, padx=15)

    tk.Label(filt_frame, text="Filter:").pack(side=tk.LEFT, padx=5)
    filt_combo = ttk.Combobox(filt_frame, textvariable=selected_filter,
                              values=list(FILTER_OPTIONS.keys()),
                              state="readonly", width=28)
    filt_combo.pack(side=tk.LEFT, padx=5)
    ToolTip(filt_combo, "Choose a digital filter.\n"
                        "IIR = Butterworth (recursive), FIR = windowed sinc.")

    # ── Custom cutoff parameter row ──────────────────────────────────────
    param_frame = tk.LabelFrame(root, text="Filter Parameters (leave blank for defaults)",
                                padx=8, pady=4)
    param_frame.pack(pady=4, fill=tk.X, padx=15)

    tk.Label(param_frame, text="Cutoff (Hz):").grid(row=0, column=0, padx=4, sticky="e")
    cutoff_entry = tk.Entry(param_frame, textvariable=cutoff_var, width=10)
    cutoff_entry.grid(row=0, column=1, padx=4)
    ToolTip(cutoff_entry, "Single cutoff for LPF or HPF.\nLeave blank to use signal-type default.")

    tk.Label(param_frame, text="Low (Hz):").grid(row=0, column=2, padx=4, sticky="e")
    low_entry = tk.Entry(param_frame, textvariable=low_cutoff_var, width=10)
    low_entry.grid(row=0, column=3, padx=4)
    ToolTip(low_entry, "Low cutoff for Band-Pass filter.")

    tk.Label(param_frame, text="High (Hz):").grid(row=0, column=4, padx=4, sticky="e")
    high_entry = tk.Entry(param_frame, textvariable=high_cutoff_var, width=10)
    high_entry.grid(row=0, column=5, padx=4)
    ToolTip(high_entry, "High cutoff for Band-Pass filter.")

    # Populate defaults when signal type or filter changes
    def _update_default_hints(*_args):
        sig = selected_signal.get()
        fk = FILTER_OPTIONS.get(selected_filter.get(), "")
        if fk.startswith("BPF"):
            lo, hi = get_default_bandpass_cutoffs(sig)
            low_cutoff_var.set("")
            high_cutoff_var.set("")
            cutoff_var.set("")
            low_entry.config(state="normal")
            high_entry.config(state="normal")
            cutoff_entry.config(state="disabled")
            low_entry.delete(0, tk.END)
            low_entry.insert(0, f"{lo}")
            high_entry.delete(0, tk.END)
            high_entry.insert(0, f"{hi}")
        else:
            low_entry.config(state="disabled")
            high_entry.config(state="disabled")
            cutoff_entry.config(state="normal")
            if fk.startswith("LPF"):
                c = get_default_lowpass_cutoff(sig)
            else:
                c = get_default_highpass_cutoff(sig)
            cutoff_var.set("")
            cutoff_entry.delete(0, tk.END)
            cutoff_entry.insert(0, f"{c}")

    selected_signal.trace_add("write", _update_default_hints)
    selected_filter.trace_add("write", _update_default_hints)
    _update_default_hints()  # initial fill

    # ── Buttons ──────────────────────────────────────────────────────────
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=8)

    run_btn = tk.Button(btn_frame, text="▶  Run Analysis",
                        font=("Arial", 11, "bold"), bg="#4CAF50", fg="white",
                        command=lambda: _run(root, selected_file, selected_signal,
                                             selected_filter, cutoff_var,
                                             low_cutoff_var, high_cutoff_var,
                                             stats_text, info_text,
                                             time_tab, raw_fft_tab, filt_fft_tab))
    run_btn.pack(side=tk.LEFT, padx=8)
    ToolTip(run_btn, "Run the full processing pipeline on the loaded data.")

    reset_btn = tk.Button(btn_frame, text="↺  Reset", font=("Arial", 11),
                          command=lambda: _reset(selected_file, selected_signal,
                                                 selected_filter, cutoff_var,
                                                 low_cutoff_var, high_cutoff_var,
                                                 stats_text, info_text,
                                                 time_tab, raw_fft_tab, filt_fft_tab))
    reset_btn.pack(side=tk.LEFT, padx=8)
    ToolTip(reset_btn, "Clear all results and reload a new file.")

    # ── Info & Stats panel (side by side) ────────────────────────────────
    bottom_panel = tk.Frame(root)
    bottom_panel.pack(fill=tk.X, padx=15, pady=4)

    info_lf = tk.LabelFrame(bottom_panel, text="Signal Info", padx=4, pady=2)
    info_lf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
    info_text = tk.Text(info_lf, height=6, width=35, font=("Consolas", 9))
    info_text.pack(fill=tk.BOTH, expand=True)

    stats_lf = tk.LabelFrame(bottom_panel, text="Statistics & Features", padx=4, pady=2)
    stats_lf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
    stats_text = tk.Text(stats_lf, height=6, width=45, font=("Consolas", 9))
    stats_text.pack(fill=tk.BOTH, expand=True)

    # ── Plot notebook ────────────────────────────────────────────────────
    plot_notebook = ttk.Notebook(root)
    plot_notebook.pack(fill="both", expand=True, padx=15, pady=(4, 10))

    time_tab = tk.Frame(plot_notebook)
    raw_fft_tab = tk.Frame(plot_notebook)
    filt_fft_tab = tk.Frame(plot_notebook)

    plot_notebook.add(time_tab, text="Time Domain (Raw vs Filtered)")
    plot_notebook.add(raw_fft_tab, text="Raw Signal FFT")
    plot_notebook.add(filt_fft_tab, text="Filtered Signal FFT")

    root.mainloop()


# ── Internal callbacks ───────────────────────────────────────────────────────

def _browse_file(var):
    path = filedialog.askopenfilename(
        title="Select CSV File",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
    )
    if path:
        var.set(path)


def _reset(file_var, sig_var, filt_var, cutoff_var,
           low_var, high_var, stats_text, info_text,
           time_tab, raw_fft_tab, filt_fft_tab):
    file_var.set("")
    sig_var.set(VALID_SIGNAL_TYPES[0])
    filt_var.set("Low-Pass Filter (IIR)")
    cutoff_var.set("")
    low_var.set("")
    high_var.set("")
    stats_text.delete("1.0", tk.END)
    info_text.delete("1.0", tk.END)
    _clear_frame(time_tab)
    _clear_frame(raw_fft_tab)
    _clear_frame(filt_fft_tab)


def _safe_float(s: str):
    """Return float or None if blank / invalid."""
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _run(root, file_var, sig_var, filt_var, cutoff_var,
         low_var, high_var, stats_text, info_text,
         time_tab, raw_fft_tab, filt_fft_tab):
    """Execute the full pipeline and update the GUI."""

    file_path = file_var.get().strip()
    signal_type = sig_var.get().strip()
    filter_label = filt_var.get().strip()

    if not file_path:
        messagebox.showerror("Error", "Please select a CSV file first.")
        return

    filter_key = FILTER_OPTIONS.get(filter_label)
    if filter_key is None:
        messagebox.showerror("Error", "Please select a valid filter type.")
        return

    # Parse custom parameters
    custom_cutoff = _safe_float(cutoff_var.get())
    custom_low = _safe_float(low_var.get())
    custom_high = _safe_float(high_var.get())

    try:
        # ── 1. Load & validate ───────────────────────────────────────
        raw_df = load_csv_file(file_path)
        clean_df, clean_report = validate_and_clean_data(raw_df)

        # Keep a copy of the truly raw signal (before preprocessing)
        raw_signal = clean_df["signal"].to_numpy().copy()

        # ── 2. Preprocessing ─────────────────────────────────────────
        proc_df = apply_preprocessing(clean_df, signal_type)

        fs = estimate_sampling_frequency(proc_df)
        time_arr = proc_df["time"].to_numpy()
        signal_arr = proc_df["signal"].to_numpy()

        # ── 3. Filter ────────────────────────────────────────────────
        filtered = apply_filter(
            signal_arr, fs, filter_key, signal_type,
            custom_cutoff=custom_cutoff,
            custom_low=custom_low,
            custom_high=custom_high,
        )

        # ── 4. Statistics ────────────────────────────────────────────
        raw_stats = compute_statistics(signal_arr)
        filt_stats = compute_statistics(filtered)

        # ── 5. Features ──────────────────────────────────────────────
        filt_features = extract_features(filtered, time_arr, signal_type, fs)

        # ── 6. FFT ───────────────────────────────────────────────────
        raw_freqs, raw_mags = compute_fft(signal_arr, fs)
        filt_freqs, filt_mags = compute_fft(filtered, fs)

        raw_dom = find_dominant_frequencies(raw_freqs, raw_mags)
        filt_dom = find_dominant_frequencies(filt_freqs, filt_mags)

        # ── 7. Build figures ─────────────────────────────────────────
        time_fig = create_time_figure(
            time_arr, signal_arr, filtered, signal_type, filter_label)
        raw_fft_fig = create_fft_figure(
            raw_freqs, raw_mags, signal_type, "Raw Signal FFT",
            dominant_freqs=raw_dom)
        filt_fft_fig = create_fft_figure(
            filt_freqs, filt_mags, signal_type,
            f"Filtered Signal FFT ({filter_label})",
            dominant_freqs=filt_dom)

        # ── 8. Update GUI ────────────────────────────────────────────
        # Signal info panel
        info_text.delete("1.0", tk.END)
        duration = time_arr[-1] - time_arr[0] if len(time_arr) > 1 else 0
        info_text.insert(tk.END, f"File:        {file_path.split('/')[-1]}\n")
        info_text.insert(tk.END, f"Signal Type: {signal_type}\n")
        info_text.insert(tk.END, f"Samples:     {len(signal_arr)}\n")
        info_text.insert(tk.END, f"Duration:    {duration:.3f} s\n")
        info_text.insert(tk.END, f"Sampling Fs: {fs:.2f} Hz\n")
        info_text.insert(tk.END, f"Rows cleaned:{clean_report['rows_removed']}\n")
        info_text.insert(tk.END, f"Filter:      {filter_label}\n")

        # Stats & features panel
        stats_text.delete("1.0", tk.END)
        stats_text.insert(tk.END, "── Preprocessed Signal ──\n")
        for k, v in raw_stats.items():
            stats_text.insert(tk.END, f"  {k}: {v:.6f}\n")

        stats_text.insert(tk.END, "\n── Filtered Signal ──\n")
        for k, v in filt_stats.items():
            stats_text.insert(tk.END, f"  {k}: {v:.6f}\n")

        stats_text.insert(tk.END, f"\n── Features ({signal_type}) ──\n")
        for k, v in filt_features.items():
            if isinstance(v, (int, np.integer)):
                stats_text.insert(tk.END, f"  {k}: {v}\n")
            else:
                stats_text.insert(tk.END, f"  {k}: {v:.4f}\n")

        # Dominant frequencies
        if filt_dom:
            stats_text.insert(tk.END, "\n── Dominant Freq (Filtered) ──\n")
            for d in filt_dom:
                stats_text.insert(
                    tk.END, f"  {d['frequency']:.2f} Hz  (mag {d['magnitude']:.4f})\n")

        # Embed plots
        _embed_figure(time_tab, time_fig)
        _embed_figure(raw_fft_tab, raw_fft_fig)
        _embed_figure(filt_fft_tab, filt_fft_fig)

        messagebox.showinfo("Done", "Analysis completed successfully.")

    except Exception as exc:
        messagebox.showerror("Processing Error", str(exc))
