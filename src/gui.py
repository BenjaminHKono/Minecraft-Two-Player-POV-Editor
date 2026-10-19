"""Desktop interface for the Minecraft POV editor."""
from __future__ import annotations

import math
import queue
import threading
from pathlib import Path
from typing import Callable

from . import pipeline
from .logging_config import setup_logging


def parse_offset(value: str) -> float | None:
    """Parse the optional Player B offset entered in the desktop interface."""
    if not value.strip():
        return None
    try:
        offset = float(value)
    except ValueError:
        raise ValueError("Enter a number for the Player B offset, or leave it blank.") from None
    if not math.isfinite(offset):
        raise ValueError("The Player B offset must be a finite number.")
    return offset


def run_project(
    player_a: str,
    player_b: str,
    output: str,
    settings: str,
    offset: float | None,
    progress: Callable[[float], None] | None = None,
):
    """Run the shared edit pipeline using paths selected in the desktop interface."""
    project = pipeline.load_project(
        settings or None,
        {"player_a": player_a, "player_b": player_b, "output": output},
    )
    return pipeline.run_all(project, manual_offset=offset, progress=progress)


def launch() -> None:
    """Open the desktop editor window."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    background = "#f2f1eb"
    surface = "#fffefa"
    ink = "#1f2924"
    muted = "#66716b"
    green = "#2e654b"
    queue_events: queue.Queue[tuple[str, object]] = queue.Queue()

    class EditorWindow(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Minecraft POV Editor")
            self.geometry("900x720")
            self.minsize(760, 650)
            self.configure(background=background)

            self.player_a = tk.StringVar()
            self.player_b = tk.StringVar()
            self.output = tk.StringVar(value="output/final.mp4")
            self.settings = tk.StringVar()
            self.offset = tk.StringVar()
            self.status = tk.StringVar(value="Choose both recordings to get started.")

            style = ttk.Style(self)
            style.theme_use("clam")
            style.configure("TFrame", background=background)
            style.configure("Surface.TFrame", background=surface)
            style.configure("TLabel", background=background, foreground=ink, font=("Segoe UI", 10))
            style.configure("Muted.TLabel", background=background, foreground=muted, font=("Segoe UI", 10))
            style.configure("Title.TLabel", background=background, foreground=ink, font=("Segoe UI", 27, "bold"))
            style.configure("Section.TLabel", background=surface, foreground=ink, font=("Segoe UI", 12, "bold"))
            style.configure("Surface.TLabel", background=surface, foreground=ink, font=("Segoe UI", 10))
            style.configure("Field.TEntry", padding=(10, 9), fieldbackground="#ffffff", foreground=ink)
            style.configure("Browse.TButton", padding=(12, 8), background="#e7e8e1", foreground=ink)
            style.map("Browse.TButton", background=[("active", "#daddd4")])
            style.configure("Run.TButton", padding=(18, 11), background=green, foreground="#ffffff",
                            font=("Segoe UI", 11, "bold"))
            style.map("Run.TButton", background=[("active", "#24543d"), ("disabled", "#9eaaa2")])
            style.configure("Horizontal.TProgressbar", troughcolor="#e1e3dc", background=green, thickness=8)

            page = ttk.Frame(self, padding=(42, 34))
            page.pack(fill="both", expand=True)

            ttk.Label(page, text="MINECRAFT  /  VIDEO EDITOR", style="Muted.TLabel").pack(anchor="w")
            ttk.Label(page, text="Bring both views together.", style="Title.TLabel").pack(anchor="w", pady=(8, 5))
            ttk.Label(page, text="Choose two recordings. The editor syncs them and creates one finished video.",
                      style="Muted.TLabel").pack(anchor="w")

            form = ttk.Frame(page, style="Surface.TFrame", padding=(24, 23))
            form.pack(fill="x", pady=(27, 18))
            ttk.Label(form, text="Recordings", style="Section.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 17))
            form.columnconfigure(1, weight=1)

            self._path_row(form, 1, "Player A recording", self.player_a, self._choose_video)
            self._path_row(form, 2, "Player B recording", self.player_b, self._choose_video)
            self._path_row(form, 3, "Save finished video", self.output, self._choose_output)

            ttk.Label(form, text="Player B offset", style="Surface.TLabel").grid(
                row=4, column=0, sticky="w", pady=(17, 5), padx=(0, 18))
            offset_entry = ttk.Entry(form, textvariable=self.offset, style="Field.TEntry", width=19)
            offset_entry.grid(row=4, column=1, sticky="w", pady=(17, 5))
            ttk.Label(form, text="Optional, in seconds. Leave blank to sync from audio.",
                      style="Muted.TLabel").grid(row=5, column=1, sticky="w")
            ttk.Label(form, text="Project settings", style="Surface.TLabel").grid(
                row=6, column=0, sticky="w", pady=(17, 0), padx=(0, 18))
            settings_row = ttk.Frame(form, style="Surface.TFrame")
            settings_row.grid(row=6, column=1, sticky="ew", pady=(17, 0))
            settings_row.columnconfigure(0, weight=1)
            ttk.Entry(settings_row, textvariable=self.settings, style="Field.TEntry").grid(
                row=0, column=0, sticky="ew", padx=(0, 9))
            ttk.Button(settings_row, text="Browse", style="Browse.TButton",
                       command=self._choose_settings).grid(row=0, column=1)

            ttk.Label(page, text="Settings are optional. Use a project JSON to customize speaker mapping and output.",
                      style="Muted.TLabel").pack(anchor="w", pady=(0, 20))
            footer = ttk.Frame(page)
            footer.pack(fill="x", side="bottom")
            self.run_button = ttk.Button(footer, text="Create video", style="Run.TButton", command=self._start)
            self.run_button.pack(side="right")
            self.progress = ttk.Progressbar(page, mode="determinate", maximum=100)
            self.progress.pack(fill="x", side="bottom", pady=(0, 12))
            ttk.Label(page, textvariable=self.status, style="Muted.TLabel").pack(anchor="w", side="bottom", pady=(0, 9))

            self.after(100, self._poll)

        @staticmethod
        def _path_row(parent, row, label, variable, command):
            ttk.Label(parent, text=label, style="Surface.TLabel").grid(
                row=row, column=0, sticky="w", pady=5, padx=(0, 18))
            ttk.Entry(parent, textvariable=variable, style="Field.TEntry").grid(
                row=row, column=1, sticky="ew", pady=5, padx=(0, 9))
            ttk.Button(parent, text="Browse", style="Browse.TButton",
                       command=lambda: command(variable)).grid(row=row, column=2, pady=5)

        def _choose_video(self, variable):
            path = filedialog.askopenfilename(
                title="Choose a recording",
                filetypes=[("Video recordings", "*.mp4 *.mkv *.mov *.avi *.webm"), ("All files", "*.*")],
            )
            if path:
                variable.set(path)

        def _choose_output(self):
            path = filedialog.asksaveasfilename(
                title="Save finished video",
                defaultextension=".mp4",
                initialfile=Path(self.output.get()).name or "final.mp4",
                filetypes=[("MP4 video", "*.mp4"), ("All files", "*.*")],
            )
            if path:
                self.output.set(path)

        def _choose_settings(self):
            path = filedialog.askopenfilename(
                title="Choose project settings",
                filetypes=[("JSON settings", "*.json"), ("All files", "*.*")],
            )
            if path:
                self.settings.set(path)

        def _start(self):
            player_a, player_b = self.player_a.get().strip(), self.player_b.get().strip()
            if not player_a or not Path(player_a).is_file():
                messagebox.showerror("Recording required", "Choose an existing Player A recording.", parent=self)
                return
            if not player_b or not Path(player_b).is_file():
                messagebox.showerror("Recording required", "Choose an existing Player B recording.", parent=self)
                return
            output = self.output.get().strip()
            if not output:
                messagebox.showerror("Output required", "Choose where to save the finished video.", parent=self)
                return
            try:
                offset = parse_offset(self.offset.get())
            except ValueError as exc:
                messagebox.showerror("Check the offset", str(exc), parent=self)
                return

            self.run_button.configure(state="disabled")
            self.progress.configure(value=0)
            self.status.set("Preparing recordings…")
            args = (player_a, player_b, output, self.settings.get().strip(), offset)

            def work():
                try:
                    result = run_project(*args, progress=lambda value: queue_events.put(("progress", value)))
                    queue_events.put(("done", result))
                except Exception as exc:
                    queue_events.put(("error", str(exc)))

            threading.Thread(target=work, daemon=True).start()

        def _poll(self):
            try:
                while True:
                    event, payload = queue_events.get_nowait()
                    if event == "progress":
                        percent = max(0.0, min(100.0, float(payload) * 100))
                        self.progress.configure(value=percent)
                        self.status.set(f"Rendering video  ·  {percent:.0f}%")
                    elif event == "done":
                        output, edl, sync = payload
                        self.progress.configure(value=100)
                        self.status.set(f"Finished  ·  {output}")
                        self.run_button.configure(state="normal")
                        messagebox.showinfo(
                            "Video ready",
                            f"Saved to:\n{output}\n\nSync offset: {sync.offset_b_seconds:+.3f} seconds"
                            f"\nEdit segments: {len(edl.segments)}",
                            parent=self,
                        )
                    else:
                        self.run_button.configure(state="normal")
                        self.status.set("Could not create the video.")
                        messagebox.showerror("Could not create video", str(payload), parent=self)
            except queue.Empty:
                pass
            self.after(100, self._poll)

    setup_logging()
    EditorWindow().mainloop()


if __name__ == "__main__":
    launch()
