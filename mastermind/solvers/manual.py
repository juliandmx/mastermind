import os
import threading
import tkinter as tk
from tkinter import messagebox
import numpy as np
from .base import AbstractSolver

class Manual(AbstractSolver):
    """Grafische Version des manuellen Solvers mit farbigen Kreisen in der Historie.
    Öffnet automatisch eine GUI; Fenster schließen beendet das gesamte Programm.
    """
    def __init__(self, n: int, k: int):
        super().__init__(n, k)
        self._guess_ready = threading.Event()
        self._guess = None
        self._root = None

        # GUI in eigenem Thread starten (blockiert Hauptthread nicht)
        self._gui_thread = threading.Thread(target=self._start_gui, daemon=True)
        self._gui_thread.start()

    # ---------- GUI-Grundaufbau ----------
    def _start_gui(self):
        self._root = tk.Tk()
        self._root.title("Mastermind – Manual Solver")
        self._root.geometry("560x420")
        self._root.minsize(520, 380)

        header = tk.Frame(self._root)
        header.pack(fill="x", padx=10, pady=(10, 6))
        tk.Label(
            header,
            text=f"n={self.n}, Farben 0..{self.k-1}",
            font=("Segoe UI", 10, "bold")
        ).pack(side="left")

        # History: Scrollbarer Canvas, in dem wir Reihen mit Kreisen zeichnen
        hist_wrap = tk.Frame(self._root)
        hist_wrap.pack(fill="both", expand=True, padx=10, pady=6)

        self._hist_canvas = tk.Canvas(hist_wrap, bg="#fafafa", highlightthickness=0)
        self._hist_vsb = tk.Scrollbar(hist_wrap, orient="vertical", command=self._hist_canvas.yview)
        self._hist_canvas.configure(yscrollcommand=self._hist_vsb.set)

        self._hist_vsb.pack(side="right", fill="y")
        self._hist_canvas.pack(side="left", fill="both", expand=True)

        # Ein internes Frame, das der Canvas scrollt
        self._hist_frame = tk.Frame(self._hist_canvas, bg="#fafafa")
        self._hist_window = self._hist_canvas.create_window((0, 0), window=self._hist_frame, anchor="nw")

        self._hist_frame.bind("<Configure>", self._on_hist_configure)
        self._hist_canvas.bind("<Configure>", self._on_canvas_configure)

        # Eingabezeile
        input_bar = tk.Frame(self._root)
        input_bar.pack(fill="x", padx=10, pady=(6, 10))

        tk.Label(input_bar, text="Dein Tipp:").pack(side="left")
        self.entry = tk.Entry(input_bar, font=("Consolas", 12))
        self.entry.pack(side="left", fill="x", expand=True, padx=6)
        self.entry.bind("<Return>", lambda e: self._on_submit())

        tk.Button(input_bar, text="Absenden", command=self._on_submit).pack(side="left")

        self._status = tk.StringVar(value="Bereit.")
        tk.Label(self._root, textvariable=self._status, anchor="w").pack(fill="x", padx=10)

        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root.mainloop()

    def _on_hist_configure(self, event):
        # Scrollregion an Inhalt anpassen
        self._hist_canvas.configure(scrollregion=self._hist_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        # Breite des inneren Frames an Canvas-Breite anpassen
        self._hist_canvas.itemconfig(self._hist_window, width=event.width)

    # ---------- Farbpalette ----------
    def _color_for(self, v: int) -> str:
        """Gibt für eine Zahl 0..k-1 eine Tk-Farbe (#RRGGBB) zurück."""
        base = [
            "#2E86AB", "#F6A01A", "#E64C3C", "#7D3C98", "#27AE60", "#C0392B",
            "#16A085", "#8E44AD", "#D35400", "#2C3E50", "#95A5A6", "#F39C12",
            "#1ABC9C", "#9B59B6", "#3498DB", "#E67E22"
        ]
        if v < len(base):
            return base[v]
        # Wenn k größer ist, generiere weitere Farben gleichmäßig im HSV-Farbraum
        # Einfache Approximation ohne Import von colorsys
        hue = (v * 137) % 360  # verteilte Hügelzahl
        return self._hsv_to_hex(hue / 360.0, 0.65, 0.95)

    def _hsv_to_hex(self, h, s, v):
        i = int(h * 6)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        i = i % 6
        if i == 0: r, g, b = v, t, p
        elif i == 1: r, g, b = q, v, p
        elif i == 2: r, g, b = p, v, t
        elif i == 3: r, g, b = p, q, v
        elif i == 4: r, g, b = t, p, v
        else: r, g, b = v, p, q
        return "#%02x%02x%02x" % (int(r*255), int(g*255), int(b*255))

    # ---------- Eingabe / Steuerung ----------
    def _on_submit(self):
        text = self.entry.get().strip()
        try:
            # Erlaube "0123" und "0 1 2 3" / "0,1,2,3"
            if " " in text or "," in text:
                parts = [p for p in text.replace(",", " ").split() if p]
                vals = [int(p) for p in parts]
            else:
                vals = [int(ch) for ch in text if ch.isdigit()]
            if len(vals) != self.n:
                raise ValueError(f"Bitte genau {self.n} Ziffern angeben/eingeben.")
            if any(v < 0 or v >= self.k for v in vals):
                raise ValueError(f"Zahlen müssen in 0..{self.k-1} liegen.")
            self._guess = np.asarray(vals, dtype=np.int16)
            self.entry.delete(0, tk.END)
            self._status.set(f"Tipp eingereicht: {self._guess.tolist()}")
            self._guess_ready.set()
        except Exception as e:
            messagebox.showerror("Ungültiger Tipp", str(e))

    def _on_close(self):
        if messagebox.askokcancel("Beenden", "Möchtest du das Programm wirklich beenden?"):
            os._exit(0)  # beendet auch run_bench zuverlässig

    # ---------- Historie zeichnen ----------
    def _append_history_row(self, guess: np.ndarray, b: int, w: int):
        """Thread-sicher: zeichnet eine Zeile mit Kreisen (Guess) und Feedback."""
        if not self._root:
            return
        self._root.after(0, lambda: self._draw_history_row(guess, b, w))

    def _draw_history_row(self, guess: np.ndarray, b: int, w: int):
        """Draw a row showing guess as colored circles and feedback as small black/white dots."""
        row = tk.Frame(self._hist_frame, bg="#fafafa")
        row.pack(fill="x", padx=8, pady=4)

        circle_size = 28
        pad = 6
        feedback_dot = 10  # smaller feedback peg size

        # main guess circles
        guess_canvas = tk.Canvas(row,
                                 width=self.n * (circle_size + pad) + pad,
                                 height=circle_size + pad * 2,
                                 bg="#fafafa", highlightthickness=0)
        guess_canvas.pack(side="left")

        x = pad
        y = pad
        for val in guess:
            color = self._color_for(int(val))
            guess_canvas.create_oval(x, y, x + circle_size, y + circle_size,
                                     fill=color, outline="#202020", width=1)
            cx = x + circle_size / 2
            cy = y + circle_size / 2
            guess_canvas.create_text(cx, cy, text=str(int(val)),
                                     fill="white", font=("Segoe UI", 11, "bold"))
            x += circle_size + pad

        # feedback pegs (small black/white circles)
        fb_frame = tk.Frame(row, bg="#fafafa")
        fb_frame.pack(side="left", padx=10)

        fb_canvas = tk.Canvas(fb_frame, width=48, height=32,
                              bg="#fafafa", highlightthickness=0)
        fb_canvas.pack()

        # draw black first, then white
        total = b + w
        cols = 2
        spacing = 12
        for i in range(b):
            cx = 8 + (i % cols) * spacing
            cy = 8 + (i // cols) * spacing
            fb_canvas.create_oval(cx, cy, cx + feedback_dot, cy + feedback_dot,
                                  fill="black", outline="black")
        for i in range(w):
            cx = 8 + ((i + b) % cols) * spacing
            cy = 8 + ((i + b) // cols) * spacing
            fb_canvas.create_oval(cx, cy, cx + feedback_dot, cy + feedback_dot,
                                  fill="white", outline="black")

        # auto-scroll to the bottom
        self._hist_canvas.update_idletasks()
        self._hist_canvas.yview_moveto(1.0)

    # ---------- Solver-Interface ----------
    def next_guess(self, history, candidates: np.ndarray) -> np.ndarray:
        """Zeigt das letzte Feedback als farbige Zeile und wartet auf den nächsten Tipp."""
        if history and self._root:
            last_guess, (b, w) = history[-1]
            self._append_history_row(last_guess, b, w)

        self._guess_ready.clear()
        self._guess = None
        self._guess_ready.wait()  # blockiert, bis der Nutzer sendet
        return self._guess
