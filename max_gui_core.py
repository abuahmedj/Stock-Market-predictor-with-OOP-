import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from typing import Callable, Optional
from max_config import MaxConfig

class MaxInterface(ctk.CTk):
    """
    The main GUI Core for Max Engine.
    Inherits from customtkinter.CTk and provides layout and observer hooks.
    """
    def __init__(self, on_analyze_callback: Callable[[str], None]):
        super().__init__()
        self.config = MaxConfig()
        self.on_analyze_callback = on_analyze_callback
        
        # Window setup
        self.title(self.config.WINDOW_TITLE)
        self.geometry(self.config.WINDOW_SIZE)
        ctk.set_appearance_mode("dark")
        
        # Observer Pattern / Callbacks
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self._build_layout()

    def _build_layout(self):
        # Grid Configuration
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # --- Sidebar ---
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0, fg_color=self.config.COLOR_BG_SIDEBAR)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Max Engine", font=ctk.CTkFont(size=24, weight="bold"), text_color=self.config.COLOR_ACCENT_GOLD)
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.ticker_entry = ctk.CTkEntry(self.sidebar_frame, placeholder_text="Enter Ticker (e.g. AAPL)")
        self.ticker_entry.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        
        self.analyze_btn = ctk.CTkButton(self.sidebar_frame, text="Analyze", command=self._handle_analyze, fg_color=self.config.COLOR_ACCENT_GOLD, text_color="black", hover_color="#B8962E")
        self.analyze_btn.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        
        # --- Main Dashboard ---
        self.main_frame = ctk.CTkFrame(self, fg_color=self.config.COLOR_BG_MAIN)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_columnconfigure((0, 1, 2), weight=1)
        self.main_frame.grid_rowconfigure(1, weight=3)
        self.main_frame.grid_rowconfigure(2, weight=1)
        
        # Metric Cards
        self.price_card = self._create_metric_card(self.main_frame, "Current Price", "---", 0, 0)
        self.pred_card = self._create_metric_card(self.main_frame, "Predicted Close", "---", 0, 1)
        self.sentiment_card = self._create_metric_card(self.main_frame, "NLP Sentiment", "---", 0, 2)
        
        # Chart Frame
        self.chart_frame = ctk.CTkFrame(self.main_frame, fg_color=self.config.COLOR_CARD_BG)
        self.chart_frame.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=10)
        self._fig, self._ax = plt.subplots(figsize=(8, 4), facecolor=self.config.COLOR_CARD_BG)
        self._ax.set_facecolor(self.config.COLOR_CARD_BG)
        self._canvas = FigureCanvasTkAgg(self._fig, master=self.chart_frame)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Console Frame
        self.console_frame = ctk.CTkFrame(self.main_frame, fg_color=self.config.COLOR_CARD_BG)
        self.console_frame.grid(row=2, column=0, columnspan=3, sticky="nsew", pady=10)
        
        self.console_text = ctk.CTkTextbox(self.console_frame, text_color=self.config.COLOR_TEXT_PRIMARY, fg_color=self.config.COLOR_CARD_BG)
        self.console_text.pack(fill="both", expand=True, padx=10, pady=10)

    def _create_metric_card(self, parent, title: str, value: str, row: int, col: int):
        card = ctk.CTkFrame(parent, fg_color=self.config.COLOR_CARD_BG, corner_radius=10)
        card.grid(row=row, column=col, sticky="nsew", padx=10, pady=10)
        
        lbl_title = ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=14), text_color=self.config.COLOR_TEXT_SECONDARY)
        lbl_title.pack(pady=(15, 0))
        
        lbl_val = ctk.CTkLabel(card, text=value, font=ctk.CTkFont(size=24, weight="bold"), text_color=self.config.COLOR_ACCENT_GOLD)
        lbl_val.pack(pady=(5, 15))
        return lbl_val

    def _handle_analyze(self):
        ticker = self.ticker_entry.get().strip().upper()
        if ticker:
            self.analyze_btn.configure(state="disabled")
            self.on_analyze_callback(ticker)

    def update_metrics(self, price: str, prediction: str, sentiment: str):
        self.price_card.configure(text=price)
        self.pred_card.configure(text=prediction)
        self.sentiment_card.configure(text=sentiment)
        self.analyze_btn.configure(state="normal")

    def update_chart(self, data: pd.DataFrame, ticker: str):
        self._ax.clear()
        self._ax.plot(data.index, data['Close'], color=self.config.COLOR_ACCENT_GOLD)
        self._ax.set_title(f"{ticker} History", color=self.config.COLOR_TEXT_PRIMARY)
        self._ax.tick_params(colors=self.config.COLOR_TEXT_SECONDARY)
        for spine in self._ax.spines.values():
            spine.set_color(self.config.COLOR_TEXT_SECONDARY)
        self._fig.tight_layout()
        self._canvas.draw()

    def log_message(self, message: str):
        """Thread-safe method called via Orchestrator to update the console."""
        self.console_text.insert("end", f"{message}\n")
        self.console_text.see("end")

    def on_close(self):
        self.quit()
        self.destroy()
