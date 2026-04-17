"""Utilitas logging dengan Rich console, file logging, dan penyimpanan status JSON."""

import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme


class GameLogger:
    """Logger dengan output Rich console, file logging, dan penyimpanan status JSON."""
    
    def __init__(
        self,
        log_dir: str = "data/logs",
        log_level: str = "INFO",
        save_raw_json: bool = True,
        console_colors: bool = True
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_raw_json = save_raw_json
        
        # Buat subdirektori
        self.json_dir = self.log_dir / "raw_json"
        if self.save_raw_json:
            self.json_dir.mkdir(parents=True, exist_ok=True)
        
        # Atur Rich console
        self.console = Console(
            theme=Theme({
                "info": "blue",
                "warning": "yellow",
                "error": "red",
                "critical": "bold red",
                "success": "green",
            }) if console_colors else None,
            force_terminal=console_colors
        )
        
        # Atur logger Python dengan handler Rich
        self.logger = logging.getLogger("MoltyRoyaleBot")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.logger.handlers.clear()
        
        # Handler console Rich
        console_handler = RichHandler(
            console=self.console,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            show_time=True,
            show_path=False,
        )
        console_handler.setLevel(getattr(logging, log_level.upper()))
        self.logger.addHandler(console_handler)
        
        # Handler file
        log_file = self.log_dir / f"bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Pelacakan sesi JSON
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_session_dir = self.json_dir / self.session_id
        if self.save_raw_json:
            self.json_session_dir.mkdir(parents=True, exist_ok=True)
        
        self.json_counter = 0
        self.logger.info(f"Logger initialized. Session ID: {self.session_id}")
    
    def info(self, message: str) -> None:
        """Catat pesan info."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Catat pesan peringatan."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Catat pesan error."""
        self.logger.error(message)
    
    def critical(self, message: str) -> None:
        """Catat pesan kritis."""
        self.logger.critical(message)
    
    def debug(self, message: str) -> None:
        """Catat pesan debug."""
        self.logger.debug(message)
    
    def success(self, message: str) -> None:
        """Catat pesan sukses (hanya Rich)."""
        self.console.print(f"[success]✓ {message}[/success]")
        self.logger.info(message)
    
    def print(self, message: str, style: Optional[str] = None) -> None:
        """Cetak pesan ke console saja (tanpa file logging)."""
        self.console.print(message, style=style)
    
    def log_state(self, state: Dict[str, Any], action: Optional[Dict[str, Any]] = None) -> None:
        """Simpan status JSON mentah ke file.
        
        Args:
            state: Status permainan mentah dari API
            action: Aksi opsional yang diambil dengan status ini
        """
        if not self.save_raw_json:
            return
        
        try:
            data = {
                "timestamp": datetime.now().isoformat(),
                "step": self.json_counter,
                "state": state,
            }
            
            if action is not None:
                data["action"] = action
            
            filename = self.json_session_dir / f"state_{self.json_counter:06d}.json"
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.json_counter += 1
            
            # Catat secara berkala untuk menghindari spam
            if self.json_counter % 100 == 0:
                self.debug(f"Saved {self.json_counter} JSON states")
        
        except Exception as e:
            self.error(f"Failed to save JSON state: {e}")
    
    def log_episode(self, episode_data: Dict[str, Any]) -> None:
        """Simpan ringkasan episode ke file.
        
        Args:
            episode_data: Dictionary yang berisi data ringkasan episode
        """
        try:
            filename = self.json_session_dir / f"episode_{episode_data.get('episode_num', 0):04d}.json"
            with open(filename, 'w') as f:
                json.dump(episode_data, f, indent=2, default=str)
            
            self.info(f"Saved episode summary: episode_{episode_data.get('episode_num', 0):04d}")
        
        except Exception as e:
            self.error(f"Failed to save episode summary: {e}")
    
    def log_training_step(self, step: int, metrics: Dict[str, float]) -> None:
        """Catat metrik langkah pelatihan.
        
        Args:
            step: Nomor langkah pelatihan
            metrics: Dictionary nama metrik ke nilai
        """
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(f"Step {step} - {metrics_str}")
    
    def log_reward(self, reward: float, episode: int, step: int) -> None:
        """Catat informasi reward.
        
        Args:
            reward: Nilai reward
            episode: Nomor episode
            step: Nomor langkah dalam episode
        """
        self.debug(f"Episode {episode}, Step {step}: Reward = {reward:.4f}")
    
    def log_action(self, action_type: str, action_details: str) -> None:
        """Catat aksi yang diambil.
        
        Args:
            action_type: Tipe aksi (move, attack, dll.)
            action_details: Detail tentang aksi
        """
        self.debug(f"Action: {action_type} - {action_details}")
    
    def log_error_with_context(self, error: Exception, context: str) -> None:
        """Catat error dengan konteks tambahan.
        
        Args:
            error: Objek Exception
            context: String konteks yang menjelaskan di mana error terjadi
        """
        self.error(f"Error in {context}: {type(error).__name__}: {str(error)}")
        if self.logger.level <= logging.DEBUG:
            import traceback
            self.debug(traceback.format_exc())
    
    def log_model_save(self, model_path: str, episode: int) -> None:
        """Catat event penyimpanan model.
        
        Args:
            model_path: Jalur di mana model disimpan
            episode: Nomor episode
        """
        self.success(f"Model saved at episode {episode}: {model_path}")
    
    def log_evaluation(self, eval_results: Dict[str, Any]) -> None:
        """Catat hasil evaluasi.
        
        Args:
            eval_results: Dictionary yang berisi metrik evaluasi
        """
        self.info("Hasil Evaluasi:")
        for key, value in eval_results.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.4f}")
            else:
                self.info(f"  {key}: {value}")
    
    def create_progress_bar(self, total: int, description: str = "Progress"):
        """Buat progress bar Rich.
        
        Args:
            total: Jumlah total item
            description: Deskripsi untuk progress bar
            
        Returns:
            Objek progress Rich
        """
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
        )
        return progress
    
    def print_table(self, data: list, title: str = "") -> None:
        """Cetak data sebagai tabel Rich.
        
        Args:
            data: List dari list (baris) atau list dari dict
            title: Judul tabel opsional
        """
        from rich.table import Table
        
        if not data:
            self.warning("Tidak ada data untuk ditampilkan dalam tabel")
            return
        
        table = Table(title=title)
        
        if isinstance(data[0], dict):
            # Header dari kunci dict
            for key in data[0].keys():
                table.add_column(str(key))
            
            for row in data:
                table.add_row(*[str(v) for v in row.values()])
        else:
            # Header dari baris pertama
            for i in range(len(data[0])):
                table.add_column(f"Col {i}")
            
            for row in data:
                table.add_row(*[str(v) for v in row])
        
        self.console.print(table)
    
    def close(self) -> None:
        """Tutup logger dan flush semua handler."""
        self.logger.info(f"Session {self.session_id} ended. Total JSON states saved: {self.json_counter}")
        
        # Tutup semua handler
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
