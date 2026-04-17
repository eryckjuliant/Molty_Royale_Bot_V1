"""Titik masuk utama untuk Molty Royale Self-Learning Bot."""

import asyncio
import signal
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from src.game_loop import GameLoop
from src.ml.trainer import AutoTrainer
from src.ml.replay_buffer import ReplayBuffer
from src.utils.logger import GameLogger
from src.api_client import MoltyRoyaleClient


class BotManager:
    """Mengelola GameLoop dan AutoTrainer yang berjalan secara bersamaan."""
    
    def __init__(
        self,
        use_erc8004: bool = False,
        agent_name: str = "MoltyBot",
        register_if_needed: bool = False
    ):
        self.logger = GameLogger()
        self.console = Console()
        self.game_loop = None
        self.trainer = None
        self.replay_buffer = None
        self.use_erc8004 = use_erc8004
        self.agent_name = agent_name
        self.register_if_needed = register_if_needed
        self.shutdown_event = asyncio.Event()
        self.client = None
    
    async def initialize(self):
        """Initialize all components with API key validation."""
        self.logger.info("Menginisialisasi Molty Royale Self-Learning Bot...")
        
        # Langkah 1: Inisialisasi klien API dan validasi kunci API
        await self._setup_api_client()
        
        # Langkah 2: Periksa dan daftarkan identitas ERC-8004 jika diaktifkan
        if self.use_erc8004:
            await self._setup_onchain_identity()
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=100000,
            save_path="data/replay"
        )
        self.logger.success("Buffer replay terinisialisasi")
        
        # Initialize game loop
        self.game_loop = GameLoop(
            agent_type="rule_based",
            turn_duration=60.0,
            use_erc8004=self.use_erc8004
        )
        self.game_loop.set_replay_buffer(self.replay_buffer)
        self.game_loop.set_api_client(self.client)
        self.logger.success("Loop permainan terinisialisasi")
        
        # Initialize auto trainer (background)
        self.trainer = AutoTrainer(
            replay_buffer=self.replay_buffer,
            save_path="data/models",
            log_interval=10
        )
        self.logger.success("Pelatih otomatis terinisialisasi")
    
    async def run_game_loop(self):
        """Jalankan loop permainan di latar depan."""
        try:
            self.logger.info("Memulai loop permainan...")
            await self.game_loop.run(num_episodes=9999, max_steps=1000)
        except Exception as e:
            self.logger.log_error_with_context(e, "run_game_loop")
            self.shutdown_event.set()
    
    async def run_trainer(self):
        """Jalankan pelatih otomatis di latar belakang."""
        try:
            self.logger.info("Memulai pelatih otomatis di latar belakang...")
            await self.trainer.run()
        except Exception as e:
            self.logger.log_error_with_context(e, "run_trainer")
            self.shutdown_event.set()
    
    async def run(self):
        """Jalankan loop permainan dan pelatih secara bersamaan."""
        try:
            # Inisialisasi komponen
            await self.initialize()
            
            # Atur penangan sinyal
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._handle_signal)
            
            # Jalankan loop permainan dan pelatih secara bersamaan
            self.logger.info("Memulai bot (GameLoop + AutoTrainer)...")
            
            await asyncio.gather(
                self.run_game_loop(),
                self.run_trainer(),
                return_exceptions=True
            )
        except Exception as e:
            self.logger.log_error_with_context(e, "run")
        finally:
            await self.shutdown()
    
    def _handle_signal(self):
        """Tangani sinyal shutdown."""
        self.logger.warning("Sinyal shutdown diterima")
        self.shutdown_event.set()
    
    async def shutdown(self):
        """Shutdown yang tertib."""
        self.logger.info("Mematikan bot...")
        
        if self.game_loop:
            await self.game_loop.stop()
        
        if self.trainer:
            await self.trainer.stop()
        
        if self.client:
            await self.client.close()
        
        self.logger.close()
    
    async def _setup_api_client(self):
        """Atur dan validasi klien API."""
        self.console.print(Panel(
            "[bold cyan]Langkah 1: Pengaturan Klien API[/bold cyan]",
            title="Setup Phase",
            border_style="cyan"
        ))
        
        # Inisialisasi klien
        self.client = MoltyRoyaleClient()
        
        # Periksa validitas kunci API
        self.console.print("[yellow]Memeriksa validitas kunci API...[/yellow]")
        
        validation = await self.client.check_api_key_valid()
        
        if validation.get("valid"):
            # Kunci API valid
            agent_name = validation.get('agent_name', 'unknown')
            wallet_address = validation.get('wallet_address', 'Not linked')
            
            self.console.print(Panel(
                f"[bold green]✅ Kunci API Valid[/bold green]\n"
                f"Agent: [cyan]{agent_name}[/cyan]\n"
                f"Wallet: [cyan]{wallet_address}[/cyan]",
                title="API Key Status",
                border_style="green"
            ))
            self.logger.info(f"Kunci API OK, Agent: {agent_name} | Wallet: {wallet_address}")
        else:
            # Kunci API tidak valid atau hilang
            error = validation.get('error', 'Unknown error')
            
            self.console.print(Panel(
                f"[bold red]❌ Kunci API Tidak Valid[/bold red]\n"
                f"Error: [yellow]{error}[/yellow]",
                title="API Key Status",
                border_style="red"
            ))
            
            if self.register_if_needed:
                self.console.print("[yellow]Mencoba pendaftaran otomatis...[/yellow]")
                
                try:
                    # Coba dapatkan alamat wallet dari config
                    wallet_address = self.client.wallet_address if hasattr(self.client, 'wallet_address') else None
                    
                    new_api_key = await self.client.create_account_if_needed(
                        agent_name=self.agent_name,
                        wallet_address=wallet_address,
                        link_onchain=self.use_erc8004
                    )
                    
                    self.console.print(Panel(
                        f"[bold green]✅ Akun Berhasil Terdaftar[/bold green]\n"
                        f"Agent: [cyan]{self.agent_name}[/cyan]\n"
                        f"API Key saved to config/secrets.yaml",
                        title="Pendaftaran Berhasil",
                        border_style="green"
                    ))
                    
                    # Muat ulang klien dengan kunci API baru
                    self.client = MoltyRoyaleClient()
                    
                except Exception as e:
                    self.console.print(Panel(
                        f"[bold red]❌ Pendaftaran Gagal[/bold red]\n"
                        f"Error: [yellow]{str(e)}[/yellow]\n"
                        f"Silakan daftar secara manual di moltyroyale.com",
                        title="Error Pendaftaran",
                        border_style="red"
                    ))
                    raise RuntimeError(f"Gagal mendaftarkan akun: {str(e)}")
            else:
                self.console.print("[yellow]Gunakan --register-if-needed untuk mengaktifkan pendaftaran otomatis[/yellow]")
                raise RuntimeError("Kunci API tidak valid. Silakan daftar secara manual atau gunakan --register-if-needed")
    
    async def _setup_onchain_identity(self):
        """Atur identitas on-chain ERC-8004 jika diaktifkan."""
        self.console.print(Panel(
            "[bold cyan]Langkah 2: Pengaturan Identitas ERC-8004[/bold cyan]",
            title="Setup Phase",
            border_style="cyan"
        ))
        
        try:
            from src.utils.onchain import OnChainManager
            
            onchain_manager = OnChainManager(
                config_path="config/config.yaml",
                secrets_path="config/secrets.yaml"
            )
            
            wallet_address = onchain_manager.get_wallet_address()
            
            if not wallet_address:
                self.console.print(Panel(
                    "[bold yellow]⚠️  Alamat Wallet Tidak Ditemukan[/bold yellow]\n"
                    "Tambahkan wallet_address ke config/secrets.yaml untuk menggunakan ERC-8004",
                    title="On-chain Identity",
                    border_style="yellow"
                ))
                return
            
            self.console.print(f"[yellow]Memeriksa identitas ERC-8004 untuk {wallet_address}...[/yellow]")
            
            existing_id = await onchain_manager.check_erc8004_identity(
                wallet_address,
                self.agent_name
            )
            
            if existing_id:
                self.console.print(Panel(
                    f"[bold green]✅ Identitas ERC-8004 Ditemukan[/bold green]\n"
                    f"ID Agent: [cyan]{existing_id}[/cyan]",
                    title="On-chain Identity",
                    border_style="green"
                ))
                self.logger.info(f"Identitas ERC-8004 sudah terdaftar: {existing_id}")
            else:
                self.console.print(Panel(
                    "[bold yellow]⚠️  Identitas ERC-8004 Tidak Ditemukan[/bold yellow]\n"
                    f"Mencoba mendaftar untuk agent: [cyan]{self.agent_name}[/cyan]...",
                    title="On-chain Identity",
                    border_style="yellow"
                ))
                
                new_id = await onchain_manager.register_erc8004(self.agent_name)
                
                if new_id:
                    self.console.print(Panel(
                        f"[bold green]✅ Identitas ERC-8004 Terdaftar[/bold green]\n"
                        f"ID Agent: [cyan]{new_id}[/cyan]",
                        title="On-chain Identity",
                        border_style="green"
                    ))
                    self.logger.success(f"Identitas ERC-8004 terdaftar: {new_id}")
                else:
                    self.console.print(Panel(
                        "[bold yellow]⚠️  Pendaftaran ERC-8004 Dilewati[/bold yellow]\n"
                        "Mungkin memerlukan pendaftaran manual atau private key",
                        title="On-chain Identity",
                        border_style="yellow"
                    ))
        
        except ImportError:
            self.console.print(Panel(
                "[bold yellow]⚠️  web3.py Tidak Terinstal[/bold yellow]\n"
                "Instal dengan: pip install web3",
                title="On-chain Identity",
                border_style="yellow"
            ))
        except Exception as e:
            self.console.print(Panel(
                f"[bold red]❌ Pengaturan Identitas On-chain Gagal[/bold red]\n"
                f"Error: [yellow]{str(e)}[/yellow]",
                title="On-chain Identity",
                border_style="red"
            ))
            self.logger.warning(f"Pengaturan identitas on-chain gagal: {e}")


async def main():
    """Titik masuk utama."""
    import argparse
    
    console = Console()
    
    # Cetak banner selamat datang
    console.print(Panel(
        "[bold cyan]Molty Royale Self-Learning Bot[/bold cyan]\n"
        "[dim]Agent otonom bertenaga AI dengan pembelajaran berkelanjutan[/dim]",
        title="Selamat Datang",
        border_style="cyan"
    ))
    
    parser = argparse.ArgumentParser(
        description="Molty Royale Self-Learning Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  python main.py                                    # Jalankan dengan pengaturan default
  python main.py --agent-name "BotSaya"           # Atur nama agent kustom
  python main.py --register-if-needed             # Daftar otomatis jika kunci API tidak valid
  python main.py --erc8004 --agent-name "BotSaya" # Aktifkan identitas ERC-8004
"""
    )
    parser.add_argument(
        "--agent-name",
        type=str,
        default="MoltyBot",
        help="Nama agent untuk pendaftaran (default: MoltyBot)"
    )
    parser.add_argument(
        "--register-if-needed",
        action="store_true",
        help="Daftarkan akun baru secara otomatis jika kunci API tidak valid"
    )
    parser.add_argument(
        "--erc8004",
        action="store_true",
        help="Aktifkan pendaftaran identitas on-chain ERC-8004"
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        default="rule_based",
        choices=["rule_based", "rl"],
        help="Tipe agent yang akan digunakan (default: rule_based)"
    )
    
    args = parser.parse_args()
    
    # Buat dan jalankan manajer bot
    manager = BotManager(
        use_erc8004=args.erc8004,
        agent_name=args.agent_name,
        register_if_needed=args.register_if_needed
    )
    
    try:
        await manager.run()
    except KeyboardInterrupt:
        console.print("[yellow]\n⚠️  Dihentikan oleh pengguna[/yellow]")
        manager.logger.info("Dihentikan oleh pengguna")
    except Exception as e:
        console.print(Panel(
            f"[bold red]❌ Error Fatal[/bold red]\n"
            f"Error: [yellow]{str(e)}[/yellow]",
            title="Error",
            border_style="red"
        ))
        manager.logger.log_error_with_context(e, "main")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
