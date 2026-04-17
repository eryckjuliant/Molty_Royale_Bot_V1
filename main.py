"""Main entry point for Molty Royale Self-Learning Bot."""

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
    """Manages GameLoop and AutoTrainer running concurrently."""
    
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
        self.logger.info("Initializing Molty Royale Self-Learning Bot...")
        
        # Step 1: Initialize API client and validate API key
        await self._setup_api_client()
        
        # Step 2: Check and register ERC-8004 identity if enabled
        if self.use_erc8004:
            await self._setup_onchain_identity()
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=100000,
            save_path="data/replay"
        )
        self.logger.success("Replay buffer initialized")
        
        # Initialize game loop
        self.game_loop = GameLoop(
            agent_type="rule_based",
            turn_duration=60.0,
            use_erc8004=self.use_erc8004
        )
        self.game_loop.set_replay_buffer(self.replay_buffer)
        self.game_loop.set_api_client(self.client)
        self.logger.success("Game loop initialized")
        
        # Initialize auto trainer (background)
        self.trainer = AutoTrainer(
            replay_buffer=self.replay_buffer,
            save_path="data/models",
            log_interval=10
        )
        self.logger.success("Auto trainer initialized")
    
    async def run_game_loop(self):
        """Run the game loop in the foreground."""
        try:
            self.logger.info("Starting game loop...")
            await self.game_loop.run(num_episodes=9999, max_steps=1000)
        except Exception as e:
            self.logger.log_error_with_context(e, "run_game_loop")
            self.shutdown_event.set()
    
    async def run_trainer(self):
        """Run the auto trainer in the background."""
        try:
            self.logger.info("Starting background auto trainer...")
            await self.trainer.run()
        except Exception as e:
            self.logger.log_error_with_context(e, "run_trainer")
            self.shutdown_event.set()
    
    async def run(self):
        """Run both game loop and trainer concurrently."""
        try:
            # Initialize components
            await self.initialize()
            
            # Setup signal handlers
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._handle_signal)
            
            # Run game loop and trainer concurrently
            self.logger.info("Starting bot (GameLoop + AutoTrainer)...")
            
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
        """Handle shutdown signals."""
        self.logger.warning("Shutdown signal received")
        self.shutdown_event.set()
    
    async def shutdown(self):
        """Graceful shutdown."""
        self.logger.info("Shutting down bot...")
        
        if self.game_loop:
            await self.game_loop.stop()
        
        if self.trainer:
            await self.trainer.stop()
        
        if self.client:
            await self.client.close()
        
        self.logger.close()
    
    async def _setup_api_client(self):
        """Setup and validate API client."""
        self.console.print(Panel(
            "[bold cyan]Step 1: API Client Setup[/bold cyan]",
            title="Setup Phase",
            border_style="cyan"
        ))
        
        # Initialize client
        self.client = MoltyRoyaleClient()
        
        # Check API key validity
        self.console.print("[yellow]Checking API key validity...[/yellow]")
        
        validation = await self.client.check_api_key_valid()
        
        if validation.get("valid"):
            # API key is valid
            agent_name = validation.get('agent_name', 'unknown')
            wallet_address = validation.get('wallet_address', 'Not linked')
            
            self.console.print(Panel(
                f"[bold green]✅ API Key Valid[/bold green]\n"
                f"Agent: [cyan]{agent_name}[/cyan]\n"
                f"Wallet: [cyan]{wallet_address}[/cyan]",
                title="API Key Status",
                border_style="green"
            ))
            self.logger.info(f"API Key OK, Agent: {agent_name} | Wallet: {wallet_address}")
        else:
            # API key is invalid or missing
            error = validation.get('error', 'Unknown error')
            
            self.console.print(Panel(
                f"[bold red]❌ API Key Invalid[/bold red]\n"
                f"Error: [yellow]{error}[/yellow]",
                title="API Key Status",
                border_style="red"
            ))
            
            if self.register_if_needed:
                self.console.print("[yellow]Attempting automatic registration...[/yellow]")
                
                try:
                    # Try to get wallet address from config
                    wallet_address = self.client.wallet_address if hasattr(self.client, 'wallet_address') else None
                    
                    new_api_key = await self.client.create_account_if_needed(
                        agent_name=self.agent_name,
                        wallet_address=wallet_address,
                        link_onchain=self.use_erc8004
                    )
                    
                    self.console.print(Panel(
                        f"[bold green]✅ Account Registered Successfully[/bold green]\n"
                        f"Agent: [cyan]{self.agent_name}[/cyan]\n"
                        f"API Key saved to config/secrets.yaml",
                        title="Registration Success",
                        border_style="green"
                    ))
                    
                    # Reload client with new API key
                    self.client = MoltyRoyaleClient()
                    
                except Exception as e:
                    self.console.print(Panel(
                        f"[bold red]❌ Registration Failed[/bold red]\n"
                        f"Error: [yellow]{str(e)}[/yellow]\n"
                        f"Please register manually at moltyroyale.com",
                        title="Registration Error",
                        border_style="red"
                    ))
                    raise RuntimeError(f"Failed to register account: {str(e)}")
            else:
                self.console.print("[yellow]Use --register-if-needed to enable automatic registration[/yellow]")
                raise RuntimeError("API key is invalid. Please register manually or use --register-if-needed")
    
    async def _setup_onchain_identity(self):
        """Setup ERC-8004 on-chain identity if enabled."""
        self.console.print(Panel(
            "[bold cyan]Step 2: ERC-8004 Identity Setup[/bold cyan]",
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
                    "[bold yellow]⚠️  No Wallet Address Found[/bold yellow]\n"
                    "Add wallet_address to config/secrets.yaml to use ERC-8004",
                    title="On-chain Identity",
                    border_style="yellow"
                ))
                return
            
            self.console.print(f"[yellow]Checking ERC-8004 identity for {wallet_address}...[/yellow]")
            
            existing_id = await onchain_manager.check_erc8004_identity(
                wallet_address,
                self.agent_name
            )
            
            if existing_id:
                self.console.print(Panel(
                    f"[bold green]✅ ERC-8004 Identity Found[/bold green]\n"
                    f"Agent ID: [cyan]{existing_id}[/cyan]",
                    title="On-chain Identity",
                    border_style="green"
                ))
                self.logger.info(f"ERC-8004 identity already registered: {existing_id}")
            else:
                self.console.print(Panel(
                    "[bold yellow]⚠️  No ERC-8004 Identity Found[/bold yellow]\n"
                    f"Attempting to register for agent: [cyan]{self.agent_name}[/cyan]...",
                    title="On-chain Identity",
                    border_style="yellow"
                ))
                
                new_id = await onchain_manager.register_erc8004(self.agent_name)
                
                if new_id:
                    self.console.print(Panel(
                        f"[bold green]✅ ERC-8004 Identity Registered[/bold green]\n"
                        f"Agent ID: [cyan]{new_id}[/cyan]",
                        title="On-chain Identity",
                        border_style="green"
                    ))
                    self.logger.success(f"ERC-8004 identity registered: {new_id}")
                else:
                    self.console.print(Panel(
                        "[bold yellow]⚠️  ERC-8004 Registration Skipped[/bold yellow]\n"
                        "May require manual registration or private key",
                        title="On-chain Identity",
                        border_style="yellow"
                    ))
        
        except ImportError:
            self.console.print(Panel(
                "[bold yellow]⚠️  web3.py Not Installed[/bold yellow]\n"
                "Install with: pip install web3",
                title="On-chain Identity",
                border_style="yellow"
            ))
        except Exception as e:
            self.console.print(Panel(
                f"[bold red]❌ On-chain Identity Setup Failed[/bold red]\n"
                f"Error: [yellow]{str(e)}[/yellow]",
                title="On-chain Identity",
                border_style="red"
            ))
            self.logger.warning(f"On-chain identity setup failed: {e}")


async def main():
    """Main entry point."""
    import argparse
    
    console = Console()
    
    # Print welcome banner
    console.print(Panel(
        "[bold cyan]Molty Royale Self-Learning Bot[/bold cyan]\n"
        "[dim]AI-powered autonomous agent with continuous learning[/dim]",
        title="Welcome",
        border_style="cyan"
    ))
    
    parser = argparse.ArgumentParser(
        description="Molty Royale Self-Learning Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Run with default settings
  python main.py --agent-name "MyBot"             # Set custom agent name
  python main.py --register-if-needed             # Auto-register if API key invalid
  python main.py --erc8004 --agent-name "MyBot"   # Enable ERC-8004 identity
"""
    )
    parser.add_argument(
        "--agent-name",
        type=str,
        default="MoltyBot",
        help="Agent name for registration (default: MoltyBot)"
    )
    parser.add_argument(
        "--register-if-needed",
        action="store_true",
        help="Automatically register new account if API key is invalid"
    )
    parser.add_argument(
        "--erc8004",
        action="store_true",
        help="Enable ERC-8004 on-chain identity registration"
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        default="rule_based",
        choices=["rule_based", "rl"],
        help="Type of agent to use (default: rule_based)"
    )
    
    args = parser.parse_args()
    
    # Create and run bot manager
    manager = BotManager(
        use_erc8004=args.erc8004,
        agent_name=args.agent_name,
        register_if_needed=args.register_if_needed
    )
    
    try:
        await manager.run()
    except KeyboardInterrupt:
        console.print("[yellow]\n⚠️  Interrupted by user[/yellow]")
        manager.logger.info("Interrupted by user")
    except Exception as e:
        console.print(Panel(
            f"[bold red]❌ Fatal Error[/bold red]\n"
            f"Error: [yellow]{str(e)}[/yellow]",
            title="Error",
            border_style="red"
        ))
        manager.logger.log_error_with_context(e, "main")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
