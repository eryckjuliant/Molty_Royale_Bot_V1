"""Web3 utilities for ERC-8004 on-chain identity management."""

import asyncio
from typing import Optional, Dict, Any
import yaml
import os
from pathlib import Path

try:
    from web3 import Web3
    from web3.contract import Contract
    from web3.exceptions import ContractLogicError, TimeExhausted
    HAS_WEB3 = True
except ImportError:
    HAS_WEB3 = False


# ERC-8004 Identity Registry ABI (simplified for identity checking)
ERC8004_ABI = [
    {
        "inputs": [{"internalType": "address", "name": "owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "uint256", "name": "tokenId", "type": "uint256"}],
        "name": "tokenURI",
        "outputs": [{"internalType": "string", "name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "uint256", "name": "tokenId", "type": "uint256"}],
        "name": "ownerOf",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "address", "name": "from", "type": "address"},
            {"indexed": True, "internalType": "address", "name": "to", "type": "address"},
            {"indexed": True, "internalType": "uint256", "name": "tokenId", "type": "uint256"}
        ],
        "name": "Transfer",
        "type": "event"
    },
    {
        "inputs": [
            {"internalType": "string", "name": "name", "type": "string"},
            {"internalType": "string", "name": "symbol", "type": "string"}
        ],
        "name": "registerIdentity",
        "outputs": [{"internalType": "uint256", "name": "tokenId", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]


class OnChainManager:
    """Manages Web3 connections and ERC-8004 identity operations."""
    
    # Default chain configurations (can be overridden)
    CHAIN_CONFIGS = {
        "ethereum": {
            "rpc_url": "https://eth.llamarpc.com",
            "chain_id": 1,
            "identity_registry": "0x0000000000000000000000000000000000000000"  # Placeholder
        },
        "polygon": {
            "rpc_url": "https://polygon.llamarpc.com",
            "chain_id": 137,
            "identity_registry": "0x0000000000000000000000000000000000000000"  # Placeholder
        },
        "cross_gamechain": {
            "rpc_url": "https://rpc.crosschain.example.com",  # Placeholder
            "chain_id": 8888,
            "identity_registry": "0x0000000000000000000000000000000000000000"  # Placeholder
        }
    }
    
    def __init__(
        self,
        chain: str = "ethereum",
        config_path: str = "config/config.yaml",
        secrets_path: str = "config/secrets.yaml"
    ):
        if not HAS_WEB3:
            raise ImportError(
                "web3.py is not installed. Install with: pip install web3"
            )
        
        self.chain = chain
        self.config_path = config_path
        self.secrets_path = secrets_path
        
        # Web3 connection
        self.w3: Optional[Web3] = None
        self.identity_registry: Optional[Contract] = None
        self.wallet_address: Optional[str] = None
        self.private_key: Optional[str] = None
        
        # Load configuration
        self._load_config()
    
    def _load_config(self) -> None:
        """Load chain and wallet configuration."""
        try:
            # Load config for chain settings
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    chain_config = config.get('onchain', {})
                    
                    # Override default chain config if provided
                    if chain_config.get('rpc_url'):
                        self.CHAIN_CONFIGS[self.chain]['rpc_url'] = chain_config['rpc_url']
                    if chain_config.get('identity_registry'):
                        self.CHAIN_CONFIGS[self.chain]['identity_registry'] = chain_config['identity_registry']
                    if chain_config.get('chain_id'):
                        self.CHAIN_CONFIGS[self.chain]['chain_id'] = chain_config['chain_id']
            
            # Load secrets for wallet
            if os.path.exists(self.secrets_path):
                with open(self.secrets_path, 'r') as f:
                    secrets = yaml.safe_load(f)
                    if secrets:
                        self.wallet_address = secrets.get('wallet_address')
                        self.private_key = secrets.get('wallet_private_key')
        
        except Exception as e:
            print(f"Warning: Could not load onchain config: {e}")
    
    def connect(self) -> bool:
        """Connect to Web3 RPC endpoint."""
        try:
            chain_config = self.CHAIN_CONFIGS.get(self.chain, self.CHAIN_CONFIGS["ethereum"])
            rpc_url = chain_config["rpc_url"]
            
            print(f"Connecting to {self.chain} at {rpc_url}...")
            self.w3 = Web3(Web3.HTTPProvider(rpc_url))
            
            if not self.w3.is_connected():
                print(f"Failed to connect to {self.chain}")
                return False
            
            print(f"✅ Connected to {self.chain}")
            
            # Connect to identity registry contract
            registry_address = chain_config["identity_registry"]
            if registry_address and registry_address != "0x0000000000000000000000000000000000000000":
                self.identity_registry = self.w3.eth.contract(
                    address=registry_address,
                    abi=ERC8004_ABI
                )
                print(f"✅ Connected to Identity Registry at {registry_address}")
            else:
                print(f"⚠️  Identity Registry address not configured")
            
            return True
            
        except Exception as e:
            print(f"Failed to connect to Web3: {e}")
            return False
    
    async def check_erc8004_identity(
        self,
        wallet_address: str,
        agent_name: str
    ) -> Optional[str]:
        """Check if wallet has ERC-8004 agent NFT with given name.
        
        Args:
            wallet_address: Wallet address to check
            agent_name: Agent name to search for
            
        Returns:
            agent_id (tokenId) if found, None otherwise
        """
        if not self.w3 or not self.w3.is_connected():
            if not self.connect():
                print("Cannot check identity: not connected to Web3")
                return None
        
        if not self.identity_registry:
            print("Cannot check identity: Identity Registry not configured")
            return None
        
        try:
            # Normalize address
            wallet_address = self.w3.to_checksum_address(wallet_address)
            
            # Check balance (number of agent NFTs owned)
            balance = self.identity_registry.functions.balanceOf(wallet_address).call()
            
            print(f"Wallet {wallet_address} has {balance} agent NFT(s)")
            
            if balance == 0:
                return None
            
            # Iterate through tokens to find matching agent name
            for token_id in range(balance):
                try:
                    # Get token URI (contains metadata including agent name)
                    token_uri = self.identity_registry.functions.tokenURI(token_id).call()
                    
                    # Parse token URI (typically IPFS or HTTP URL with JSON metadata)
                    if self._check_agent_name_in_metadata(token_uri, agent_name):
                        print(f"✅ Found agent '{agent_name}' with token_id: {token_id}")
                        return str(token_id)
                
                except Exception as e:
                    print(f"Error checking token {token_id}: {e}")
                    continue
            
            print(f"No agent NFT with name '{agent_name}' found")
            return None
            
        except Exception as e:
            print(f"Error checking ERC-8004 identity: {e}")
            return None
    
    def _check_agent_name_in_metadata(self, token_uri: str, agent_name: str) -> bool:
        """Check if token metadata contains the agent name."""
        # This is a simplified check - in production, you would:
        # 1. Fetch the metadata from token_uri (IPFS or HTTP)
        # 2. Parse the JSON metadata
        # 3. Check if the 'name' or 'agent_name' field matches
        
        # For now, just check if agent_name appears in the URI
        return agent_name.lower() in token_uri.lower()
    
    async def register_erc8004(
        self,
        agent_name: str,
        agent_symbol: Optional[str] = None
    ) -> Optional[str]:
        """Register ERC-8004 identity if not already registered.
        
        Args:
            agent_name: Name for the agent
            agent_symbol: Optional symbol for the agent
            
        Returns:
            token_id (agent_id) if successful, None otherwise
        """
        if not self.private_key:
            print("Cannot register identity: private key not found in secrets")
            return None
        
        if not self.w3 or not self.w3.is_connected():
            if not self.connect():
                print("Cannot register identity: not connected to Web3")
                return None
        
        if not self.identity_registry:
            print("Cannot register identity: Identity Registry not configured")
            return None
        
        try:
            # Get wallet address from private key
            account = self.w3.eth.account.from_key(self.private_key)
            wallet_address = account.address
            
            # Check if already registered
            existing_id = await self.check_erc8004_identity(wallet_address, agent_name)
            if existing_id:
                print(f"Agent '{agent_name}' already registered with ID: {existing_id}")
                return existing_id
            
            # Prepare transaction
            symbol = agent_symbol or agent_name[:4].upper()
            
            # Build transaction
            transaction = self.identity_registry.functions.registerIdentity(
                agent_name,
                symbol
            ).build_transaction({
                'from': wallet_address,
                'nonce': self.w3.eth.get_transaction_count(wallet_address),
                'gas': 200000,  # Estimate gas
                'gasPrice': self.w3.eth.gas_price
            })
            
            # Sign transaction
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
            
            # Send transaction
            print(f"Registering agent '{agent_name}' on-chain...")
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for transaction receipt
            print(f"Waiting for transaction confirmation... (tx: {tx_hash.hex()})")
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            if receipt['status'] == 1:
                print(f"✅ Agent registered successfully!")
                print(f"   Transaction hash: {tx_hash.hex()}")
                
                # Get the token ID from the event logs
                for log in receipt['logs']:
                    if log['topics'][0] == self.w3.keccak(text="Transfer(address,address,uint256)"):
                        # Token ID is in the third topic (indexed parameter)
                        token_id = int(log['topics'][2].hex(), 16)
                        print(f"   Agent ID (token_id): {token_id}")
                        return str(token_id)
            else:
                print(f"❌ Transaction failed")
                return None
            
        except ContractLogicError as e:
            print(f"Contract logic error: {e}")
            return None
        except TimeExhausted:
            print("Transaction timed out")
            return None
        except Exception as e:
            print(f"Error registering ERC-8004 identity: {e}")
            return None
    
    def get_wallet_address(self) -> Optional[str]:
        """Get the wallet address from private key or config."""
        if self.wallet_address:
            return self.wallet_address
        
        if self.private_key:
            account = self.w3.eth.account.from_key(self.private_key)
            return account.address
        
        return None
    
    def is_connected(self) -> bool:
        """Check if connected to Web3."""
        return self.w3 and self.w3.is_connected()


# Convenience functions for async usage
async def check_identity(
    wallet_address: str,
    agent_name: str,
    chain: str = "ethereum"
) -> Optional[str]:
    """Convenience function to check ERC-8004 identity."""
    manager = OnChainManager(chain=chain)
    return await manager.check_erc8004_identity(wallet_address, agent_name)


async def register_identity(
    agent_name: str,
    agent_symbol: Optional[str] = None,
    chain: str = "ethereum"
) -> Optional[str]:
    """Convenience function to register ERC-8004 identity."""
    manager = OnChainManager(chain=chain)
    return await manager.register_erc8004(agent_name, agent_symbol)
