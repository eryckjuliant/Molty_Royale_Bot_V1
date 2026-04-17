"""Utilitas Web3 untuk manajemen identitas on-chain ERC-8004."""

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
    """Mengelola koneksi Web3 dan operasi identitas ERC-8004."""
    
    # Konfigurasi chain default (dapat ditimpa)
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
        """Muat konfigurasi chain dan wallet."""
        try:
            # Muat config untuk pengaturan chain
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    chain_config = config.get('onchain', {})
                    
                    # Timpa konfigurasi chain default jika disediakan
                    if chain_config.get('rpc_url'):
                        self.CHAIN_CONFIGS[self.chain]['rpc_url'] = chain_config['rpc_url']
                    if chain_config.get('identity_registry'):
                        self.CHAIN_CONFIGS[self.chain]['identity_registry'] = chain_config['identity_registry']
                    if chain_config.get('chain_id'):
                        self.CHAIN_CONFIGS[self.chain]['chain_id'] = chain_config['chain_id']
            
            # Muat secrets untuk wallet
            if os.path.exists(self.secrets_path):
                with open(self.secrets_path, 'r') as f:
                    secrets = yaml.safe_load(f)
                    if secrets:
                        self.wallet_address = secrets.get('wallet_address')
                        self.private_key = secrets.get('wallet_private_key')
        
        except Exception as e:
            print(f"Peringatan: Tidak dapat memuat config onchain: {e}")
    
    def connect(self) -> bool:
        """Hubungkan ke endpoint RPC Web3."""
        try:
            chain_config = self.CHAIN_CONFIGS.get(self.chain, self.CHAIN_CONFIGS["ethereum"])
            rpc_url = chain_config["rpc_url"]
            
            print(f"Menghubungkan ke {self.chain} di {rpc_url}...")
            self.w3 = Web3(Web3.HTTPProvider(rpc_url))
            
            if not self.w3.is_connected():
                print(f"Gagal menghubungkan ke {self.chain}")
                return False
            
            print(f"✅ Terhubung ke {self.chain}")
            
            # Hubungkan ke kontrak registry identitas
            registry_address = chain_config["identity_registry"]
            if registry_address and registry_address != "0x0000000000000000000000000000000000000000":
                self.identity_registry = self.w3.eth.contract(
                    address=registry_address,
                    abi=ERC8004_ABI
                )
                print(f"✅ Terhubung ke Identity Registry di {registry_address}")
            else:
                print(f"⚠️  Alamat Identity Registry tidak dikonfigurasi")
            
            return True
            
        except Exception as e:
            print(f"Gagal menghubungkan ke Web3: {e}")
            return False
    
    async def check_erc8004_identity(
        self,
        wallet_address: str,
        agent_name: str
    ) -> Optional[str]:
        """Periksa apakah wallet memiliki NFT agent ERC-8004 dengan nama yang diberikan.
        
        Args:
            wallet_address: Alamat wallet untuk diperiksa
            agent_name: Nama agent untuk dicari
            
        Returns:
            agent_id (tokenId) jika ditemukan, None jika tidak
        """
        if not self.w3 or not self.w3.is_connected():
            if not self.connect():
                print("Tidak dapat memeriksa identitas: tidak terhubung ke Web3")
                return None
        
        if not self.identity_registry:
            print("Tidak dapat memeriksa identitas: Identity Registry tidak dikonfigurasi")
            return None
        
        try:
            # Normalisasi alamat
            wallet_address = self.w3.to_checksum_address(wallet_address)
            
            # Periksa saldo (jumlah NFT agent yang dimiliki)
            balance = self.identity_registry.functions.balanceOf(wallet_address).call()
            
            print(f"Wallet {wallet_address} memiliki {balance} NFT agent")
            
            if balance == 0:
                return None
            
            # Iterasi melalui token untuk menemukan nama agent yang cocok
            for token_id in range(balance):
                try:
                    # Dapatkan token URI (berisi metadata termasuk nama agent)
                    token_uri = self.identity_registry.functions.tokenURI(token_id).call()
                    
                    # Parse token URI (biasanya IPFS atau URL HTTP dengan metadata JSON)
                    if self._check_agent_name_in_metadata(token_uri, agent_name):
                        print(f"✅ Agent '{agent_name}' ditemukan dengan token_id: {token_id}")
                        return str(token_id)
                
                except Exception as e:
                    print(f"Error checking token {token_id}: {e}")
                    continue
            
            print(f"Tidak ada NFT agent dengan nama '{agent_name}' ditemukan")
            return None
            
        except Exception as e:
            print(f"Error memeriksa identitas ERC-8004: {e}")
            return None
    
    def _check_agent_name_in_metadata(self, token_uri: str, agent_name: str) -> bool:
        """Periksa apakah metadata token berisi nama agent."""
        # Ini adalah pemeriksaan yang disederhanakan - dalam produksi, Anda akan:
        # 1. Mengambil metadata dari token_uri (IPFS atau HTTP)
        # 2. Parse metadata JSON
        # 3. Periksa apakah field 'name' atau 'agent_name' cocok
        
        # Untuk saat ini, periksa saja apakah agent_name muncul di URI
        return agent_name.lower() in token_uri.lower()
    
    async def register_erc8004(
        self,
        agent_name: str,
        agent_symbol: Optional[str] = None
    ) -> Optional[str]:
        """Daftarkan identitas ERC-8004 jika belum terdaftar.
        
        Args:
            agent_name: Nama untuk agent
            agent_symbol: Simbol opsional untuk agent
            
        Returns:
            token_id (agent_id) jika berhasil, None jika tidak
        """
        if not self.private_key:
            print("Tidak dapat mendaftarkan identitas: private key tidak ditemukan di secrets")
            return None
        
        if not self.w3 or not self.w3.is_connected():
            if not self.connect():
                print("Tidak dapat mendaftarkan identitas: tidak terhubung ke Web3")
                return None
        
        if not self.identity_registry:
            print("Tidak dapat mendaftarkan identitas: Identity Registry tidak dikonfigurasi")
            return None
        
        try:
            # Dapatkan alamat wallet dari private key
            account = self.w3.eth.account.from_key(self.private_key)
            wallet_address = account.address
            
            # Periksa apakah sudah terdaftar
            existing_id = await self.check_erc8004_identity(wallet_address, agent_name)
            if existing_id:
                print(f"Agent '{agent_name}' sudah terdaftar dengan ID: {existing_id}")
                return existing_id
            
            # Siapkan transaksi
            symbol = agent_symbol or agent_name[:4].upper()
            
            # Bangun transaksi
            transaction = self.identity_registry.functions.registerIdentity(
                agent_name,
                symbol
            ).build_transaction({
                'from': wallet_address,
                'nonce': self.w3.eth.get_transaction_count(wallet_address),
                'gas': 200000,  # Perkiraan gas
                'gasPrice': self.w3.eth.gas_price
            })
            
            # Tandatangani transaksi
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
            
            # Kirim transaksi
            print(f"Mendaftarkan agent '{agent_name}' on-chain...")
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Tunggu bukti transaksi
            print(f"Menunggu konfirmasi transaksi... (tx: {tx_hash.hex()})")
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            if receipt['status'] == 1:
                print(f"✅ Agent berhasil terdaftar!")
                print(f"   Transaction hash: {tx_hash.hex()}")
                
                # Dapatkan ID token dari log event
                for log in receipt['logs']:
                    if log['topics'][0] == self.w3.keccak(text="Transfer(address,address,uint256)"):
                        # ID token ada di topik ketiga (parameter terindeks)
                        token_id = int(log['topics'][2].hex(), 16)
                        print(f"   Agent ID (token_id): {token_id}")
                        return str(token_id)
            else:
                print(f"❌ Transaksi gagal")
                return None
            
        except ContractLogicError as e:
            print(f"Error logika kontrak: {e}")
            return None
        except TimeExhausted:
            print("Transaksi timeout")
            return None
        except Exception as e:
            print(f"Error mendaftarkan identitas ERC-8004: {e}")
            return None
    
    def get_wallet_address(self) -> Optional[str]:
        """Dapatkan alamat wallet dari private key atau config."""
        if self.wallet_address:
            return self.wallet_address
        
        if self.private_key:
            account = self.w3.eth.account.from_key(self.private_key)
            return account.address
        
        return None
    
    def is_connected(self) -> bool:
        """Periksa apakah terhubung ke Web3."""
        return self.w3 and self.w3.is_connected()
    
    async def verify_wallet_ownership(
        self,
        wallet_address: str,
        agent_name: str,
        private_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Verifikasi kepemilikan wallet untuk ERC-8004 agent identity.
        
        Args:
            wallet_address: Alamat wallet yang akan diverifikasi
            agent_name: Nama agent yang dicari
            private_key: Private key opsional untuk sign message
            
        Returns:
            Dict dengan status verifikasi, signature (jika private_key disediakan), dan info agent
        """
        result = {
            "verified": False,
            "has_identity": False,
            "agent_id": None,
            "signature": None,
            "message": None,
            "error": None
        }
        
        try:
            # Pastikan terhubung ke Web3
            if not self.w3 or not self.w3.is_connected():
                if not self.connect():
                    result["error"] = "Tidak dapat terhubung ke Web3"
                    return result
            
            # Normalisasi alamat wallet
            wallet_address = self.w3.to_checksum_address(wallet_address)
            
            # Periksa apakah wallet memiliki identitas ERC-8004 dengan nama agent
            if self.identity_registry:
                existing_id = await self.check_erc8004_identity(wallet_address, agent_name)
                
                if existing_id:
                    result["has_identity"] = True
                    result["agent_id"] = existing_id
                    result["verified"] = True
                    print(f"✅ Wallet {wallet_address} memiliki identitas agent '{agent_name}' (ID: {existing_id})")
                else:
                    result["error"] = f"Wallet tidak memiliki identitas agent '{agent_name}'"
                    print(f"⚠️  Wallet {wallet_address} tidak memiliki identitas agent '{agent_name}'")
            else:
                result["error"] = "Identity Registry tidak dikonfigurasi"
                print("⚠️  Identity Registry tidak dikonfigurasi")
            
            # Jika private_key disediakan, sign message untuk verifikasi
            if private_key and result["verified"]:
                try:
                    message = f"Recover API Key for agent {agent_name}"
                    encoded_message = self.w3.eth.account.messages.encode_defunct(text=message)
                    signed_message = self.w3.eth.account.sign_message(encoded_message, private_key=private_key)
                    
                    result["message"] = message
                    result["signature"] = signed_message.signature.hex()
                    result["address"] = signed_message.recover_address
                    
                    # Verifikasi bahwa signature cocok dengan wallet address
                    if self.w3.to_checksum_address(signed_message.recover_address) == wallet_address:
                        print(f"✅ Signature berhasil diverifikasi untuk wallet {wallet_address}")
                    else:
                        print(f"⚠️  Signature tidak cocok dengan wallet address")
                        result["verified"] = False
                        
                except Exception as e:
                    print(f"⚠️  Gagal sign message: {e}")
                    result["error"] = f"Gagal sign message: {str(e)}"
            
            return result
            
        except Exception as e:
            result["error"] = f"Error verifikasi kepemilikan wallet: {str(e)}"
            print(f"❌ Error verifikasi kepemilikan wallet: {e}")
            return result


# Fungsi kenyamanan untuk penggunaan async
async def check_identity(
    wallet_address: str,
    agent_name: str,
    chain: str = "ethereum"
) -> Optional[str]:
    """Fungsi kenyamanan untuk memeriksa identitas ERC-8004."""
    manager = OnChainManager(chain=chain)
    return await manager.check_erc8004_identity(wallet_address, agent_name)


async def register_identity(
    agent_name: str,
    agent_symbol: Optional[str] = None,
    chain: str = "ethereum"
) -> Optional[str]:
    """Fungsi kenyamanan untuk mendaftarkan identitas ERC-8004."""
    manager = OnChainManager(chain=chain)
    return await manager.register_erc8004(agent_name, agent_symbol)
