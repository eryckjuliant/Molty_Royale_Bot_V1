import asyncio
import time
from typing import Optional, Dict, Any, List
import httpx
from httpx import AsyncClient, Response, HTTPStatusError, RequestError
import yaml
import os


class MoltyRoyaleClient:
    """Klien HTTP asinkron untuk API Molty Royale dengan retry otomatis dan penanganan rate limit."""
    
    def __init__(self, config_path: str = "config/config.yaml", secrets_path: str = "config/secrets.yaml"):
        self.base_url: str = ""
        self.api_key: str = ""
        self.game_id: str = ""
        self.max_games: int = 9999
        self.ml_training_interval: int = 20
        self._client: Optional[AsyncClient] = None
        self.config_path = config_path
        self.secrets_path = secrets_path
        self._load_config(config_path, secrets_path)
        
        # Rate limiting settings
        self._rate_limit_delay: float = 0.1  # 100ms between requests
        self._last_request_time: float = 0
        self._max_retries: int = 3
        self._retry_delay: float = 1.0
        self._timeout: float = 10.0
    
    def _load_config(self, config_path: str, secrets_path: str) -> None:
        """Muat konfigurasi dari file YAML dan variabel lingkungan."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.base_url = config.get('api_base', 'https://api.moltyroyale.com')
                self.game_id = config.get('game_id', 'default_free')
                self.api_key = config.get('api_key', '')
                self.max_games = config.get('max_games', 9999)
                self.ml_training_interval = config.get('ml_training_interval', 20)
            
            # Override API key from secrets if available
            if os.path.exists(secrets_path):
                with open(secrets_path, 'r') as f:
                    secrets = yaml.safe_load(f)
                    if secrets and secrets.get('api_key'):
                        self.api_key = secrets['api_key']
            
            # Override API key from environment variable if set (highest priority)
            env_api_key = os.getenv("MOLTY_API_KEY")
            if env_api_key:
                self.api_key = env_api_key
        except FileNotFoundError as e:
            raise RuntimeError(f"File konfigurasi tidak ditemukan: {e}")
        except yaml.YAMLError as e:
            raise RuntimeError(f"YAML tidak valid dalam konfigurasi: {e}")
    
    async def __aenter__(self):
        """Masuk manajer konteks asinkron."""
        await self._ensure_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Keluar manajer konteks asinkron."""
        await self.close()
    
    async def _ensure_client(self) -> None:
        """Pastikan klien HTTP terinisialisasi."""
        if self._client is None or self._client.is_closed:
            self._client = AsyncClient(
                base_url=self.base_url,
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": self.api_key,
                },
                timeout=self._timeout,
            )
    
    async def close(self) -> None:
        """Tutup klien HTTP."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    async def _rate_limit(self) -> None:
        """Terapkan rate limiting antara permintaan."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self._rate_limit_delay:
            await asyncio.sleep(self._rate_limit_delay - time_since_last)
        self._last_request_time = time.time()
    
    async def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Response:
        """Buat permintaan HTTP dengan retry otomatis saat gagal."""
        await self._ensure_client()
        
        last_error = None
        for attempt in range(self._max_retries):
            try:
                await self._rate_limit()
                response = await self._client.request(method, endpoint, **kwargs)
                response.raise_for_status()
                return response
            except HTTPStatusError as e:
                last_error = e
                if e.response.status_code in (429, 502, 503, 504):
                    # Rate limit atau error server - retry dengan backoff
                    await asyncio.sleep(self._retry_delay * (2 ** attempt))
                else:
                    # Error HTTP lainnya - jangan retry
                    raise
            except RequestError as e:
                last_error = e
                await asyncio.sleep(self._retry_delay * (2 ** attempt))
        
        raise RuntimeError(f"Maksimum retry terlampaui: {last_error}")
    
    async def create_account(self, username: str, email: str) -> Dict[str, Any]:
        """Buat akun Molty Royale baru.
        
        Args:
            username: Username yang diinginkan
            email: Alamat email
            
        Returns:
            Respons pembuatan akun dengan user_id dan kredensial awal
        """
        response = await self._request_with_retry(
            "POST",
            "/v1/auth/create-account",
            json={
                "username": username,
                "email": email,
            }
        )
        return response.json()
    
    async def register_agent(
        self,
        agent_name: str,
        agent_type: str = "rl",
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Daftarkan agent AI baru untuk permainan.
        
        Args:
            agent_name: Nama agent
            agent_type: Tipe agent (rl, rule_based, hybrid)
            description: Deskripsi agent opsional
            
        Returns:
            Respons pendaftaran agent dengan agent_id
        """
        response = await self._request_with_retry(
            "POST",
            "/v1/agents/register",
            json={
                "name": agent_name,
                "type": agent_type,
                "description": description or "",
                "game_id": self.game_id,
            }
        )
        return response.json()
    
    async def get_state(self, agent_id: str, game_session_id: Optional[str] = None) -> Dict[str, Any]:
        """Dapatkan status permainan saat ini untuk agent.
        
        Args:
            agent_id: Identifier agent
            game_session_id: ID sesi permainan spesifik opsional
            
        Returns:
            Status permainan saat ini dengan observasi, skor, dan metadata
        """
        params = {"agent_id": agent_id}
        if game_session_id:
            params["session_id"] = game_session_id
        
        response = await self._request_with_retry(
            "GET",
            "/v1/game/state",
            params=params
        )
        return response.json()
    
    async def send_action(
        self,
        agent_id: str,
        action: Dict[str, Any],
        game_session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Kirim aksi ke mesin permainan.
        
        Args:
            agent_id: Identifier agent
            action: Dictionary aksi dengan action_type dan parameter
            game_session_id: ID sesi permainan spesifik opsional
            
        Returns:
            Respons aksi dengan status baru, reward, dan flag done
        """
        params = {"agent_id": agent_id}
        if game_session_id:
            params["session_id"] = game_session_id
        
        response = await self._request_with_retry(
            "POST",
            "/v1/game/action",
            params=params,
            json=action
        )
        return response.json()
    
    async def get_game_list(self) -> List[Dict[str, Any]]:
        """Dapatkan daftar mode/konfigurasi permainan yang tersedia.
        
        Returns:
            Daftar permainan yang tersedia dengan metadata mereka
        """
        response = await self._request_with_retry(
            "GET",
            "/v1/games/list"
        )
        return response.json()
    
    async def join_game(
        self,
        game_id: str,
        agent_id: str,
        mode: str = "standard"
    ) -> Dict[str, Any]:
        """Bergabung dengan instance permainan spesifik.
        
        Args:
            game_id: Identifier permainan
            agent_id: Identifier agent
            mode: Mode permainan (standard, ranked, practice)
            
        Returns:
            Informasi sesi permainan dengan session_id
        """
        response = await self._request_with_retry(
            "POST",
            "/v1/game/join",
            json={
                "game_id": game_id,
                "agent_id": agent_id,
                "mode": mode,
            }
        )
        return response.json()
    
    async def leave_game(
        self,
        session_id: str,
        agent_id: str
    ) -> Dict[str, Any]:
        """Tinggalkan sesi permainan.
        
        Args:
            session_id: Identifier sesi permainan
            agent_id: Identifier agent
            
        Returns:
            Respons konfirmasi keluar
        """
        response = await self._request_with_retry(
            "POST",
            "/v1/game/leave",
            json={
                "session_id": session_id,
                "agent_id": agent_id,
            }
        )
        return response.json()
    
    async def get_leaderboard(self, game_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Dapatkan papan peringkat untuk permainan.
        
        Args:
            game_id: Identifier permainan opsional (menggunakan game_id saat ini jika tidak disediakan)
            limit: Jumlah maksimum entri yang akan dikembalikan
            
        Returns:
            Entri papan peringkat dengan peringkat dan skor
        """
        params = {"limit": limit}
        if game_id:
            params["game_id"] = game_id
        else:
            params["game_id"] = self.game_id
        
        response = await self._request_with_retry(
            "GET",
            "/v1/game/leaderboard",
            params=params
        )
        return response.json()
    
    async def get_agent_stats(self, agent_id: str) -> Dict[str, Any]:
        """Dapatkan statistik untuk agent spesifik.
        
        Args:
            agent_id: Identifier agent
            
        Returns:
            Statistik agent termasuk permainan yang dimainkan, tingkat kemenangan, dll.
        """
        response = await self._request_with_retry(
            "GET",
            f"/v1/agents/{agent_id}/stats"
        )
        return response.json()
    
    async def upload_model(
        self,
        agent_id: str,
        model_path: str,
        model_version: str
    ) -> Dict[str, Any]:
        """Unggah model yang dilatih untuk agent.
        
        Args:
            agent_id: Identifier agent
            model_path: Jalur ke file model
            model_version: String versi untuk model
            
        Returns:
            Konfirmasi unggahan model dengan model_id
        """
        with open(model_path, 'rb') as f:
            files = {"model": f}
            data = {
                "agent_id": agent_id,
                "version": model_version,
            }
            response = await self._request_with_retry(
                "POST",
                "/v1/agents/model/upload",
                data=data,
                files=files
            )
        return response.json()
    
    async def get_replay_data(self, session_id: str) -> Dict[str, Any]:
        """Unduh data replay untuk sesi permainan.
        
        Args:
            session_id: Identifier sesi permainan
            
        Returns:
            Data replay dengan semua aksi dan status dari sesi
        """
        response = await self._request_with_retry(
            "GET",
            f"/v1/game/replay/{session_id}"
        )
        return response.json()
    
    async def check_api_key_valid(self) -> Dict[str, Any]:
        """Periksa apakah kunci API saat ini valid.
        
        Returns:
            Dict dengan boolean 'valid' dan info agent jika valid, atau error jika tidak valid.
            Mengembalikan {"valid": False, "error": "..."} pada error 401/403.
        """
        try:
            await self._ensure_client()
            
            # Coba beberapa endpoint untuk validasi kunci API
            endpoints = ["/v1/me", "/v1/profile", "/v1/agent/status"]
            
            last_error = None
            for endpoint in endpoints:
                for attempt in range(self._max_retries):
                    try:
                        await self._rate_limit()
                        response = await self._client.get(endpoint)
                        
                        if response.status_code == 200:
                            data = response.json()
                            return {
                                "valid": True,
                                "agent_name": data.get("agent_name") or data.get("name"),
                                "agent_id": data.get("agent_id") or data.get("id"),
                                "wallet_address": data.get("wallet_address"),
                                "status": data.get("status"),
                                "reputation": data.get("reputation"),
                                "raw_response": data
                            }
                        elif response.status_code in (401, 403):
                            return {
                                "valid": False,
                                "error": "Invalid or expired API Key"
                            }
                        elif response.status_code == 404:
                            # Endpoint tidak ditemukan, coba endpoint berikutnya
                            break
                        else:
                            response.raise_for_status()
                    except HTTPStatusError as e:
                        last_error = e
                        if e.response.status_code in (429, 502, 503, 504):
                            await asyncio.sleep(self._retry_delay * (2 ** attempt))
                        elif e.response.status_code in (401, 403):
                            return {
                                "valid": False,
                                "error": "Invalid or expired API Key"
                            }
                        else:
                            break
                    except RequestError as e:
                        last_error = e
                        await asyncio.sleep(self._retry_delay * (2 ** attempt))
            
            # Jika semua endpoint gagal dengan 404, API mungkin valid tapi endpoint berbeda
            return {
                "valid": False,
                "error": f"Tidak dapat memvalidasi kunci API: {last_error}"
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Validasi kunci API gagal: {str(e)}"
            }
    
    async def create_account_if_needed(self, agent_name: str, wallet_address: Optional[str] = None, link_onchain: bool = False) -> str:
        """Buat akun dan kembalikan kunci API jika kunci saat ini tidak valid atau hilang.
        
        Args:
            agent_name: Nama untuk agent/akun baru
            wallet_address: Alamat wallet opsional untuk pendaftaran ERC-8004
            link_onchain: Apakah akan menautkan ke identitas on-chain ERC-8004 setelah pembuatan akun
            
        Returns:
            Kunci API baru jika akun dibuat, atau kunci yang ada jika valid
        """
        # Pertama periksa apakah kunci API saat ini valid
        if self.api_key:
            validation = await self.check_api_key_valid()
            if validation.get("valid"):
                print(f"✅ Kunci API valid untuk agent: {validation.get('agent_name', 'unknown')}")
                return self.api_key
            else:
                print(f"❌ Kunci API tidak valid: {validation.get('error')}")
                print("Mencoba mendaftarkan akun baru...")
        else:
            print("⚠️  Kunci API tidak ditemukan, mencoba mendaftarkan akun baru...")
        
        # Buat akun baru
        try:
            response = await self._request_with_retry(
                "POST",
                "/v1/auth/create-account",
                json={
                    "agent_name": agent_name,
                    "wallet_address": wallet_address
                }
            )
            data = response.json()
            new_api_key = data.get("api_key")
            
            if new_api_key:
                self.api_key = new_api_key
                # Perbarui klien dengan kunci API baru
                await self._ensure_client()
                
                # Simpan ke secrets.yaml
                self._save_api_key_to_secrets(new_api_key)
                
                print(f"✅ Akun baru berhasil dibuat")
                print(f"✅ Kunci API disimpan ke config/secrets.yaml")
                
                # Tautkan ke identitas on-chain jika diminta
                if link_onchain:
                    await self._link_onchain_identity(agent_name, wallet_address)
                
                return new_api_key
            else:
                raise RuntimeError("Tidak ada kunci API yang dikembalikan dari pembuatan akun")
                
        except Exception as e:
            raise RuntimeError(f"Gagal membuat akun: {str(e)}")
    
    async def _link_onchain_identity(self, agent_name: str, wallet_address: Optional[str] = None) -> None:
        """Tautkan akun ke identitas on-chain ERC-8004 jika memungkinkan."""
        try:
            from src.utils.onchain import OnChainManager
            
            print("Mencoba menautkan ke identitas on-chain ERC-8004...")
            
            # Inisialisasi manajer on-chain
            onchain_manager = OnChainManager(
                config_path=self.config_path,
                secrets_path=self.secrets_path
            )
            
            # Dapatkan alamat wallet dari secrets jika tidak disediakan
            if not wallet_address:
                wallet_address = onchain_manager.get_wallet_address()
            
            if not wallet_address:
                print("⚠️  Tidak ada alamat wallet yang tersedia, melewatkan penautan identitas on-chain")
                return
            
            # Periksa apakah identitas sudah ada
            existing_id = await onchain_manager.check_erc8004_identity(wallet_address, agent_name)
            
            if existing_id:
                print(f"✅ Identitas on-chain sudah ada dengan ID: {existing_id}")
            else:
                print("Identitas on-chain yang ada tidak ditemukan, mencoba mendaftar...")
                
                # Coba daftarkan identitas baru (memerlukan private key)
                new_id = await onchain_manager.register_erc8004(agent_name)
                
                if new_id:
                    print(f"✅ Identitas on-chain berhasil terdaftar dengan ID: {new_id}")
                else:
                    print("⚠️  Pendaftaran identitas on-chain gagal (mungkin memerlukan pendaftaran manual)")
        
        except ImportError:
            print("⚠️  web3.py tidak terinstal, melewatkan penautan identitas on-chain")
            print("   Instal dengan: pip install web3")
        except Exception as e:
            print(f"⚠️  Gagal menautkan identitas on-chain: {e}")
            print("   Melanjutkan tanpa identitas on-chain...")
    
    def _save_api_key_to_secrets(self, api_key: str) -> None:
        """Simpan kunci API ke file secrets.yaml."""
        try:
            secrets = {}
            if os.path.exists(self.secrets_path):
                with open(self.secrets_path, 'r') as f:
                    secrets = yaml.safe_load(f) or {}
            
            secrets['api_key'] = api_key
            
            with open(self.secrets_path, 'w') as f:
                yaml.dump(secrets, f, default_flow_style=False)
                
        except Exception as e:
            print(f"Peringatan: Tidak dapat menyimpan kunci API ke secrets.yaml: {e}")
    
    async def get_agent_info(self) -> Dict[str, Any]:
        """Dapatkan informasi agent lengkap.
        
        Returns:
            Dict dengan nama agent, wallet yang terhubung, ERC-8004 agentId, reputasi, dll.
        """
        validation = await self.check_api_key_valid()
        
        if validation.get("valid"):
            return {
                "agent_name": validation.get("agent_name"),
                "agent_id": validation.get("agent_id"),
                "wallet_address": validation.get("wallet_address"),
                "erc8004_agent_id": validation.get("erc8004_agent_id"),
                "reputation": validation.get("reputation"),
                "status": validation.get("status"),
                "raw_data": validation.get("raw_response")
            }
        else:
            return {
                "error": validation.get("error"),
                "valid": False
            }
    
    async def setup(self, agent_name: str = "MoltyBot", wallet_address: Optional[str] = None, link_onchain: bool = False) -> bool:
        """Atur klien dengan validasi kunci API otomatis dan pendaftaran jika diperlukan.
        
        Args:
            agent_name: Nama yang akan digunakan jika membuat akun baru
            wallet_address: Alamat wallet opsional untuk ERC-8004
            link_onchain: Apakah akan menautkan ke identitas on-chain ERC-8004 setelah pembuatan akun
            
        Returns:
            True jika pengaturan berhasil, False jika tidak
        """
        try:
            print("Mengatur MoltyRoyaleClient...")
            
            # Validasi atau buat kunci API
            api_key = await self.create_account_if_needed(agent_name, wallet_address, link_onchain)
            
            # Dapatkan info agent
            agent_info = await self.get_agent_info()
            
            if agent_info.get("valid"):
                print(f"✅ Pengaturan selesai untuk agent: {agent_info.get('agent_name')}")
                print(f"   Agent ID: {agent_info.get('agent_id')}")
                if agent_info.get('wallet_address'):
                    print(f"   Wallet: {agent_info.get('wallet_address')}")
                return True
            else:
                print(f"❌ Pengaturan gagal: {agent_info.get('error')}")
                return False
                
        except Exception as e:
            print(f"❌ Pengaturan gagal dengan exception: {str(e)}")
            return False
