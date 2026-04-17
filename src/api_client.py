import asyncio
import time
from typing import Optional, Dict, Any, List
import httpx
from httpx import AsyncClient, Response, HTTPStatusError, RequestError
import yaml
import os


class MoltyRoyaleClient:
    """Async HTTP client for Molty Royale API with automatic retry and rate limit handling."""
    
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
        """Load configuration from YAML files and environment variables."""
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
            raise RuntimeError(f"Config file not found: {e}")
        except yaml.YAMLError as e:
            raise RuntimeError(f"Invalid YAML in config: {e}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_client(self) -> None:
        """Ensure HTTP client is initialized."""
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
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    async def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
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
        """Make HTTP request with automatic retry on failure."""
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
                    # Rate limit or server error - retry with backoff
                    await asyncio.sleep(self._retry_delay * (2 ** attempt))
                else:
                    # Other HTTP errors - don't retry
                    raise
            except RequestError as e:
                last_error = e
                await asyncio.sleep(self._retry_delay * (2 ** attempt))
        
        raise RuntimeError(f"Max retries exceeded: {last_error}")
    
    async def create_account(self, username: str, email: str) -> Dict[str, Any]:
        """Create a new Molty Royale account.
        
        Args:
            username: Desired username
            email: Email address
            
        Returns:
            Account creation response with user_id and initial credentials
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
        """Register a new AI agent for the game.
        
        Args:
            agent_name: Name of the agent
            agent_type: Type of agent (rl, rule_based, hybrid)
            description: Optional agent description
            
        Returns:
            Agent registration response with agent_id
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
        """Get current game state for an agent.
        
        Args:
            agent_id: Agent identifier
            game_session_id: Optional specific game session ID
            
        Returns:
            Current game state with observations, scores, and metadata
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
        """Send an action to the game engine.
        
        Args:
            agent_id: Agent identifier
            action: Action dictionary with action_type and parameters
            game_session_id: Optional specific game session ID
            
        Returns:
            Action response with new state, reward, and done flag
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
        """Get list of available game modes/configurations.
        
        Returns:
            List of available games with their metadata
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
        """Join a specific game instance.
        
        Args:
            game_id: Game identifier
            agent_id: Agent identifier
            mode: Game mode (standard, ranked, practice)
            
        Returns:
            Game session information with session_id
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
        """Leave a game session.
        
        Args:
            session_id: Game session identifier
            agent_id: Agent identifier
            
        Returns:
            Leave confirmation response
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
        """Get leaderboard for a game.
        
        Args:
            game_id: Optional game identifier (uses current game_id if not provided)
            limit: Maximum number of entries to return
            
        Returns:
            Leaderboard entries with rankings and scores
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
        """Get statistics for a specific agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent statistics including games played, win rate, etc.
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
        """Upload a trained model for the agent.
        
        Args:
            agent_id: Agent identifier
            model_path: Path to the model file
            model_version: Version string for the model
            
        Returns:
            Model upload confirmation with model_id
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
        """Download replay data for a game session.
        
        Args:
            session_id: Game session identifier
            
        Returns:
            Replay data with all actions and states from the session
        """
        response = await self._request_with_retry(
            "GET",
            f"/v1/game/replay/{session_id}"
        )
        return response.json()
    
    async def check_api_key_valid(self) -> Dict[str, Any]:
        """Check if the current API key is valid.
        
        Returns:
            Dict with 'valid' boolean and agent info if valid, or error if invalid.
            Returns {"valid": False, "error": "..."} on 401/403 errors.
        """
        try:
            await self._ensure_client()
            
            # Try multiple endpoints for API key validation
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
                            # Endpoint not found, try next endpoint
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
            
            # If all endpoints failed with 404, API might be valid but endpoints different
            return {
                "valid": False,
                "error": f"Could not validate API key: {last_error}"
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"API key validation failed: {str(e)}"
            }
    
    async def create_account_if_needed(self, agent_name: str, wallet_address: Optional[str] = None, link_onchain: bool = False) -> str:
        """Create account and return API key if current key is invalid or missing.
        
        Args:
            agent_name: Name for the new agent/account
            wallet_address: Optional wallet address for ERC-8004 registration
            link_onchain: Whether to link to ERC-8004 on-chain identity after account creation
            
        Returns:
            New API key if account was created, or existing key if valid
        """
        # First check if current API key is valid
        if self.api_key:
            validation = await self.check_api_key_valid()
            if validation.get("valid"):
                print(f"✅ API Key valid for agent: {validation.get('agent_name', 'unknown')}")
                return self.api_key
            else:
                print(f"❌ API Key invalid: {validation.get('error')}")
                print("Attempting to register new account...")
        else:
            print("⚠️  No API Key found, attempting to register new account...")
        
        # Create new account
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
                # Update the client with new API key
                await self._ensure_client()
                
                # Save to secrets.yaml
                self._save_api_key_to_secrets(new_api_key)
                
                print(f"✅ New account created successfully")
                print(f"✅ API Key saved to config/secrets.yaml")
                
                # Link to on-chain identity if requested
                if link_onchain:
                    await self._link_onchain_identity(agent_name, wallet_address)
                
                return new_api_key
            else:
                raise RuntimeError("No API key returned from account creation")
                
        except Exception as e:
            raise RuntimeError(f"Failed to create account: {str(e)}")
    
    async def _link_onchain_identity(self, agent_name: str, wallet_address: Optional[str] = None) -> None:
        """Link account to ERC-8004 on-chain identity if possible."""
        try:
            from src.utils.onchain import OnChainManager
            
            print("Attempting to link to ERC-8004 on-chain identity...")
            
            # Initialize on-chain manager
            onchain_manager = OnChainManager(
                config_path=self.config_path,
                secrets_path=self.secrets_path
            )
            
            # Get wallet address from secrets if not provided
            if not wallet_address:
                wallet_address = onchain_manager.get_wallet_address()
            
            if not wallet_address:
                print("⚠️  No wallet address available, skipping on-chain identity linking")
                return
            
            # Check if identity already exists
            existing_id = await onchain_manager.check_erc8004_identity(wallet_address, agent_name)
            
            if existing_id:
                print(f"✅ On-chain identity already exists with ID: {existing_id}")
            else:
                print("No existing on-chain identity found, attempting to register...")
                
                # Try to register new identity (requires private key)
                new_id = await onchain_manager.register_erc8004(agent_name)
                
                if new_id:
                    print(f"✅ On-chain identity registered successfully with ID: {new_id}")
                else:
                    print("⚠️  On-chain identity registration failed (may require manual registration)")
        
        except ImportError:
            print("⚠️  web3.py not installed, skipping on-chain identity linking")
            print("   Install with: pip install web3")
        except Exception as e:
            print(f"⚠️  Failed to link on-chain identity: {e}")
            print("   Continuing without on-chain identity...")
    
    def _save_api_key_to_secrets(self, api_key: str) -> None:
        """Save API key to secrets.yaml file."""
        try:
            secrets = {}
            if os.path.exists(self.secrets_path):
                with open(self.secrets_path, 'r') as f:
                    secrets = yaml.safe_load(f) or {}
            
            secrets['api_key'] = api_key
            
            with open(self.secrets_path, 'w') as f:
                yaml.dump(secrets, f, default_flow_style=False)
                
        except Exception as e:
            print(f"Warning: Could not save API key to secrets.yaml: {e}")
    
    async def get_agent_info(self) -> Dict[str, Any]:
        """Get complete agent information.
        
        Returns:
            Dict with agent name, wallet linked, ERC-8004 agentId, reputation, etc.
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
        """Setup client with automatic API key validation and registration if needed.
        
        Args:
            agent_name: Name to use if creating a new account
            wallet_address: Optional wallet address for ERC-8004
            link_onchain: Whether to link to ERC-8004 on-chain identity after account creation
            
        Returns:
            True if setup successful, False otherwise
        """
        try:
            print("Setting up MoltyRoyaleClient...")
            
            # Validate or create API key
            api_key = await self.create_account_if_needed(agent_name, wallet_address, link_onchain)
            
            # Get agent info
            agent_info = await self.get_agent_info()
            
            if agent_info.get("valid"):
                print(f"✅ Setup complete for agent: {agent_info.get('agent_name')}")
                print(f"   Agent ID: {agent_info.get('agent_id')}")
                if agent_info.get('wallet_address'):
                    print(f"   Wallet: {agent_info.get('wallet_address')}")
                return True
            else:
                print(f"❌ Setup failed: {agent_info.get('error')}")
                return False
                
        except Exception as e:
            print(f"❌ Setup failed with exception: {str(e)}")
            return False
