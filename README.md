# Molty Royale Self-Learning Bot

AI agent berbasis Reinforcement Learning yang dapat belajar mandiri (self-learning) untuk bermain Molty Royale. Bot ini menggunakan PPO (Proximal Policy Optimization) dengan Stable-Baselines3 dan secara otomatis meningkatkan performanya melalui training berkelanjutan.

## Fitur Utama

- **Self-Learning Mandiri**: Bot secara otomatis melatih model baru setiap 20 game
- **PPO with Stable-Baselines3**: Algoritma RL state-of-the-art untuk decision making
- **Rule-Based Fallback**: Strategi heuristik sebagai baseline dan fallback
- **Experience Replay**: Menyimpan pengalaman game untuk training offline
- **Automatic Evaluation**: Membandingkan model baru vs model terbaik secara otomatis
- **Shadow Mode**: Evaluasi model tanpa mempengaruhi game nyata
- **Rich Logging**: Console logging dengan Rich + file logging + JSON state saving
- **Web3 Ready**: Support ERC-8004 on-chain identity (opsional)
- **Docker Support**: Siap deploy 24/7 di Railway/Render/VPS

## Project Structure

```
molty_royale_bot/
├── config/
│   ├── config.yaml          # Konfigurasi API dan game
│   └── secrets.yaml         # API key dan wallet private key
├── src/
│   ├── api_client.py        # Async HTTP client untuk Molty Royale API
│   ├── state_parser.py      # Parser JSON state ke 128-dim features
│   ├── strategy/
│   │   ├── rule_based.py    # Agent rule-based dengan heuristik
│   │   └── rl_agent.py      # Agent RL (placeholder)
│   ├── game_loop.py         # Main game loop
│   ├── ml/
│   │   ├── environment.py   # Gymnasium environment
│   │   ├── replay_buffer.py # Experience replay (Parquet/HDF5)
│   │   ├── model.py         # PPO model wrapper
│   │   ├── trainer.py       # AutoTrainer untuk self-learning
│   │   └── evaluator.py     # Model evaluation di shadow mode
│   ├── utils/
│   │   ├── logger.py        # Rich console + file logging
│   │   └── onchain.py       # Web3 utilities (ERC-8004)
│   └── constants.py         # Konstanta game dan training
├── data/
│   ├── logs/                # Log files dan JSON states
│   ├── replay/              # Replay buffer data
│   └── models/              # Trained models
├── main.py                 # Entry point
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
└── README.md              # Dokumentasi ini
```

## Cara Install

### Prerequisites

- Python 3.11 atau lebih baru
- pip
- (Opsional) Docker untuk containerized deployment

### Local Installation

```bash
# Clone repository
git clone <repository-url>
cd molty_royale_bot

# Install dependencies
pip install -r requirements.txt
```

### Docker Installation

```bash
# Build image
docker build -t molty-royale-bot .

# Run container
docker run -d --name molty-bot molty-royale-bot
```

## Cara Mendapatkan API Key

1. Buka [moltyroyale.com](https://moltyroyale.com)
2. Sign up atau login ke akun Anda
3. Navigasi ke **Settings** → **API Keys**
4. Generate new API key
5. Copy API key (format: `mr_live_XXXXXXXX`)
6. Paste ke file `config/secrets.yaml`:

```yaml
api_key: "mr_live_YOUR_API_KEY_HERE"
wallet_private_key: ""  # Opsional untuk ERC-8004
```

**Catatan**: Jangan commit `config/secrets.yaml` ke git! File ini sudah ada di `.gitignore`.

## Setup & API Key Management

### Cara Mengelola API Key & Agent

#### 1. Pertama kali jalankan:

```bash
python main.py --agent-name "NamaBotKamu"
```

Bot akan otomatis:
- Cek API Key yang ada di config/secrets.yaml atau environment variable
- Jika belum ada atau invalid → buat akun baru via API
- Link dengan wallet (jika disediakan di secrets.yaml)
- Daftarkan ERC-8004 jika `--erc8004` flag digunakan

**Output yang diharapkan:**
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Welcome                                              ┃
┃ Molty Royale Self-Learning Bot                      ┃
┃ AI-powered autonomous agent with continuous learning ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Setup Phase                                          ┃
┃ Step 1: API Client Setup                            ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

Checking API key validity...

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ API Key Status                                       ┃
┃ ✅ API Key Valid                                     ┃
┃ Agent: NamaBotKamu                                  ┃
┃ Wallet: 0x1234...abcd                                ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

#### 2. Jika lupa API Key:

**Opsi A: Auto-register dengan flag `--register-if-needed`**
```bash
# Hapus baris api_key di config/secrets.yaml
# Lalu jalankan:
python main.py --agent-name "NamaBotKamu" --register-if-needed
```

Bot akan:
- Mendeteksi API key kosong/invalid
- Register akun baru secara otomatis
- Simpan API key baru ke config/secrets.yaml
- Lanjutkan dengan setup biasa

**Opsi B: Manual register di website**
1. Buka [https://www.moltyroyale.com/](https://www.moltyroyale.com/)
2. Login atau sign up
3. Navigasi ke Settings → API Keys
4. Generate new API key
5. Copy API key (format: `mr_live_XXXXXXXX`)
6. Paste ke `config/secrets.yaml`

#### 3. config/secrets.yaml contoh:

```yaml
api_key: "mr_live_xxxxxxxxxxxxxxxxxxxxxxxx"
wallet_private_key: "0x..."   # hanya jika mau auto register ERC-8004
wallet_address: "0x..."      # wallet address untuk ERC-8004
```

⚠️ **PERINGATAN KEAMANAN**:
- **Jangan pernah share** file `config/secrets.yaml` kepada siapapun
- File ini sudah ada di `.gitignore` untuk mencegah commit ke git
- Jangan commit file ini ke repository publik atau private
- Jika terlanjur commit, revoke API key di moltyroyale.com dan generate baru
- `wallet_private_key` sangat sensitif - hanya isi jika Anda memahami risikonya

## Cara Run Bot

### Local Development

```bash
# Run dengan rule-based agent
python main.py --agent-type rule_based

# Run dengan RL agent (setelah training)
python main.py --agent-type rl

# Run dengan ERC-8004 identity
python main.py --erc8004
```

### Environment Variables

Anda juga bisa set environment variables:

```bash
export MOLTY_API_KEY="mr_live_YOUR_API_KEY"
export MOLTY_GAME_ID="default_free"
python main.py
```

### Docker Deployment

```bash
# Run dengan environment variables
docker run -d \
  -e MOLTY_API_KEY="mr_live_YOUR_API_KEY" \
  -e MOLTY_GAME_ID="default_free" \
  --name molty-bot \
  molty-royale-bot
```

## Cara Monitor Training

### TensorBoard

Bot menulis training logs ke TensorBoard secara otomatis:

```bash
# Jalankan TensorBoard
tensorboard --logdir data/models/tensorboard

# Buka browser ke http://localhost:6006
```

### Log Files

Log tersimpan di `data/logs/`:

```bash
# Lihat log terbaru
tail -f data/logs/bot_YYYYMMDD_HHMMSS.log

# Lihat JSON states
ls data/logs/raw_json/
```

### Training Statistics

Bot menampilkan statistik training secara real-time:

- Games since last train
- Training progress
- Win rate improvement
- Replay buffer usage

### AutoTrainer Stats

Anda bisa cek statistik trainer dengan memodifikasi `main.py` untuk menampilkan stats:

```python
stats = trainer.get_stats()
print(f"Games since last train: {stats['games_since_last_train']}")
print(f"Best win rate: {stats['best_win_rate']:.2%}")
print(f"Replay buffer usage: {stats['replay_buffer_usage']:.2%}")
```

## Fitur Self-Learning

Bot ini memiliki sistem self-learning mandiri yang bekerja sebagai berikut:

### 1. Experience Collection

Bot bermain game dan mengumpulkan experience:
- State (128-dimensional feature vector)
- Action yang diambil
- Reward yang diterima
- Next state
- Done flag

Experience disimpan di replay buffer (Parquet/HDF5).

### 2. Automatic Training Trigger

Setiap 20 game (configurable), bot akan:
1. Cek apakah ada cukup data di replay buffer
2. Trigger training session di background thread
3. Training berjalan selama 30-60 menit (configurable)

### 3. PPO Training

Training menggunakan PPO dengan konfigurasi:
- Learning rate: 3e-4
- Batch size: 256
- Network architecture: [256, 256, 128]
- Gamma: 0.99
- TensorBoard logging enabled

### 4. Model Evaluation

Setelah training selesai:
1. Model baru dievaluasi vs rule-based agent di shadow mode
2. Win rate dihitung dari 10 episode evaluation
3. Hasil dibandingkan dengan best model saat ini

### 5. Model Deployment

Jika win rate meningkat >5% (configurable):
1. Old best model dibackup
2. New model disalin sebagai `best_model.zip`
3. Game loop otomatis load model baru
4. Game berikutnya menggunakan model improved

Jika win rate tidak meningkat:
1. Model baru disimpan sebagai `model_{count}.zip`
2. Best model tetap dipertahankan
3. Bot tetap menggunakan model terbaik

### 6. Continuous Improvement

Proses ini berulang secara otomatis:
- Bot terus bermain dan mengumpulkan experience
- Training periodik menghasilkan model yang lebih baik
- Model yang lebih baik otomatis dideploy
- Bot menjadi lebih kuat seiring waktu

## Konfigurasi

### config/config.yaml

```yaml
api_base: "https://api.moltyroyale.com"
game_id: "default_free"
api_key: "mr_live_XXXXXXXX"  # Placeholder, gunakan secrets.yaml
max_games: 9999
ml_training_interval: 20
```

### config/secrets.yaml

```yaml
api_key: "mr_live_YOUR_REAL_API_KEY"
wallet_private_key: ""  # Opsional untuk ERC-8004
```

### Training Constants (src/constants.py)

```python
class TrainingConfig:
    LEARNING_RATE = 3e-4
    GAMMA = 0.99
    BUFFER_SIZE = 100000
    BATCH_SIZE = 256
    TRAINING_INTERVAL = 20
    SAVE_INTERVAL = 100
    EVAL_INTERVAL = 50
```

## Deployment Platforms

### Railway

1. Push code ke GitHub
2. Connect repository ke Railway
3. Set environment variables:
   - `MOLTY_API_KEY`
   - `MOLTY_GAME_ID`
4. Deploy

### Render

1. Push code ke GitHub
2. Connect repository ke Render
3. Create new Web Service
4. Set environment variables
5. Deploy

### VPS (Ubuntu/Debian)

```bash
# Clone repository
git clone <repository-url>
cd molty_royale_bot

# Install dependencies
pip install -r requirements.txt

# Setup systemd service
sudo nano /etc/systemd/system/molty-bot.service
```

```
[Unit]
Description=Molty Royale Bot
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/molty_royale_bot
ExecStart=/usr/bin/python3 /path/to/molty_royale_bot/main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable molty-bot
sudo systemctl start molty-bot

# Check status
sudo systemctl status molty-bot
```

## Troubleshooting

### API Connection Error

- Pastikan API key valid
- Cek koneksi internet
- Verifikasi `api_base` di config.yaml

### Training Not Starting

- Pastikan replay buffer memiliki cukup data (min 2560 transitions)
- Cek log untuk pesan "Not enough data in replay buffer"
- Tingkatkan `train_interval_games` jika perlu

### Model Not Improving

- Tingkatkan `train_duration_minutes`
- Cek reward shaping di `src/constants.py`
- Pertimbangkan untuk menggunakan PrioritizedReplayBuffer
- Review hyperparameters di TrainingConfig

### Memory Issues

- Kurangi `BUFFER_SIZE` di TrainingConfig
- Gunakan storage_format="parquet" untuk replay buffer
- Kurangi batch size

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Disclaimer

Bot ini untuk tujuan edukasi dan research. Gunakan dengan bertanggung jawab dan patuhi terms of service Molty Royale.
