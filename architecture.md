neural_gravity_assist/
├── agents/                   # Custom RL logic
│   ├── __init__.py
│   ├── ppo_custom.py         # Your scratch implementation or SB3 wrapper
├── envs/                     # Gymnasium environments
│   ├── __init__.py
│   ├── gravity_env.py        # The core REBOUND + Gym wrapper
├── data/                     # Ephemeris and training output
│   └── results/              # TensorBoard logs and saved .zip models
├── visuals/                  # Data visualization 
│   └── matplot.py            # Rebound simulation in matplotlibs
├── main.py                   # Entry point for running the program
├── requirements.txt          # Required libraries and downloads
└── architecture              # The structure of the codebase 