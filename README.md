# Self-Aware-Snake (Unlike you)

# 🐍 RL-Snake: Reinforcement Learning Snake Game

Welcome to **RL-Snake**, a machine learning-enhanced version of the classic Snake game, where an AI agent learns to play using Q-learning. Yes, it’s exactly as unnecessary and over-engineered as it sounds.

## 🎮 Features

* ✅ Q-Learning-based AI agent
* ✅ Obstacle generation for added chaos
* ✅ Optional human mode (for you control freaks)
* ✅ Menu system with keypress navigation (we're fancy now)
* ✅ Custom reward shaping & state representation
* ✅ Training visualization and score plotting
* ✅ Saves and loads Q-table using `pickle` like it’s 2009

## 🧠 How It Works

The snake learns by interacting with the environment:

* **State Space**: Considers immediate dangers, food direction, and current movement direction.
* **Actions**: `'straight'`, `'left'`, `'right'`
* **Rewards**:

  * +10 for eating food
  * -10 for hitting walls, self, or obstacles
  * ±0.1 for moving toward/away from food
* **Q-Table**: Stored in a `defaultdict` mapping states to action-values.

Training uses a simple epsilon-greedy policy with decay, so the snake starts off dumb and slowly becomes less dumb. Like most of us.

## 🖥️ Requirements

* Python 3.7+
* `pygame`
* `numpy`
* `matplotlib`

Install them like this (unless you enjoy suffering):

```bash
pip install pygame numpy matplotlib
```

## 🚀 How to Run

```bash
python rl_snake.py
```

Then choose a mode:

* **T**: Train the AI from scratch
* **P**: Watch the AI slither around and pretend it’s impressive
* **H**: Play the game yourself (so you can lose to your own creation)
* **Q**: Quit. The only mode that respects your time.

## 📈 Training Output

* Progress is visualized during training (optional).
* Scores and moving averages are saved to `training_scores.png`.
* The Q-table is saved to `q_table.pkl` after training.

## 🪦 Known Issues

* The AI still occasionally dies doing something inexplicably stupid. Just like real intelligence.
* Large Q-tables can grow rapidly depending on the state encoding. Use responsibly.
* If you forget to save your Q-table... that’s on you.

## 🤷‍♂️ Future Work

* Neural network version (DQN) because we love complexity.
* Smarter state representations (vision grid, danger radius).
* Online leaderboard? LOL.

