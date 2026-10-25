# SpaceCadetPinball-AI
Python code that trains an AI with reinforcement learning to play [SpaceCadetPinball](https://github.com/k4zmu2a/SpaceCadetPinball), the source port of 3D Pinball. A patch is applied to SpaceCadetPinball to let it communicate with the AI.
- Train mode: runs the games in parallel and trains the AI.
- Play mode: runs a trained model in the real game. A new game window opens.

## Installation
**These are build instructions for Windows.**
You need `git`, `Python 3`, `C++11`, `CMake`, the developer packages of [SDL2](https://github.com/libsdl-org/SDL/releases) and [SDL2_mixer](https://github.com/libsdl-org/SDL_mixer/releases), and Visual Studio.
```sh
# Clone this repository
git clone https://github.com/c00380/ScapeCadetPinball-AI.git
cd SpaceCadetPinball-AI

# Install the Python dependencies
pip install -r requirements.txt

# Clone the game
git clone https://github.com/k4zmu2a/SpaceCadetPinball.git
cd SpaceCadetPinball

# Apply the patch
git apply --ignore-whitespace ../patches/ai-player-cb9b7b8.patch

# Install the SDL2 and SDL2_mixer developer packages.
# Create a Libs folder in the root of the game repository and put the "SDL2" and "SDL2_mixer" folders in it.

# Build the game
mkdir build && cd build
cmake ..
cmake --build . --config Release
```
The build output goes to the bin/Release/ folder. For the game to work correctly, copy the original game data, such as `PINBALL.DAT` and the sound files, to the folder that contains the executable.

## Usage
```sh
# Run from the repository root
python ai/pinball_ai.py train # train mode
python ai/pinball_ai.py play  # play mode
```

#### Main options
| Command | Option | Meaning |
|---|---|---|
| train | `--fresh` | Delete `model/` and start over. |
| train | `--envs 32 --workers 4` | Number of parallel games, and number of Python processes that drive them |
| train | `--steps [num]` | Number of steps to train (default 20 million) |
| train | `--no-bc` | Skip the initial training with the rule-based model |
| play | `--model [custom_model.zip]` | Load the model from the given file |

## Files
| File | Role |
|---|---|
| `pinball_ai.py` | Command line (`train`, `play`) |
| `trainer.py`, `player.py`, `model_store.py` | Training loop, playback, model and snapshot files |
| `pinball_env.py` | Gymnasium environment: observations, rewards, human-like flipper control limits |
| `pinball_vec.py` | Parallel environment that runs many games without one Python process per game |
| `game_client.py` | Launches the game and talks to it over a socket |
| `fast_policy.py` | Makes decisions with numpy while collecting data (same maths as PyTorch, lower overhead) |
| `heuristic.py`, `rewards.py` | The rule-based demonstrator AI, reward constants |

## How training works
The AI learns with the left and right flippers. Table nudging is not implemented because it is hard to learn. The ball is launched automatically.
Training starts by imitating a rule-based model, then switches to reinforcement learning.
Random actions are mixed in during training, so play mode, which uses the best action, scores higher.

### Information given to the AI
At every decision (6 game frames = 1/20 s), 77 numbers are given: the state of the balls, the flippers and the table's features.
They are the position and speed of up to three balls, the position of the ball relative to each flipper, each flipper's angle, whether it is pressed and for how long, the remaining balls, multiball, extra ball, shoot again, ball save, the score multiplier, whether the plunger holds a ball, the current mission (one-hot) and its readiness, fuel, tilt, the logarithm of the score, and the state of the 17 lane lights.

### Rewards
| Stage | Main rewards |
|---|---|
| Rule-based model training | Hitting the ball with a flipper +1.0, touching a flipper that is not pressed +0.1, losing the ball −1.0 |
| Reinforcement learning | Score gain ÷ 5000 (up to 20 per step), hitting the ball with a flipper +0.1, losing the ball −2.0, a lane light newly lit +0.2 |

In both stages, holding a flipper costs −0.003 per step, having a ball in play gives +0.001, and a forced release costs −0.05. The switch from the rule-based training stage to reinforcement learning happens when the AI hits the ball 3 times per ball on average over 30 games.

### Control limits
Decisions are made 20 times per second. A flipper stays pressed for at least 0.1 s once pressed, and stays released for at least 0.1 s once released. If it is held for 0.6 s, it is forcibly released for 0.15 s.

### Model files
| File | Content |
|---|---|
| `model.zip` | The main model file |
| `snapshots/snap_<steps>_<stage>.zip` | Snapshot every 500,000 steps. |
| `snapshots/best_score.zip` | The model with the best mean score over the last 100 games |
| `meta.json`, `train_log.csv` | Training stage and step count, log |
| `vecnormalize.pkl` | Statistics of the reward scale. Needed only to resume training. |

## License
MIT-License