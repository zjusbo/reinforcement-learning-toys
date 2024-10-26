# reinforcement-learning-toys
Some toy examples in reinforcement learning. Inspired by the David Silver's lecture


## q_learn.py

This is the Dyna-Q learning algorithm optimizes a grid game: Sarsa on the Windy Gridworld
The game is introduced in David Silver's reinforcement course [lecture 5](https://youtu.be/0g4j2k_Ggc4?feature=shared&t=2983).

The Dyna-Q algorithm is introduced in [lecture 8](https://youtu.be/ItMutbeOHtc?si=vmOCb08iacUwLyuT&t=3240)

In this file, we explored different hyper-parameters, and find the best one, which takes around 2 episode runs to get the optimal path.

1. search for the best learing rate

2. search for the best discount factor

3. search for the best epsilon (in the epsilon-greedy function)

4. search for the best simulation steps (in the model rerunning)




## Configs

To install a package in the python virtual environment

1. activate the environment: `source /path/to/your/venv/bin/activate`
2. install the package: `pip install <package-name>`



To config the sublime build system.

$ which python3

Create a new build system in Sublime, and fill in the following config
```
{
    "shell_cmd": "<path output of <which python 3>> -u \"$file\"",
    "file_regex": "^[ ]*File \"(...*?)\", line ([0-9]*)",
    "selector": "source.python"
}
```

∏