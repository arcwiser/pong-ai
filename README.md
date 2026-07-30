# scratch-pong

pong ai trained with neuroevolution. everything from scratch - no numpy, no tensorflow, no nothing. just python stdlib.

### how it works

- **nn.py** — matrix class + feedforward neural network (lists of lists style, no libs)
- **pong.py** — pong game engine with terminal rendering
- **train.py** — genetic algorithm: tournament selection, 1-point crossover, gaussian mutation
- **main.py** — cli entry point

the ai learns by playing against a simple follow-the-ball bot. fitness = how many times it returns the ball.

### usage

```
python main.py train               # runs forever, ctrl+c to stop
python main.py train --pop 100 --hidden 16 --games 3  
python main.py watch                # watch trained ai play
python main.py play                 # play against the ai (windows)
python main.py demo                 # quick score check
```

the brain saves to `best_brain.json` every 10 gens. `watch` and `play` load from there.
