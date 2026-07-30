import json
import math
import random
import sys
import time
import multiprocessing

from nn import NeuralNetwork
from pong import Pong, H, PADDLE_H

# ga defaults - tweak these if you want
POP_SIZE = 80
GAMES_PER_EVAL = 2
MAX_STEPS = 600
ELITE_COUNT = 4
TOURNEY_K = 4
MUT_RATE = 0.15
MUT_SCALE = 0.2
XOVER_RATE = 0.7


# Inlined logic for performance instead of external function calls


# play n games and return avg fitness
def evaluate(brain, n_games, fps=25.0):
    speed_mult = fps / 25.0
    total = 0
    for _ in range(n_games):
        g = Pong(ball_speed=0.9 * speed_mult, paddle_speed=0.6 * speed_mult)
        while not g.done and g.steps < MAX_STEPS:
            s = g.get_state(for_player=2)
            # inline brain action
            out = brain.forward(s)
            a2 = 1 if out[0] > 0.5 else -1
            # inline simple AI
            diff = g.ball_y - g.paddle1_y
            a1 = 0 if abs(diff) < 0.8 else (1 if diff > 0 else -1)
            g.step(a1, a2)
        # better fitness: reward hits, heavily reward scoring, penalize conceding
        total += (g.hits * 10) + (g.score2 * 100) - (g.score1 * 10)
    return total / n_games


# worker for multiprocessing
def eval_worker(args):
    brain, n_games, fps = args
    return evaluate(brain, n_games, fps)


# tournament selection: pick k random, return index of best
def tourney_select(fitnesses, k):
    best = None
    for _ in range(k):
        idx = random.randint(0, len(fitnesses) - 1)
        if best is None or fitnesses[idx] > fitnesses[best]:
            best = idx
    return best


# uniform crossover - mathematically much better for neural network breeding
def crossover(p1, p2):
    if random.random() < XOVER_RATE:
        return [a if random.random() < 0.5 else b for a, b in zip(p1, p2)]
    return p1[:]


# gaussian mutation
def mutate(params):
    return [p + random.gauss(0, MUT_SCALE) if random.random() < MUT_RATE else p for p in params]


# save brain to json
def save_brain(brain, fitness, path="best_brain.json"):
    with open(path, "w") as f:
        json.dump({
            "layer_sizes": brain.layer_sizes,
            "params": brain.get_params(),
            "fitness": round(fitness, 2),
        }, f)


# load brain from json
def load_brain(path="best_brain.json"):
    try:
        with open(path) as f:
            d = json.load(f)
        b = NeuralNetwork(d["layer_sizes"])
        b.set_params(d["params"])
        return b
    except:
        return None  # file not found or whatever


# run 1 quick game and return stats (no rendering)
def demo_game(brain, fps=25.0):
    speed_mult = fps / 25.0
    g = Pong(ball_speed=0.9 * speed_mult, paddle_speed=0.6 * speed_mult)
    while not g.done and g.steps < 300:
        s = g.get_state(for_player=2)
        out = brain.forward(s)
        a2 = 1 if out[0] > 0.5 else -1
        diff = g.ball_y - g.paddle1_y
        a1 = 0 if abs(diff) < 0.8 else (1 if diff > 0 else -1)
        g.step(a1, a2)
    return g.hits, g.score1, g.score2


# main training loop - runs forever until ctrl+c
def continuous_train(layer_sizes, pop_size=None, games_per_eval=None,
                     show_every=5, save_every=10, save_path="best_brain.json",
                     seed=None, fps=25.0):
    if seed is not None:
        random.seed(seed)

    ps = pop_size or POP_SIZE
    gpe = games_per_eval or GAMES_PER_EVAL

    # try to load existing brain to pick up where we left off
    saved_brain = load_brain(save_path)
    if saved_brain:
        print(f"loaded existing brain from {save_path}, resuming training...")
        pop = [saved_brain.copy()]
        # seed rest of pop with mutations of saved brain
        for _ in range(ps - 1):
            child = NeuralNetwork(layer_sizes)
            child.set_params(mutate(saved_brain.get_params()))
            pop.append(child)
        best_brain = saved_brain
        best_fit = 0.0
    else:
        # create initial population
        pop = [NeuralNetwork(layer_sizes) for _ in range(ps)]
        best_fit = 0.0
        best_brain = None

    gen = 0

    print("training pong ai with neuroevolution (ctrl+c to stop)")
    print(f"  pop={ps}  net={layer_sizes}  mut_rate={MUT_RATE}  mut_scale={MUT_SCALE}")
    print()

    try:
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            while True:
                gen += 1
                fits = []

                # evaluate every individual in the population using multiprocessing
                tasks = [(brain, gpe, fps) for brain in pop]
                chunk_size = max(1, len(tasks) // multiprocessing.cpu_count())
                for i, f in enumerate(pool.imap(eval_worker, tasks, chunksize=chunk_size)):
                    fits.append(f)

                    # progress bar
                    bar = "#" * int((i + 1) / ps * 20) + "." * (20 - int((i + 1) / ps * 20))
                    sys.stdout.write(f"\r  gen {gen:5d} [{bar}]")
                    sys.stdout.flush()

                # stats for this generation
                gen_best = max(fits)
                gen_avg = sum(fits) / len(fits)

                # save if new record
                if gen_best > best_fit:
                    best_fit = gen_best
                    best_brain = pop[fits.index(gen_best)].copy()
                    if gen % save_every == 0 or gen == 1:
                        save_brain(best_brain, best_fit, save_path)

                sys.stdout.write(f"\r  gen {gen:5d}  best={gen_best:6.1f}  avg={gen_avg:6.1f}  record={best_fit:6.1f}")
                sys.stdout.flush()

                # show a quick demo game so you can see how its doing
                if gen % show_every == 0 and best_brain is not None:
                    hits, s1, s2 = demo_game(best_brain, fps)
                    sys.stdout.write(f"  demo: {hits} hits ({s1}-{s2})")
                sys.stdout.write("\n")

                # keep the elites
                order = sorted(range(len(fits)), key=lambda i: fits[i], reverse=True)
                elites = [pop[i].copy() for i in order[:ELITE_COUNT]]

                # breed the rest
                next_pop = elites[:]
                while len(next_pop) < ps:
                    p1 = pop[tourney_select(fits, TOURNEY_K)]
                    p2 = pop[tourney_select(fits, TOURNEY_K)]
                    child_params = crossover(p1.get_params(), p2.get_params())
                    child_params = mutate(child_params)
                    child = NeuralNetwork(layer_sizes)
                    child.set_params(child_params)
                    next_pop.append(child)

                pop = next_pop

    except KeyboardInterrupt:
        print(f"\n\nstopped at gen {gen}. best fitness: {best_fit:.1f}")
        if best_brain is not None:
            save_brain(best_brain, best_fit, save_path)
            print(f"saved best brain to {save_path}")
        return best_brain, best_fit
