#!/usr/bin/env python3
import argparse
import sys
import time

from nn import NeuralNetwork
from pong import Pong
from train import continuous_train, simple_ai, brain_action, POP_SIZE


def do_train(args):
    layers = [4, args.hidden, 1]
    continuous_train(
        layers,
        pop_size=args.pop,
        games_per_eval=args.games,
        show_every=args.show,
        save_every=args.save_every,
        save_path=args.save,
        seed=args.seed,
    )


def do_watch(args):
    from train import load_brain
    brain = load_brain(args.load)
    if brain is None:
        print(f"cant load brain from '{args.load}' - did you train one?")
        sys.exit(1)

    print("watching ai vs ai - ctrl+c to quit")
    try:
        while True:
            g = Pong()
            while not g.done:
                s = g.get_state(for_player=2)
                a2 = brain_action(brain, s)
                a1 = simple_ai(g.ball_y, g.paddle1_y)
                g.step(a1, a2)
                g.render()
                time.sleep(0.04)
            time.sleep(1.5)
    except KeyboardInterrupt:
        print("\ndone")


def do_play(args):
    from train import load_brain
    brain = load_brain(args.load)
    if brain is None:
        print(f"cant load brain from '{args.load}'")
        sys.exit(1)

    try:
        import msvcrt
    except ImportError:
        print("play mode needs windows (msvcrt). try 'watch' instead")
        sys.exit(1)

    g = Pong()
    print("you are player 1 (left). w=up  s=down  q=quit")
    time.sleep(1)

    try:
        while not g.done:
            s = g.get_state(for_player=2)
            a2 = brain_action(brain, s)
            a1 = 0
            if msvcrt.kbhit():
                k = msvcrt.getch().decode().lower()
                if k == "w":
                    a1 = -1
                elif k == "s":
                    a1 = 1
                elif k == "q":
                    break
            g.step(a1, a2)
            g.render()
            time.sleep(0.04)
    except KeyboardInterrupt:
        pass

    print(f"\nfinal: you {g.score1} - ai {g.score2}")


def do_demo(args):
    from train import load_brain, demo_game
    brain = load_brain(args.load)
    if brain is None:
        print(f"cant load brain from '{args.load}'")
        sys.exit(1)
    hits, s1, s2 = demo_game(brain)
    print(f"demo: {hits} hits, score {s1}-{s2}")


def main():
    p = argparse.ArgumentParser(description="scratch pong - nn from scratch")

    # kinda janky but works for switching subcommands
    if len(sys.argv) < 2:
        p.print_help()
        return

    cmd = sys.argv[1]
    rest = sys.argv[2:]

    if cmd == "train":
        sp = argparse.ArgumentParser()
        sp.add_argument("--pop", type=int, default=POP_SIZE)
        sp.add_argument("--hidden", type=int, default=10)
        sp.add_argument("--games", type=int, default=2)
        sp.add_argument("--show", type=int, default=5)
        sp.add_argument("--save-every", type=int, default=10)
        sp.add_argument("--save", default="best_brain.json")
        sp.add_argument("--seed", type=int, default=None)
        do_train(sp.parse_args(rest))

    elif cmd == "watch":
        sp = argparse.ArgumentParser()
        sp.add_argument("--load", default="best_brain.json")
        do_watch(sp.parse_args(rest))

    elif cmd == "play":
        sp = argparse.ArgumentParser()
        sp.add_argument("--load", default="best_brain.json")
        do_play(sp.parse_args(rest))

    elif cmd == "demo":
        sp = argparse.ArgumentParser()
        sp.add_argument("--load", default="best_brain.json")
        do_demo(sp.parse_args(rest))

    else:
        print(f"unknown command: {cmd}")
        print("try: train, watch, play, demo")


if __name__ == "__main__":
    main()
