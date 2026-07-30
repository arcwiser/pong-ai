#!/usr/bin/env python3
import argparse
import sys
import time

from nn import NeuralNetwork
from pong import Pong
from train import continuous_train, POP_SIZE


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
        fps=args.fps,
    )


def do_watch(args):
    from train import load_brain
    brain = load_brain(args.load)
    if brain is None:
        print(f"cant load brain from '{args.load}' - did you train one?")
        sys.exit(1)

    print("watching ai vs ai - ctrl+c to quit")
    speed_mult = args.fps / 25.0
    try:
        while True:
            g = Pong(ball_speed=0.9 * speed_mult, paddle_speed=0.6 * speed_mult)
            while not g.done:
                s = g.get_state(for_player=2)
                out = brain.forward(s)
                a2 = 1 if out[0] > 0.5 else -1
                diff = g.ball_y - g.paddle1_y
                a1 = 0 if abs(diff) < 0.8 else (1 if diff > 0 else -1)
                g.step(a1, a2)
                g.render()
                time.sleep(1.0 / args.fps)
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

    speed_mult = args.fps / 25.0
    g = Pong(ball_speed=0.9 * speed_mult, paddle_speed=0.6 * speed_mult)
    print("you are player 1 (left). w=up  s=down  q=quit")
    time.sleep(1)

    try:
        while not g.done:
            s = g.get_state(for_player=2)
            out = brain.forward(s)
            a2 = 1 if out[0] > 0.5 else -1
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
            time.sleep(1.0 / args.fps)
    except KeyboardInterrupt:
        pass

    print(f"\nfinal: you {g.score1} - ai {g.score2}")


def do_demo(args):
    from train import load_brain, demo_game
    brain = load_brain(args.load)
    if brain is None:
        print(f"cant load brain from '{args.load}'")
        sys.exit(1)
    hits, s1, s2 = demo_game(brain, getattr(args, 'fps', 25.0))
    print(f"demo: {hits} hits, score {s1}-{s2}")


def interactive_menu():
    import argparse
    import json
    import os
    
    current_fps = 25.0
    if os.path.exists('settings.json'):
        try:
            with open('settings.json', 'r') as f:
                current_fps = json.load(f).get('fps', 25.0)
        except:
            pass

    while True:
        print("\n=======================")
        print("       PONG AI         ")
        print("=======================")
        print("1. Train AI")
        print("2. Watch AI vs AI")
        print("3. Play vs AI")
        print("4. Demo (Quick simulation)")
        print("5. Settings")
        print("q. Quit")
        choice = input("Select an option: ").strip().lower()

        if choice == '1':
            sp = argparse.ArgumentParser()
            sp.add_argument("--pop", type=int, default=POP_SIZE)
            sp.add_argument("--hidden", type=int, default=10)
            sp.add_argument("--games", type=int, default=2)
            sp.add_argument("--show", type=int, default=5)
            sp.add_argument("--save-every", type=int, default=10)
            sp.add_argument("--save", default="best_brain.json")
            sp.add_argument("--seed", type=int, default=None)
            sp.add_argument("--fps", type=float, default=current_fps)
            do_train(sp.parse_args([]))
        elif choice == '2':
            sp = argparse.ArgumentParser()
            sp.add_argument("--load", default="best_brain.json")
            sp.add_argument("--fps", type=float, default=current_fps)
            do_watch(sp.parse_args([]))
        elif choice == '3':
            sp = argparse.ArgumentParser()
            sp.add_argument("--load", default="best_brain.json")
            sp.add_argument("--fps", type=float, default=current_fps)
            do_play(sp.parse_args([]))
        elif choice == '4':
            sp = argparse.ArgumentParser()
            sp.add_argument("--load", default="best_brain.json")
            sp.add_argument("--fps", type=float, default=current_fps)
            do_demo(sp.parse_args([]))
        elif choice == '5':
            print(f"\nCurrent Game Speed (FPS): {current_fps}")
            try:
                new_fps = float(input("Enter new game speed (e.g., 25 for normal, 60 for fast): "))
                if new_fps > 0:
                    current_fps = new_fps
                    with open('settings.json', 'w') as f:
                        json.dump({'fps': current_fps}, f)
                    print(f"Speed updated to {current_fps} FPS and saved.")
                else:
                    print("Speed must be greater than 0.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        elif choice == 'q':
            break
        else:
            print("Invalid option.")

def main():
    p = argparse.ArgumentParser(description="scratch pong - nn from scratch")
    if len(sys.argv) < 2:
        interactive_menu()
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
        sp.add_argument("--fps", type=float, default=25.0)
        do_train(sp.parse_args(rest))

    elif cmd == "watch":
        sp = argparse.ArgumentParser()
        sp.add_argument("--load", default="best_brain.json")
        sp.add_argument("--fps", type=float, default=25.0)
        do_watch(sp.parse_args(rest))

    elif cmd == "play":
        sp = argparse.ArgumentParser()
        sp.add_argument("--load", default="best_brain.json")
        sp.add_argument("--fps", type=float, default=25.0)
        do_play(sp.parse_args(rest))

    elif cmd == "demo":
        sp = argparse.ArgumentParser()
        sp.add_argument("--load", default="best_brain.json")
        sp.add_argument("--fps", type=float, default=25.0)
        do_demo(sp.parse_args(rest))

    else:
        print(f"unknown command: {cmd}")
        print("try: train, watch, play, demo")


if __name__ == "__main__":
    import os
    os.system("") # enables ANSI escape codes on older windows terminals
    import multiprocessing
    multiprocessing.freeze_support()
    main()
