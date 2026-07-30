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


# pit two trained brains against each other
def do_fight(args):
    from train import load_brain
    brain1 = load_brain(args.load1)
    brain2 = load_brain(args.load2)
    if brain1 is None:
        print(f"cant load brain 1 from '{args.load1}'")
        sys.exit(1)
    if brain2 is None:
        print(f"cant load brain 2 from '{args.load2}'")
        sys.exit(1)

    speed_mult = args.fps / 25.0
    rounds = args.rounds
    wins1 = 0
    wins2 = 0
    total_s1 = 0
    total_s2 = 0

    print(f"\n  AI FIGHT: {args.load1} vs {args.load2}")
    print(f"  {rounds} round(s) at {args.fps} FPS")
    print(f"  {'visual mode' if args.visual else 'fast sim mode'}")
    print()

    for r in range(1, rounds + 1):
        g = Pong(ball_speed=0.9 * speed_mult, paddle_speed=0.6 * speed_mult)

        while not g.done:
            # brain 1 controls left paddle (player 1)
            s1 = g.get_state(for_player=1)
            out1 = brain1.forward(s1)
            a1 = 1 if out1[0] > 0.5 else -1

            # brain 2 controls right paddle (player 2)
            s2 = g.get_state(for_player=2)
            out2 = brain2.forward(s2)
            a2 = 1 if out2[0] > 0.5 else -1

            g.step(a1, a2)

            if args.visual:
                g.render()
                time.sleep(1.0 / args.fps)

        total_s1 += g.score1
        total_s2 += g.score2
        if g.score1 > g.score2:
            wins1 += 1
            winner = args.load1
        else:
            wins2 += 1
            winner = args.load2

        print(f"  round {r}: {g.score1}-{g.score2}  winner: {winner}")

        if args.visual and r < rounds:
            time.sleep(1.0)

    # final results
    print()
    print("  ========== RESULTS ==========")
    print(f"  {args.load1}: {wins1} wins  ({total_s1} total points)")
    print(f"  {args.load2}: {wins2} wins  ({total_s2} total points)")
    if wins1 > wins2:
        print(f"  CHAMPION: {args.load1}")
    elif wins2 > wins1:
        print(f"  CHAMPION: {args.load2}")
    else:
        print("  DRAW!")
    print("  =============================")


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
        print("2. Watch AI vs Simple AI")
        print("3. Play vs AI")
        print("4. Demo (Quick simulation)")
        print("5. Fight (AI vs AI)")
        print("6. Settings")
        print("q. Quit")
        choice = input("Select an option: ").strip().lower()

        if choice == '1':
            name = input("Save file name (default: best_brain.json): ").strip()
            if not name:
                name = "best_brain.json"
            if not name.endswith(".json"):
                name += ".json"
            sp = argparse.ArgumentParser()
            sp.add_argument("--pop", type=int, default=POP_SIZE)
            sp.add_argument("--hidden", type=int, default=10)
            sp.add_argument("--games", type=int, default=2)
            sp.add_argument("--show", type=int, default=5)
            sp.add_argument("--save-every", type=int, default=10)
            sp.add_argument("--save", default=name)
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
            # list available brains
            brain_files = [f for f in os.listdir('.') if f.endswith('.json') and f != 'settings.json']
            if len(brain_files) < 2:
                print(f"\nYou need at least 2 trained brains to fight!")
                print(f"Found: {brain_files if brain_files else 'none'}")
                print("Train more models first (option 1) with different save names.")
                continue

            print("\nAvailable brains:")
            for i, f in enumerate(brain_files, 1):
                print(f"  {i}. {f}")

            try:
                p1 = input(f"Pick brain 1 (1-{len(brain_files)}): ").strip()
                p2 = input(f"Pick brain 2 (1-{len(brain_files)}): ").strip()
                b1 = brain_files[int(p1) - 1]
                b2 = brain_files[int(p2) - 1]
            except (ValueError, IndexError):
                print("Invalid selection.")
                continue

            try:
                rounds = int(input("How many rounds? (default: 5): ").strip() or "5")
            except ValueError:
                rounds = 5

            visual = input("Watch the fights? (y/n, default: y): ").strip().lower()
            visual = visual != 'n'

            sp = argparse.ArgumentParser()
            sp.add_argument("--load1", default=b1)
            sp.add_argument("--load2", default=b2)
            sp.add_argument("--fps", type=float, default=current_fps)
            sp.add_argument("--rounds", type=int, default=rounds)
            sp.add_argument("--visual", type=bool, default=visual)
            do_fight(sp.parse_args([]))
        elif choice == '6':
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

    elif cmd == "fight":
        sp = argparse.ArgumentParser()
        sp.add_argument("--load1", required=True)
        sp.add_argument("--load2", required=True)
        sp.add_argument("--fps", type=float, default=25.0)
        sp.add_argument("--rounds", type=int, default=5)
        sp.add_argument("--visual", action="store_true")
        do_fight(sp.parse_args(rest))

    else:
        print(f"unknown command: {cmd}")
        print("try: train, watch, play, demo, fight")


if __name__ == "__main__":
    import os
    os.system("") # enables ANSI escape codes on older windows terminals
    import multiprocessing
    multiprocessing.freeze_support()
    main()
