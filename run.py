import time
import generate_data
import queries
import visualize


def main():
    start = time.time()

    print("=" * 50)
    print("  StackOverflow Analytics Pipeline")
    print("=" * 50)

    print("\n[1/3] Generating dataset...")
    generate_data.run()

    print("[2/3] Running SQL queries...")
    queries.run()

    print("[3/3] Building charts...")
    visualize.run()

    elapsed = round(time.time() - start, 1)
    print("=" * 50)
    print(f"  Pipeline complete in {elapsed}s")
    print("=" * 50)


if __name__ == "__main__":
    main()