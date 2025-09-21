import argparse
from src.inference import generate_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt',
                        type=str,
                        required=True,
                        help='Input prompt')
    parser.add_argument('--length',
                        type=int,
                        default=100,
                        help="No. of tokens to generate")
    
    args = parser.parse_args()

    try:
        generated = generate_text(start=args.prompt,length=args.length)
        print(f"Prompt:{args.prompt}")
        print(f"\nGenerated Text:{generated}")
    except Exception as e:
        print(f"Error generating text:{e}")

if __name__ == '__main__':
    main()