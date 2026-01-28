import argparse
from textwrap import dedent
from transformers import pipeline, set_seed

def build_prompt(bullets):
    # Format bullets into a clear list
    if len(bullets) == 1:
        context = bullets[0]
    elif len(bullets) == 2:
        context = f"{bullets[0]} and {bullets[1]}"
    else:
        context = ", ".join(bullets[:-1]) + f", and {bullets[-1]}"
    
    prompt = f"Write a professional email to a manager about the following points: {context}.\n\nEmail:\nDear Manager,\n\nI am writing to inform you that"
    return prompt, context

def generate_email(prompt: str,
                   max_new_tokens: int = 200,
                   temperature: float = 0.8,
                   top_k: int = 50,
                   top_p: float = 0.95,
                   seed: int = 42) -> str:
    try:
        set_seed(seed)
        
        # Load model with error handling
        try:
            generator = pipeline(
                "text-generation",
                model="distilgpt2"
            )
        except Exception as e:
            return f"Error loading model: {str(e)}\n\nPlease ensure transformers and torch are installed."
        
        # Generate text with proper max_new_tokens (no override)
        try:
            outputs = generator(
                prompt,
                max_new_tokens=max_new_tokens,  # Use the actual parameter value
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=1,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3
            )
        except Exception as e:
            return f"Error during text generation: {str(e)}"
        
        full_text = outputs[0]["generated_text"]
        generated_part = full_text[len(prompt):].strip()
        
        
        # Construct full email with standard professional text
        email = (f"Dear Manager,\n\n"
                 f"I am writing to inform you that {generated_part}\n\n"
                 f"I apologize for any inconvenience this may cause. Thank you for your understanding.\n\n"
                 f"Best regards,\n"
                 f"[Your Name]")
        
        return email
        
    except Exception as e:
        # Catch-all error handler
        return f"Unexpected error: {str(e)}\n\nPlease check your inputs and try again."

def read_bullets_from_user() :
    print("Enter bullet points for the email (empty line to finish):")
    bullets = []
    while True:
        line = input("- ").strip()
        if line == "":
            break
        bullets.append(line)
    if not bullets:
        raise ValueError("No bullet points provided.")
    return bullets

def parse_args():
    parser = argparse.ArgumentParser(
        description="Email Auto-Drafter using distilgpt2 (formal, polite emails)."
    )
    parser.add_argument(
        "--bullets",
        nargs="+",
        help="Bullet points as command-line arguments (e.g. --bullets 'Sick leave' Monday 'Back Tuesday')."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to generate for the email."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (higher = more random)."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter."
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling parameter."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    if args.bullets:
        bullets = args.bullets
    else:
        bullets = read_bullets_from_user()

    print("\n[Bullets]")
    for b in bullets:
        print(f"- {b}")

    prompt, context = build_prompt(bullets)
    print("\n[Prompt sent to model]")
    print(prompt)

    print("\n[Generated Email]\n")
    email = generate_email(
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )
    print(email)

if __name__ == "__main__":
    main()
