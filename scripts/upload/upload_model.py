import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import create_repo, upload_folder


def main():
    """Main function to handle model preparation and upload."""
    parser = argparse.ArgumentParser(description="Upload a custom Hugging Face model with its self-contained code.")
    parser.add_argument(
        "--model_code_path",
        type=str,
        required=True,
        help="Path to the self-contained, single Python model file.",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Directory containing the model weights and tokenizer files (hf_ckpt).",
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Name of the repository on Hugging Face Hub (e.g., 'username/repo-name').",
    )
    parser.add_argument(
        "--readme_path",
        type=str,
        required=True,
        help="Path to the README.md file to be included in the repository.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="If set, creates a private repository.",
    )

    args = parser.parse_args()

    staging_dir = Path("./temp_upload_staging")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir()
    print(f"Created temporary staging directory: {staging_dir}")

    try:
        # --- 2. Copy All Necessary Files ---
        print("\nCopying files to staging directory...")

        # Copy checkpoint files
        for f in os.listdir(args.ckpt_dir):
            shutil.copy(os.path.join(args.ckpt_dir, f), staging_dir)

        # Copy the single, self-contained model code file
        model_code_source = Path(args.model_code_path)
        if not model_code_source.exists():
            print(f"Error: Model code file not found at {model_code_source}")
            sys.exit(1)

        # The destination file MUST be named correctly for auto_map to work.
        model_code_dest = staging_dir / "modeling_qwen2.py"
        print(f"Copying model code from {model_code_source} to {model_code_dest}")
        shutil.copy(model_code_source, model_code_dest)

        print("File copying complete.")

        # --- 3. Configure `config.json` for Auto-Loading ---
        print("\nConfiguring config.json for auto-loading...")
        config_path = staging_dir / "config.json"
        if not config_path.exists():
            print(f"Error: config.json not found in {args.ckpt_dir}")
            sys.exit(1)

        with open(config_path, encoding="utf-8") as f:
            config_data = json.load(f)

        config_data["auto_map"] = {"AutoModelForCausalLM": "modeling_qwen2.Qwen2ForCausalLM"}
        config_data["architectures"] = ["Qwen2ForCausalLM"]
        config_data["trust_remote_code"] = True

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
        print("config.json updated successfully.")

        # --- 4. Copy `README.md` ---
        print("\nCopying README.md...")
        readme_source = Path(args.readme_path)
        if not readme_source.exists():
            print(f"Error: README file not found at {readme_source}")
            sys.exit(1)

        with open(readme_source, encoding="utf-8") as f:
            readme_content = f.read()

        readme_content = readme_content.replace("{repo_id}", args.repo)

        with open(staging_dir / "README.md", "w", encoding="utf-8") as f:
            f.write(readme_content)
        print("README.md copied and processed.")

        # --- 5. Upload to the Hub ---
        print(f"\nPreparing to upload to repository: {args.repo}")
        repo_url = create_repo(args.repo, repo_type="model", exist_ok=True, private=args.private)

        upload_folder(
            folder_path=staging_dir,
            repo_id=args.repo,
            repo_type="model",
            commit_message="Initial model upload with self-contained custom code",
        )
        print("\n🚀 Upload complete! 🚀")
        print(f"Check out your model at: {repo_url}")

    finally:
        # --- 6. Clean Up ---
        print("\nCleaning up temporary staging directory...")
        shutil.rmtree(staging_dir)
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
