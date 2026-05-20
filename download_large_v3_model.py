import time
from pathlib import Path

from huggingface_hub import hf_hub_download


BASE_DIR = Path(__file__).resolve().parent
TARGET_DIR = BASE_DIR / "models" / "faster-whisper-large-v3"

REPO_ID = "Systran/faster-whisper-large-v3"
FILENAME = "model.bin"

MAX_RETRIES = 20
WAIT_SECONDS = 20


def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\nAttempt {attempt}/{MAX_RETRIES}")
        print(f"Downloading {FILENAME} from {REPO_ID}")
        print(f"Target folder: {TARGET_DIR}")

        try:
            file_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=FILENAME,
                local_dir=str(TARGET_DIR),
                local_dir_use_symlinks=False,
                resume_download=True,
            )

            final_path = Path(file_path)
            size_gb = final_path.stat().st_size / (1024 ** 3)

            print("\nDownload finished.")
            print(f"File: {final_path}")
            print(f"Size: {size_gb:.2f} GB")

            if size_gb < 2.5:
                print("Warning: file size looks too small. Retrying...")
                time.sleep(WAIT_SECONDS)
                continue

            print("model.bin looks complete.")
            return

        except KeyboardInterrupt:
            print("\nDownload stopped by user.")
            return

        except Exception as error:
            print(f"\nDownload failed: {error}")
            print(f"Waiting {WAIT_SECONDS} seconds before retrying...")
            time.sleep(WAIT_SECONDS)

    print("\nDownload did not complete after all retries.")


if __name__ == "__main__":
    main()