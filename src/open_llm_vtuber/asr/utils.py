import requests
import shutil
import tarfile
import tempfile
from pathlib import Path
from tqdm import tqdm
from loguru import logger


def get_github_asset_url(owner, repo, release_tag, filename_without_ext):
    """
    Fetch the URL of a GitHub release asset by its filename (without extension).

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        release_tag (str): The tag of the release.
        filename_without_ext (str): The filename to search for (without extension).

    Returns:
        str: The download URL of the matched asset, or None if no match is found.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{release_tag}"
    headers = {}  # Add authentication headers if needed

    try:
        # Make a GET request to fetch release data
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Parse the JSON response
        release_data = response.json()
        assets = release_data.get("assets", [])

        # Look for a matching file
        for asset in assets:
            if asset["name"].startswith(filename_without_ext):
                logger.info(f"Match found: {asset['name']}")
                return asset["browser_download_url"]

        # If no match found, log the error
        logger.error(
            f"No match found for filename: {filename_without_ext} in release {release_tag}."
        )
        return None

    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred while fetching release data: {e}")
        return None


def download_and_extract(url: str, output_dir: str, force: bool = False) -> Path:
    """
    Download a file from a URL and extract it if it is a tar.bz2 archive.

    Args:
        url (str): The URL to download the file from.
        output_dir (str): The directory to save the downloaded file.

    Returns:
        Path: Path to the extracted directory if it's a tar.bz2 file,
             otherwise Path to the downloaded file.
    """
    # Create the output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get the file name from the URL
    file_name = url.split("/")[-1]
    file_path = Path(output_dir) / file_name
    partial_path = file_path.with_suffix(f"{file_path.suffix}.part")

    # Extract the root directory name from the filename (removing .tar.bz2)
    root_dir = file_name.replace(".tar.bz2", "")
    extracted_dir_path = Path(output_dir) / root_dir

    # Check if the extracted directory already exists
    if extracted_dir_path.exists() and not force:
        logger.info(
            f"✅ The directory {extracted_dir_path} already exists. I would assume that the model is already downloaded and we are ready to go. Skipping download and extraction."
        )
        return extracted_dir_path

    # Download the file
    logger.info(f"🏃‍♂️Downloading {url} to {file_path}...")
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            logger.debug(f"Total file size: {total_size / 1024 / 1024:.2f} MB")

            with (
                partial_path.open("wb") as file,
                tqdm(
                    desc=file_name,
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar,
            ):
                for chunk in response.iter_content(chunk_size=8192):
                    size = file.write(chunk)
                    pbar.update(size)
        if total_size and partial_path.stat().st_size != total_size:
            raise IOError(
                f"Downloaded size mismatch for {file_name}: "
                f"{partial_path.stat().st_size} != {total_size}"
            )
        partial_path.replace(file_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise

    logger.info(f"Downloaded {file_name} successfully.")

    # Extract the tar.bz2 file
    if file_name.endswith(".tar.bz2"):
        logger.info(f"Extracting {file_name}...")
        _extract_archive(file_path, Path(output_dir), root_dir)
        logger.info("Extraction completed.")
        return extracted_dir_path
    else:
        logger.warning("The downloaded file is not a tar.bz2 archive.")
        return Path(file_path)


def check_and_extract_local_file(url: str, output_dir: str) -> Path | None:
    """
    Check if a local file exists and extract it if it is a tar.bz2 archive.

    Args:
        url (str): The URL of the file.
        output_dir (str): The directory to save the extracted files.

    Returns:
        Path | None: Path to the extracted directory if it's a tar.bz2 file,
            otherwise None.
    """
    # Get the file name from the URL
    file_name = url.split("/")[-1]
    compressed_path = Path(output_dir) / file_name

    # Check if the compressed file exists and is a tar.bz2 archive
    extracted_dir = Path(output_dir) / file_name.replace(".tar.bz2", "")

    if extracted_dir.exists() and not compressed_path.exists():
        logger.info(
            f"✅ Extracted directory exists: {extracted_dir}, no operation needed."
        )
        return extracted_dir

    if compressed_path.exists() and file_name.endswith(".tar.bz2"):
        logger.info(f"🔍 Found local archive file: {compressed_path}")

        try:
            logger.info("⏳ Extracting archive file...")
            _extract_archive(
                compressed_path,
                Path(output_dir),
                file_name.replace(".tar.bz2", ""),
            )
            logger.success(f"Extracted archive to the path: {extracted_dir}")
            return extracted_dir
        except Exception as e:
            logger.error(f"Fail to extract file: {str(e)}")
            return None

    logger.warning(f"Local file not found or not a tar.bz2 archive: {compressed_path}")
    return None


def _extract_archive(archive: Path, output_dir: Path, root_dir: str) -> None:
    staging = Path(tempfile.mkdtemp(prefix=f".{root_dir}-", dir=output_dir))
    extracted = output_dir / root_dir
    backup = output_dir / f".{root_dir}.previous"
    try:
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(path=staging)
        candidate = staging / root_dir
        if not candidate.is_dir():
            raise IOError(f"Archive did not contain the expected directory: {root_dir}")
        if backup.exists():
            shutil.rmtree(backup)
        if extracted.exists():
            extracted.replace(backup)
        try:
            candidate.replace(extracted)
        except Exception:
            if backup.exists() and not extracted.exists():
                backup.replace(extracted)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        archive.unlink()
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2"
    output_dir = "./models"

    # Try local extraction first.
    local_result = check_and_extract_local_file(url, output_dir)

    # Download if not available locally.
    if local_result is None:
        logger.info("Local archive not found. Starting download...")
        download_and_extract(url, output_dir)
    else:
        logger.info("Extraction completed using local file.")
