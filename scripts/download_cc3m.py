"""Async image downloader for CC3M.

Downloads images from the CC3M TSV (caption<TAB>url format) into a flat
directory using content-hashed filenames. Designed for fault tolerance:
  - Skips already-downloaded images (resume-friendly)
  - Ignores individual failures (404s, timeouts, corrupt data)
  - Validates that downloaded bytes are actually decodable images
  - Progress reporting with throughput stats

Usage:
    uv run python -m scripts.download_cc3m \
        --tsv data/cc3m/Train_GCC-training.tsv \
        --output-dir data/cc3m/images \
        --max-concurrent 128 \
        --timeout 10

For a test run on a small slice:
    uv run python -m scripts.download_cc3m \
        --tsv data/cc3m/Train_GCC-training.tsv \
        --output-dir data/cc3m/images \
        --limit 1000
"""

import argparse
import asyncio
import csv
import logging
import time
from pathlib import Path

import aiohttp
from PIL import Image
import io

from src.dataset import url_to_filename

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class DownloadStats:
    """Thread-safe-ish counters for progress reporting."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.success = 0
        self.skipped = 0  # Already existed on disk
        self.failed = 0
        self.start_time = time.monotonic()

    @property
    def processed(self) -> int:
        return self.success + self.skipped + self.failed

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start_time

    @property
    def rate(self) -> float:
        elapsed = self.elapsed
        return self.processed / elapsed if elapsed > 0 else 0.0

    def log_progress(self) -> None:
        pct = 100 * self.processed / self.total if self.total else 0
        logger.info(
            f"[{pct:5.1f}%] {self.processed}/{self.total} "
            f"({self.success} ok, {self.skipped} skip, {self.failed} fail) "
            f"@ {self.rate:.0f} img/s"
        )


async def download_one(
    session: aiohttp.ClientSession,
    url: str,
    output_path: Path,
    timeout: float,
    stats: DownloadStats,
    min_size: int = 1024,
) -> bool:
    """Download a single image, validate it, and save to disk.

    Returns True if the image was successfully saved (or already existed).
    """
    if output_path.exists():
        stats.skipped += 1
        return True

    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=timeout),
            allow_redirects=True,
        ) as resp:
            if resp.status != 200:
                stats.failed += 1
                return False

            data = await resp.read()

            # Reject tiny responses — likely error pages, not images
            if len(data) < min_size:
                stats.failed += 1
                return False

            # Validate that the bytes actually decode as an image.
            # This catches HTML error pages, truncated downloads, etc.
            try:
                img = Image.open(io.BytesIO(data))
                img.verify()
            except Exception:
                stats.failed += 1
                return False

            output_path.write_bytes(data)
            stats.success += 1
            return True

    except Exception:
        # Timeouts, DNS failures, connection resets — all expected at
        # scale. Individual failures don't matter; we just need enough
        # images to train on.
        stats.failed += 1
        return False


async def download_batch(
    urls: list[tuple[str, Path]],
    max_concurrent: int,
    timeout: float,
) -> DownloadStats:
    """Download images with bounded concurrency."""
    stats = DownloadStats(total=len(urls))
    semaphore = asyncio.Semaphore(max_concurrent)

    # Connection pooling: reuse TCP connections across requests.
    # limit_per_host prevents hammering any single CDN.
    connector = aiohttp.TCPConnector(
        limit=max_concurrent,
        limit_per_host=32,
        ttl_dns_cache=300,
    )

    async with aiohttp.ClientSession(
        connector=connector,
        headers={"User-Agent": "CC3M-Dataset-Downloader/1.0"},
    ) as session:

        async def bounded_download(url: str, path: Path) -> bool:
            async with semaphore:
                return await download_one(session, url, path, timeout, stats)

        # Fire all downloads concurrently (bounded by semaphore)
        tasks = [bounded_download(url, path) for url, path in urls]

        # Report progress periodically while tasks complete
        log_interval = max(len(tasks) // 20, 100)  # ~20 progress lines
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            await coro
            if (i + 1) % log_interval == 0:
                stats.log_progress()

    stats.log_progress()  # Final report
    return stats


def load_urls_from_tsv(
    tsv_path: Path,
    output_dir: Path,
    limit: int | None = None,
) -> list[tuple[str, Path]]:
    """Parse TSV and return (url, output_path) pairs."""
    urls = []
    with open(tsv_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            url = row[1]
            output_path = output_dir / url_to_filename(url)
            urls.append((url, output_path))
            if limit and len(urls) >= limit:
                break
    return urls


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download CC3M images (async, fault-tolerant)"
    )
    parser.add_argument(
        "--tsv", type=Path, required=True,
        help="Path to CC3M TSV file (caption<TAB>url)",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to save downloaded images",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=128,
        help="Max simultaneous downloads (default: 128)",
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0,
        help="Per-request timeout in seconds (default: 10)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only download first N images (for testing)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading URLs from {args.tsv}...")
    urls = load_urls_from_tsv(args.tsv, args.output_dir, limit=args.limit)
    logger.info(f"Found {len(urls)} URLs to process")

    stats = asyncio.run(download_batch(
        urls,
        max_concurrent=args.max_concurrent,
        timeout=args.timeout,
    ))

    logger.info(
        f"\nDone in {stats.elapsed:.0f}s: "
        f"{stats.success} downloaded, {stats.skipped} already existed, "
        f"{stats.failed} failed ({100 * stats.failed / stats.total:.1f}% failure rate)"
    )


if __name__ == "__main__":
    main()
