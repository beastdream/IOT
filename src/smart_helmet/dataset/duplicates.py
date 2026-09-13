"""Content-based duplicate candidates; never remove files."""

from collections import defaultdict
import hashlib
from pathlib import Path
import re

import numpy as np
from PIL import Image

PHASH_VERY_SIMILAR = 4
PHASH_POSSIBLY_SIMILAR = 8


def source_group_id(filename: str) -> str:
    return re.sub(r"\.rf\.[0-9a-fA-F]+$", "", Path(filename).stem)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(image: Image.Image) -> str:
    """63 AC bits from a 32x32 grayscale DCT; the DC bit is fixed to zero."""
    pixels = np.asarray(image.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    k = np.arange(8)[:, None]
    x = np.arange(32)[None, :]
    basis = np.cos(np.pi * (2 * x + 1) * k / 64) * np.sqrt(2 / 32)
    basis[0] /= np.sqrt(2)
    low = (basis @ pixels @ basis.T).ravel()
    bits = low > np.median(low[1:])
    bits[0] = False
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f'{value:016x}'


def hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def similarity(distance: int) -> str | None:
    if distance <= PHASH_VERY_SIMILAR:
        return 'VERY_SIMILAR'
    if distance <= PHASH_POSSIBLY_SIMILAR:
        return 'POSSIBLY_SIMILAR'
    return None


def grouped_records(records: list[dict], key: str) -> list[list[dict]]:
    groups = defaultdict(list)
    for row in records:
        if row.get(key):
            groups[row[key]].append(row)
    return list(groups.values())


def exact_duplicates(records: list[dict]) -> list[dict]:
    result = []
    for group in grouped_records(records, 'sha256'):
        if len(group) < 2:
            continue
        splits = sorted({r['split'] for r in group})
        result.append(dict(sha256=group[0]['sha256'], count=len(group),
                           splits='|'.join(splits), cross_split=len(splits) > 1,
                           severity='CRITICAL' if len(splits) > 1 else 'REVIEW',
                           images='|'.join(r['image_path'] for r in group)))
    return result


def near_duplicates(records: list[dict]) -> list[dict]:
    usable = [r for r in records if r.get('perceptual_hash')]
    hashes = [int(r['perceptual_hash'], 16) for r in usable]
    pairs = []
    for i, left in enumerate(usable):
        for j in range(i + 1, len(usable)):
            distance = (hashes[i] ^ hashes[j]).bit_count()
            kind = similarity(distance)
            if kind is None:
                continue
            right = usable[j]
            cross = left['split'] != right['split']
            pairs.append(dict(image_a=left['image_path'], image_b=right['image_path'],
                              split_a=left['split'], split_b=right['split'],
                              distance=distance, similarity=kind,
                              perceptual_match=distance == 0, cross_split=cross,
                              severity='HIGH' if cross else 'REVIEW'))
    return pairs


def connected_group_count(pairs: list[dict]) -> int:
    """Candidate components; transitive membership does not prove equivalence."""
    parents = {}

    def find(value):
        parents.setdefault(value, value)
        if parents[value] != value:
            parents[value] = find(parents[value])
        return parents[value]

    for pair in pairs:
        a, b = find(pair['image_a']), find(pair['image_b'])
        parents[a] = b
    return len({find(value) for value in parents})
