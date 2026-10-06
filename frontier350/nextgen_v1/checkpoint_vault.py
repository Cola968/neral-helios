"""Local append-only checkpoint vault for Neral. Standard library; no torch.load.

Protect copies to a NEW directory, hashes source/replica independently, flushes
file contents, then publishes a manifest LAST. No source file is ever modified.
A volume-separated copy is NOT an off-site backup or a zero-loss guarantee.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import time

SHA = re.compile(r"[0-9a-f]{64}\Z")
LABEL = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
CHUNK = 4 * 1024 * 1024
SPACE_BUFFER = 1024 ** 3


def pinned_sha(value):
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ValueError('independently pinned lowercase SHA-256 required')
    return value


def regular(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('symlink file rejected')
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_BINARY', 0))
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
        os.close(fd)
        raise ValueError('nonempty regular file required')
    return fd, info


def assert_unchanged(fd, initial):
    now = os.fstat(fd)
    attrs = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    if any(getattr(initial, key) != getattr(now, key) for key in attrs):
        raise ValueError('input changed while being protected')


def hash_fd(fd):
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        block = os.read(fd, CHUNK)
        if not block:
            break
        digest.update(block)
    return digest.hexdigest()


def hash_regular(path):
    fd, info = regular(path)
    try:
        digest = hash_fd(fd)
        assert_unchanged(fd, info)
        return digest, info.st_size
    finally:
        os.close(fd)


def sync_directory(directory):
    """False explicitly records volumes lacking directory fsync support."""
    try:
        fd = os.open(directory, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    except OSError:
        return False
    try:
        try:
            os.fsync(fd)
        except OSError:
            return False
        return True
    finally:
        os.close(fd)


def copy_checked(source, destination, expected):
    fd, info = regular(source)
    try:
        if hash_fd(fd) != expected:
            raise ValueError('source checksum differs from independent pin')
        assert_unchanged(fd, info)
        os.lseek(fd, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        destfd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                         getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0), 0o600)
        try:
            while True:
                block = os.read(fd, CHUNK)
                if not block:
                    break
                digest.update(block)
                offset = 0
                while offset < len(block):
                    offset += os.write(destfd, block[offset:])
            os.fsync(destfd)
        finally:
            os.close(destfd)
        assert_unchanged(fd, info)
        if digest.hexdigest() != expected:
            raise ValueError('source changed during copy')
        replica_hash, replica_size = hash_regular(destination)
        if not hmac.compare_digest(replica_hash, expected) or replica_size != info.st_size:
            raise ValueError('replica verification failed')
        return info.st_size
    finally:
        os.close(fd)


def read_manifest(path):
    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ValueError('duplicate manifest field')
            obj[key] = value
        return obj
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('invalid JSON constant')))


def verify(backup, expected_sha):
    expected_sha = pinned_sha(expected_sha)
    backup = Path(backup)
    if backup.is_symlink() or not backup.is_dir():
        raise ValueError('backup directory must be a real directory')
    if {p.name for p in backup.iterdir()} != {'resume.pt', 'manifest.json'}:
        raise ValueError('incomplete or unexpected backup contents')
    manifest = read_manifest(backup / 'manifest.json')
    if not isinstance(manifest, dict) or manifest.get('schema') != 1 or manifest.get('complete') is not True:
        raise ValueError('uncommitted backup')
    if manifest.get('sha256') != expected_sha:
        raise ValueError('manifest digest differs from independent pin')
    if not isinstance(manifest.get('bytes'), int) or manifest['bytes'] <= 0:
        raise ValueError('invalid manifest size')
    if not LABEL.fullmatch(str(manifest.get('label', ''))):
        raise ValueError('invalid backup label')
    actual, size = hash_regular(backup / 'resume.pt')
    if not hmac.compare_digest(actual, expected_sha) or size != manifest['bytes']:
        raise ValueError('backup data corrupt or truncated')
    return {'schema': 1, 'status': 'verified', 'sha256': actual,
            'bytes': size, 'backup': str(backup), 'label': manifest['label']}


def protect(source, backup_root, expected_sha, label, *, different_volume=False):
    expected_sha = pinned_sha(expected_sha)
    if not LABEL.fullmatch(label):
        raise ValueError('invalid backup label')
    source = Path(source)
    root = Path(backup_root)
    if root.is_symlink() or root.parent.is_symlink():
        raise ValueError('symlink backup root rejected')
    # Never create a backup inside the source file's directory.
    if root.resolve() == source.parent.resolve() or source.parent.resolve() in root.resolve().parents:
        raise ValueError('backup must be outside source directory tree')
    root.mkdir(parents=True, exist_ok=True)
    if different_volume and os.stat(source).st_dev == os.stat(root).st_dev:
        raise ValueError('backup and source must reside on distinct mounted volumes')
    target = root / (label + '-' + expected_sha)
    if target.exists() or target.is_symlink():
        result = verify(target, expected_sha)
        if result['label'] != label:
            raise ValueError('existing backup label differs')
        return dict(result, status='verified_existing_no_overwrite')
    fd, info = regular(source)
    try:
        if hash_fd(fd) != expected_sha:
            raise ValueError('source does not match pinned SAFE digest')
        assert_unchanged(fd, info)
        if shutil.disk_usage(root).free < info.st_size + SPACE_BUFFER:
            raise ValueError('insufficient free space for complete copy plus 1 GiB reserve')
    finally:
        os.close(fd)
    # mkdir is exclusive: another invocation cannot silently overwrite this vault.
    target.mkdir(mode=0o700, exist_ok=False)
    part = target / 'resume.pt.part'
    size = copy_checked(source, part, expected_sha)
    os.rename(part, target / 'resume.pt')  # target was exclusively created above
    sync_directory(target)
    manifest = {'schema': 1, 'complete': True, 'label': label,
                'sha256': expected_sha, 'bytes': size, 'created_unix': int(time.time()),
                'note': 'verified byte copy only; no torch deserialization or model-quality claim'}
    temp_manifest = target / 'manifest.json.part'
    with temp_manifest.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.rename(temp_manifest, target / 'manifest.json')  # commit marker written LAST
    directory_synced = sync_directory(target)
    parent_synced = sync_directory(root)
    result = verify(target, expected_sha)
    return dict(result, status='protected_new_copy', directory_fsync=directory_synced,
                parent_fsync=parent_synced, distinct_volume_requested=different_volume)


def restore(backup, expected_sha, destination):
    verified = verify(backup, expected_sha)
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('restore refuses to overwrite any existing destination')
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise ValueError('restore destination parent must already exist and be real')
    tmpfd, tmpname = tempfile.mkstemp(prefix='.neral-restore-', dir=destination.parent)
    os.close(tmpfd)
    # Exclusive temporary file already belongs to us; unlink only this new scratch file.
    os.unlink(tmpname)
    size = copy_checked(Path(backup) / 'resume.pt', Path(tmpname), expected_sha)
    if size != verified['bytes']:
        raise ValueError('restore size mismatch')
    # Atomic hard-link refuses an existing destination without a TOCTOU overwrite.
    os.link(tmpname, destination, follow_symlinks=False)
    os.unlink(tmpname)
    sync_directory(destination.parent)
    actual, _ = hash_regular(destination)
    if actual != expected_sha:
        raise ValueError('restored file digest mismatch')
    return {'schema': 1, 'status': 'restored_no_overwrite', 'sha256': actual,
            'bytes': size, 'destination': str(destination)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('protect', 'verify', 'restore'))
    parser.add_argument('--source')
    parser.add_argument('--backup-root')
    parser.add_argument('--backup')
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--label')
    parser.add_argument('--destination')
    parser.add_argument('--different-volume', action='store_true')
    args = parser.parse_args()
    if args.action == 'protect':
        if not args.source or not args.backup_root or not args.label:
            parser.error('protect requires --source --backup-root --label')
        result = protect(args.source, args.backup_root, args.expected_sha, args.label,
                         different_volume=args.different_volume)
    elif args.action == 'verify':
        if not args.backup:
            parser.error('verify requires --backup')
        result = verify(args.backup, args.expected_sha)
    else:
        if not args.backup or not args.destination:
            parser.error('restore requires --backup --destination')
        result = restore(args.backup, args.expected_sha, args.destination)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()