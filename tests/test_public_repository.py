"""Integrity failures must not silently overwrite data or extract unsafe members."""
import io,tarfile,hashlib
from pathlib import Path
import pytest
from pat16.release_assets import bundle,restore,verify,digest,SplitWriter,JoinedReader


def test_multipart_roundtrip_and_changed_destination_protection(tmp_path):
    src=tmp_path/'source';(src/'data/nested').mkdir(parents=True)
    (src/'data/nested/example.bin').write_bytes(bytes(range(256))*30)
    (src/'data/nested/.python-version').write_text('3.12.14\n')
    (src/'data/a.txt').write_text('A real input with literal $ and unicode α\n')
    assets=tmp_path/'assets';spec=bundle(src,'data',assets,'input',limit=100)
    assert len(spec['parts'])>1
    dest=tmp_path/'clone';restore(dest,spec,assets);verify(dest,spec)
    restore(dest,spec,assets) # identical files are safe on repeated restoration
    (dest/'data/a.txt').write_text('local edit')
    with pytest.raises(ValueError,match='Modified destination'):restore(dest,spec,assets)
    assert (dest/'data/a.txt').read_text()=='local edit'
    restore(dest,spec,assets,overwrite=True);verify(dest,spec)
    part=assets/spec['parts'][0]['name'];part.write_bytes(part.read_bytes()+b'corruption')
    with pytest.raises(ValueError,match='checksum/size'):restore(dest,spec,assets)


def test_unexpected_traversal_symlink_and_duplicate_members_rejected(tmp_path):
    payload=b'expected';h=hashlib.sha256(payload).hexdigest()
    for case in ['traversal','symlink','duplicate','unexpected']:
        stream=io.BytesIO()
        with tarfile.open(fileobj=stream,mode='w:gz') as tar:
            names=['../escape'] if case=='traversal' else ['data/a','data/a'] if case=='duplicate' else ['data/other'] if case=='unexpected' else ['data/a']
            for name in names:
                info=tarfile.TarInfo(name);info.size=len(payload)
                if case=='symlink':info.type=tarfile.SYMTYPE;info.linkname='../escape';tar.addfile(info)
                else:tar.addfile(info,io.BytesIO(payload))
        asset=tmp_path/(case+'.gz');asset.write_bytes(stream.getvalue())
        spec=dict(prefix='data',members=[dict(path='data/a',bytes=len(payload),sha256=h)],parts=[dict(name=asset.name,bytes=asset.stat().st_size,sha256=digest(asset))])
        with pytest.raises(ValueError):restore(tmp_path/case,spec,tmp_path)
    assert not (tmp_path/'escape').exists()


def test_truncated_assets_and_symlink_destination_are_rejected(tmp_path):
    src=tmp_path/'src';(src/'data').mkdir(parents=True);(src/'data/a').write_text('input')
    spec=bundle(src,'data',tmp_path/'assets','x')
    dest=tmp_path/'dest';dest.mkdir();(dest/'data').symlink_to(src/'data',target_is_directory=True)
    with pytest.raises(ValueError,match='Symlink'):restore(dest,spec,tmp_path/'assets')
    part=tmp_path/'assets'/spec['parts'][0]['name'];part.write_bytes(part.read_bytes()[:20])
    with pytest.raises(ValueError):restore(tmp_path/'new',spec,tmp_path/'assets')
