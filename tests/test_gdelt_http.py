"""Local tests for the revised GDELT 2.1 HTTPS ingestion pipeline.

Place this file in the project's tests/test_gdelt_http.py and run pytest.
Requires: requests, boto3, pytest. No real HTTP or AWS operations are performed.
"""
import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest
import requests
from botocore.exceptions import ClientError

PROJECT = Path(__file__).resolve().parents[1] / 'ingestion'
SCRIPT = PROJECT / '06_ingest_gdelt.py'
RULES = PROJECT / 'gdelt_filter_rules_v1.json'
# When installed into a project: tests/../ingestion/
if not SCRIPT.exists():
    SCRIPT = PROJECT.parent / 'ingestion' / '06_ingest_gdelt.py'
    RULES = PROJECT.parent / 'ingestion' / 'gdelt_filter_rules_v1.json'
spec = importlib.util.spec_from_file_location('gdelt_ingest_under_test', SCRIPT)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


class MockS3:
    def __init__(self):
        self.objects = {}

    def upload_file(self, src, bucket, key, ExtraArgs=None):
        self.objects[(bucket, key)] = Path(src).read_bytes()

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objects[(Bucket, Key)] = Body

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise ClientError({'Error': {'Code': 'NoSuchKey', 'Message': 'not found'}}, 'GetObject')
        return {'Body': io.BytesIO(self.objects[(Bucket, Key)])}

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise ClientError({'Error': {'Code': '404', 'Message': 'not found'}}, 'HeadObject')
        return {'ContentLength': len(self.objects[(Bucket, Key)])}


class Response:
    def __init__(self, payload):
        self.payload = payload
        self.headers = {'Content-Length': str(len(payload))}

    def __enter__(self): return self
    def __exit__(self, *args): pass
    def raise_for_status(self): pass
    def iter_content(self, chunk_size):
        yield self.payload


def zipped_rows(rows):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('20250101120000.gkg.csv', '\n'.join('\t'.join(r) for r in rows) + '\n')
    return buffer.getvalue()


def gkg_row(url='https://www.reuters.com/economy/fomc/',
            domain='www.reuters.com',
            org='Federal Reserve',
            themes='ECON_STOCKMARKET;EPU_POLICY',
            person='',
            doc_id='20250101120000-1'):
    row = [''] * 27
    row[0] = doc_id
    row[1] = '20250101120000'
    row[2] = '1'
    row[3] = domain
    row[4] = url
    row[7] = themes
    row[9] = '1#Washington#US#USDC'
    row[11] = person
    row[13] = org
    row[15] = '1.1,2.2,3.3,4.4,5.5,6.6,700'
    return row


def shard_for(payload):
    return dict(stamp='20250101120000', eastern_date='2025-01-01',
                compressed_bytes=len(payload), md5=hashlib.md5(payload).hexdigest(),
                url='https://data.gdeltproject.org/gdeltv2/20250101120000.gkg.csv.zip')


def mock_download(monkeypatch, payload):
    monkeypatch.setattr(g.requests, 'get', lambda url, **kwargs: Response(payload))


def setup():
    rules, rule_hash = g.load_rules(RULES)
    return MockS3(), rules, rule_hash


def test_domain_suffix_not_substring():
    assert g.allowed_domain('www.reuters.com', ['reuters.com'])
    assert g.allowed_domain('uk.reuters.com', ['reuters.com'])
    assert not g.allowed_domain('notreuters.com', ['reuters.com'])
    assert not g.allowed_domain('apnews.evil.test', ['apnews.com'])
    assert not g.allowed_domain('https://footprints.example', ['ft.com'])


def test_27_column_metadata_and_direct_broad_flags():
    s3, rules, rh = setup()
    row = gkg_row()
    doc = g.record_from_row(row, shard_for(b'fake'), rules, rh)
    assert doc['direct_fed'] is True and doc['broad_macro'] is True
    assert doc['representation_type'] == 'gkg_themes_orgs_persons'
    assert doc['doc_time_utc'] == '2025-01-01T12:00:00Z'
    assert doc['doc_date'] == '2025-01-01'
    assert doc['gkg_tone']['word_count'] == 700
    with pytest.raises(ValueError, match='27'):
        g.record_from_row(row[:-1], shard_for(b'fake'), rules, rh)
    row2 = gkg_row(url='https://www.reuters.com/business/oil', org='',
                   themes='ENV_OIL', doc_id='20250101120000-2')
    doc2 = g.record_from_row(row2, shard_for(b'fake'), rules, rh)
    assert not doc2['direct_fed'] and doc2['broad_macro']


def test_success_and_checkpoint(monkeypatch):
    s3, rules, rh = setup()
    payload = zipped_rows([gkg_row(), gkg_row(doc_id='20250101120000-2',
        url='https://reuters.com/oil', org='', themes='ENV_OIL')])
    shard = shard_for(payload)
    mock_download(monkeypatch, payload)
    got = g.process_shard(shard, s3, 'bucket', 'pilot/raw/gdelt', 'pilot/manifests', rules, rh)
    assert got['status'] == 'success' and got['md5_ok']
    assert got['parsed_rows'] == 2 and got['matched_rows'] == 2
    assert got['direct_rows'] == 1 and got['broad_rows'] == 2
    assert g.checkpoint_valid(s3, 'bucket', 'pilot/manifests', shard, rh)
    rows, key = g.write_run_manifest(s3, 'bucket', 'pilot/manifests', [shard])
    assert len(rows) == 1 and rows[0]['status'] == 'success'
    assert ('bucket', key) in s3.objects


def test_zero_match_is_empty_and_restartable(monkeypatch):
    s3, rules, rh = setup()
    payload = zipped_rows([gkg_row(domain='irrelevant.example')])
    shard = shard_for(payload)
    mock_download(monkeypatch, payload)
    got = g.process_shard(shard, s3, 'bucket', 'pilot/raw/gdelt', 'pilot/manifests', rules, rh)
    assert got['status'] == 'empty' and got['md5_ok'] and got['parsed_rows'] == 1
    assert not got['output_keys']
    assert g.checkpoint_valid(s3, 'bucket', 'pilot/manifests', shard, rh)


def test_md5_mismatch_is_failure_not_empty(monkeypatch):
    s3, rules, rh = setup()
    payload = zipped_rows([gkg_row()])
    shard = shard_for(payload)
    shard['md5'] = '0' * 32
    mock_download(monkeypatch, payload)
    monkeypatch.setattr(g.time, 'sleep', lambda _: None)
    got = g.process_shard(shard, s3, 'bucket', 'pilot/raw/gdelt', 'pilot/manifests', rules, rh)
    assert got['status'] == 'failed' and not got['md5_ok']
    assert not g.checkpoint_valid(s3, 'bucket', 'pilot/manifests', shard, rh)


def test_bad_zip_is_failure_not_empty(monkeypatch):
    s3, rules, rh = setup()
    payload = b'not a zip'
    shard = shard_for(payload)
    mock_download(monkeypatch, payload)
    got = g.process_shard(shard, s3, 'bucket', 'pilot/raw/gdelt', 'pilot/manifests', rules, rh)
    assert got['status'] == 'failed' and got['md5_ok']


def test_network_failure_is_failure_not_empty(monkeypatch):
    s3, rules, rh = setup()
    payload = zipped_rows([gkg_row()])
    shard = shard_for(payload)
    def fail(*args, **kwargs):
        raise requests.ConnectionError('offline')
    monkeypatch.setattr(g.requests, 'get', fail)
    monkeypatch.setattr(g.time, 'sleep', lambda _: None)
    got = g.process_shard(shard, s3, 'bucket', 'pilot/raw/gdelt', 'pilot/manifests', rules, rh)
    assert got['status'] == 'failed' and got['md5_ok'] is False


def test_plan_integer_stamp_and_eastern_date(tmp_path):
    payload = b'abc'
    shard = shard_for(payload)
    plan = tmp_path / 'plan.csv'
    plan.write_text('stamp,eastern_date,compressed_bytes,md5,url\n' +
                    ','.join([shard[k] for k in ('stamp','eastern_date')]) + ',' +
                    str(shard['compressed_bytes']) + ',' + shard['md5'] + ',' + shard['url'] + '\n')
    assert g.validated_plan(plan)[0]['stamp'] == '20250101120000'
