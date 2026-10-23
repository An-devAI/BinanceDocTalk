"""Exercise the optional real Parquet reader on original fixtures; no network or model calls."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
from import_huggingface import parquet_rows


def main():
    rows=[{'url':'https://example.invalid/fixture-1','markdown':'ORIGINAL UNIT FIXTURE ONE'},
          {'url':'https://example.invalid/fixture-2','markdown':'ORIGINAL UNIT FIXTURE TWO'}]
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'fixture.parquet'
        pq.write_table(pa.Table.from_pylist(rows),path)
        spec={'rows':2,'columns':['url','markdown']}
        assert parquet_rows(path,spec)==rows
        try:
            parquet_rows(path,dict(spec,rows=3))
        except ValueError:
            pass
        else:
            raise AssertionError('The real reader accepted the wrong row count')
    print(json.dumps({'pyarrow':pa.__version__,'original_fixture_roundtrip':True,
                      'wrong_row_count_rejected':True,'network_calls':0,'model_calls':0}))


if __name__=='__main__':main()
