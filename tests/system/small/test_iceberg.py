# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import bigframes.pandas as bpd


def test_read_iceberg_table():
    bpd.reset_session()
    bpd.config.options.bigquery.location = "us-central1"
    df = bpd.read_gbq(
        "bigquery-public-data.biglake-public-nyc-taxi-iceberg.public_data.nyc_taxicab_2021"
    )
    assert df.shape == (30904427, 20)
