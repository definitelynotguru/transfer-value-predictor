# Data source

Raw files are **not** committed. This manifest identifies the exact bytes used.

- Dataset: Transfermarkt community datasets by David Cariboo
  ([dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets),
  [Kaggle mirror](https://www.kaggle.com/datasets/davidcariboo/player-scores)).
- Underlying data: [Transfermarkt](https://www.transfermarkt.com/). All rights remain with
  Transfermarkt and the dataset maintainers.
- Base URL: `https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data`
- Snapshot note: Upstream collection paused; games end 2026-07-06, appearances end 2026-06-28.

The upstream URLs are mutable. A matching SHA-256 proves you have the same bytes; it does not
guarantee those bytes remain downloadable. Keep a private copy if exact reproduction matters.

| File | Bytes | SHA-256 | Acquired (UTC) | Upstream Last-Modified |
|------|------:|---------|----------------|------------------------|
| `appearances.csv.gz` | 45,253,638 | `e096c2b158fc3d52c4550526c4d4850dd62f31007652f4bcae112eadb1646802` | 2026-09-24T17:43:01Z | Sat, 05 Sep 2026 09:11:59 GMT |
| `clubs.csv.gz` | 51,281 | `aa57ab56e87faffaea1e8697439a32caf10116d8f2f2dfb3a57193c0bb1d641b` | 2026-09-24T17:43:15Z | Sat, 05 Sep 2026 09:12:01 GMT |
| `competitions.csv.gz` | 2,242 | `8924ddfbc0e9989f4a42a3c32ebb6faa671614625086d7fa84353f955352ea88` | 2026-09-24T17:43:05Z | Sat, 05 Sep 2026 09:12:02 GMT |
| `game_lineups.csv.gz` | 125,814,089 | `5e83d6fad28364aadfea608013700cf09990062eede1ad9ef189189b0af47ba2` | 2026-09-24T17:43:15Z | Sat, 05 Sep 2026 09:12:19 GMT |
| `games.csv.gz` | 4,995,595 | `142561989017d379bcf9e72bad0b87e0996bae0e7b710ebb3def9255f464c741` | 2026-09-24T17:43:05Z | Sat, 05 Sep 2026 09:12:20 GMT |
| `players.csv.gz` | 4,389,958 | `d22e407981d5b51a79bf8ff59835729f3526f7dc3495a6d8ed2f852ed2e86403` | 2026-09-24T17:43:03Z | Sat, 05 Sep 2026 09:12:22 GMT |
| `transfers.csv.gz` | 5,809,514 | `90326983daf7e6ac7aabdfe62b90936d9bf1dd2171e53dec75c3751eb1620a83` | 2026-09-24T17:43:02Z | Sat, 05 Sep 2026 09:12:23 GMT |

<!-- source-entries
[
  {
    "file": "appearances.csv.gz",
    "url": "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/appearances.csv.gz",
    "bytes": 45253638,
    "sha256": "e096c2b158fc3d52c4550526c4d4850dd62f31007652f4bcae112eadb1646802",
    "acquired_utc": "2026-09-24T17:43:01Z",
    "upstream_last_modified": "Sat, 05 Sep 2026 09:11:59 GMT",
    "upstream_etag": "e0444486c9d78a44a74544d7dfbcc8cc-6"
  },
  {
    "file": "clubs.csv.gz",
    "url": "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/clubs.csv.gz",
    "bytes": 51281,
    "sha256": "aa57ab56e87faffaea1e8697439a32caf10116d8f2f2dfb3a57193c0bb1d641b",
    "acquired_utc": "2026-09-24T17:43:15Z",
    "upstream_last_modified": "Sat, 05 Sep 2026 09:12:01 GMT",
    "upstream_etag": "f30713f2d0022ca4ecbb382cd695e389"
  },
  {
    "file": "competitions.csv.gz",
    "url": "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/competitions.csv.gz",
    "bytes": 2242,
    "sha256": "8924ddfbc0e9989f4a42a3c32ebb6faa671614625086d7fa84353f955352ea88",
    "acquired_utc": "2026-09-24T17:43:05Z",
    "upstream_last_modified": "Sat, 05 Sep 2026 09:12:02 GMT",
    "upstream_etag": "cc53fe35e2c82d73f57f3f5d3d8cf605"
  },
  {
    "file": "game_lineups.csv.gz",
    "url": "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/game_lineups.csv.gz",
    "bytes": 125814089,
    "sha256": "5e83d6fad28364aadfea608013700cf09990062eede1ad9ef189189b0af47ba2",
    "acquired_utc": "2026-09-24T17:43:15Z",
    "upstream_last_modified": "Sat, 05 Sep 2026 09:12:19 GMT",
    "upstream_etag": "9218250ea32f80483068fb4a2eb311b2-15"
  },
  {
    "file": "games.csv.gz",
    "url": "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/games.csv.gz",
    "bytes": 4995595,
    "sha256": "142561989017d379bcf9e72bad0b87e0996bae0e7b710ebb3def9255f464c741",
    "acquired_utc": "2026-09-24T17:43:05Z",
    "upstream_last_modified": "Sat, 05 Sep 2026 09:12:20 GMT",
    "upstream_etag": "9d56854283f5d47601688ffbd3baaedd"
  },
  {
    "file": "players.csv.gz",
    "url": "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/players.csv.gz",
    "bytes": 4389958,
    "sha256": "d22e407981d5b51a79bf8ff59835729f3526f7dc3495a6d8ed2f852ed2e86403",
    "acquired_utc": "2026-09-24T17:43:03Z",
    "upstream_last_modified": "Sat, 05 Sep 2026 09:12:22 GMT",
    "upstream_etag": "bf46ce79e2aec6db1decce7713bb618c"
  },
  {
    "file": "transfers.csv.gz",
    "url": "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/transfers.csv.gz",
    "bytes": 5809514,
    "sha256": "90326983daf7e6ac7aabdfe62b90936d9bf1dd2171e53dec75c3751eb1620a83",
    "acquired_utc": "2026-09-24T17:43:02Z",
    "upstream_last_modified": "Sat, 05 Sep 2026 09:12:23 GMT",
    "upstream_etag": "21cdd99be19205ed521890c45da44235"
  }
]
-->
