# Raw graph datasets

This directory contains the seven graph datasets from the paper, organized by
graph direction. The files are currently stored in decompressed form.

## Dataset summary

| Dataset | File | Nodes | Edges | Type | Storage notes |
| --- | --- | ---: | ---: | --- | --- |
| Email | `directed/email-Eu-core.txt` | 1,005 | 25,571 | Directed | Two-column edge list |
| Bitcoin | `directed/soc-sign-bitcoinotc.csv` | 5,881 | 35,592 | Directed | Signed, weighted, temporal edge list |
| LastFM | `undirected/lasftm_asia/lastfm_asia_edges.csv` | 7,624 | 27,806 | Undirected | Includes separate feature and target files |
| HepPh | `undirected/ca-HepPh.txt` | 12,008 | 118,521 | Undirected | Stored as 237,010 directed rows, including 32 self-loops |
| Facebook | `undirected/facebook_edges.csv` | 22,470 | 171,002 | Undirected | Two-column CSV edge list |
| Gowalla | `undirected/loc-gowalla_edges.txt` | 196,591 | 950,327 | Undirected | Stored as 1,900,654 directed rows |
| Friendster | `undirected/com-Friendster/com-Friendster.mtx` | 65,608,366 | 1,806,067,135 | Undirected | Symmetric Matrix Market pattern matrix |

## Directory structure

```text
raw_dataset/
├── directed/
│   ├── email-Eu-core.txt
│   └── soc-sign-bitcoinotc.csv
└── undirected/
    ├── ca-HepPh.txt
    ├── facebook_edges.csv
    ├── loc-gowalla_edges.txt
    ├── lasftm_asia/
    │   ├── lastfm_asia_edges.csv
    │   ├── lastfm_asia_features.json
    │   ├── lastfm_asia_target.csv
    │   └── README.txt
    └── com-Friendster/
        ├── com-Friendster.mtx
        ├── com-Friendster_nodeid.mtx
        ├── com-Friendster_Communities_all.mtx
        └── com-Friendster_Communities_top5000.mtx
```

The `lasftm_asia` directory spelling comes from the original LastFM archive.

## Sources

- Email: <https://snap.stanford.edu/data/email-Eu-core.html>
- Bitcoin OTC: <https://snap.stanford.edu/data/soc-sign-bitcoinotc.html>
- LastFM Asia: <https://snap.stanford.edu/data/feather-lastfm-social.html>
- HepPh: <https://snap.stanford.edu/data/ca-HepPh.html>
- Facebook Large Page-Page Network:
  <https://snap.stanford.edu/data/facebook-large-page-page-network.html>
  - SNAP currently marks the data file as unavailable. The edge list is from
    the dataset authors' MUSAE repository:
    <https://github.com/benedekrozemberczki/MUSAE/blob/master/input/edges/facebook_edges.csv>
- Gowalla: <https://snap.stanford.edu/data/loc-gowalla.html>
- Friendster: <https://sparse.tamu.edu/SNAP/com-Friendster>

Email, Bitcoin, LastFM, HepPh, and Gowalla were downloaded from SNAP and then
decompressed without changing their contents. Facebook is the original CSV from
the dataset authors' repository. Friendster is the extracted SuiteSparse archive
of the same SNAP graph; it uses Matrix Market format rather than SNAP's text
edge-list format.
