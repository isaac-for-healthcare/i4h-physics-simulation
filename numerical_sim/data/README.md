# Production assets

The meshes, textures, centrelines and USD files the cases use are not stored in Git. `docs/asset-manifest.json` lists
each file with its SHA-256 and size; `scripts/fetch_assets.py` fills this folder (or `$I4H_DATA`) and checks every
file against it:

```bash
uv run python scripts/fetch_assets.py --source /path/to/numerical_sim_data   # a local copy of the tree, or a base URL
uv run python scripts/fetch_assets.py --check                                  # verify what is here
```

Publication is pending: the assets are to be released through the Isaac for Healthcare asset catalog
(`omniverse-content-production` S3, `Assets/Isaac/Healthcare/<version>/<id>/`), once each manifest entry has an
approved source, owner and license. Until then, `--source` is a local copy from the maintainers.
