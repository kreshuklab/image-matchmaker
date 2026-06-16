# Example data

The repository provides example segmentation masks and example registration configs under:

```text
examples/
```

These files can be used to test the registration workflow and inspect the expected input formats.

---

## Generate synthetic moving data

Synthetic moving datasets can be generated from the provided example segmentation mask with:

```bash
python examples/deform_test_data.py
```

The script generates:

- rigidly transformed segmentations
- elastically deformed segmentations

Generated outputs are saved under:

```text
examples/data/deformed_data/
```
