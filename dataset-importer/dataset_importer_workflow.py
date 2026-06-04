"""Import Hugging Face datasets into Mixtrain.

Create the workflow from this directory:

    mixtrain workflow create . --name dataset-importer

Example runs:

    mixtrain workflow run dataset-importer --input '{"hf_dataset": "imdb", "target_name": "hf-imdb"}'

    mixtrain workflow run dataset-importer --input '{
      "hf_dataset": "OneEyeDJ/Art-Vision-Question-Answering-Dataset",
      "target_name": "hf-art-vqa-test"
    }'

"""

from mixtrain import MixFlow, Dataset, validate_resource_name


class DatasetImporter(MixFlow):
    """Import a Hugging Face dataset into Mixtrain."""

    def run(
        self,
        hf_dataset: str,
        target_name: str,
        split: str = "train",
        column_types: dict[str, str] | str = "auto",
        description: str | None = None,
        overwrite: bool = False,
    ) -> Dataset:
        """Run the import."""
        from utils import column_types_for_save

        validate_resource_name(target_name, "dataset")

        if Dataset.exists(target_name):
            if overwrite:
                print(f"Dataset {target_name!r} already exists — replacing")
                Dataset(target_name).delete()
            else:
                raise ValueError(
                    f"Dataset {target_name!r} already exists. "
                    "Set overwrite=True to replace it, or choose a different name."
                )

        print(f"Loading {hf_dataset} split={split} from Hugging Face")

        ds = Dataset.from_huggingface(hf_dataset, split=split, name=hf_dataset)
        # Apply dataset transforms here if needed, for example ds.select(...),
        # ds.filter(...), or ds.map(...).

        if column_types != "auto":
            column_types = column_types_for_save(column_types)

        source = f"hf://{hf_dataset} split={split}"
        dataset_description = description or f"Imported from {source}"

        print(f"Saving as dataset {target_name}")

        ds.save(
            name=target_name,
            description=dataset_description,
            column_types=column_types,
        )

        return Dataset(target_name)
